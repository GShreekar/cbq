use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::db::index_metadata::{read_embedding_model, read_meta, PROJECT_ROOT_KEY};
use crate::db::location::{index_database_in, index_directories};
use crate::db::queries::file_lengths;
use crate::db::schema::open_index;
use crate::services::vector_search::SearchResult;

// Matches the number of called definitions a single-project search adds, so a workspace
// search doesn't fill up with supporting context from every project at once.
const MAX_SUPPORTING_RESULTS: usize = 2;

/// An indexed project a workspace search can draw on.
pub struct WorkspaceIndex {
    pub project_root: PathBuf,
    pub db_path: PathBuf,
}

/// An index left out of a workspace search, and the reason it can't take part.
pub struct SkippedIndex {
    pub project_root: String,
    pub reason: String,
}

/// Sorts every index cbq has built into those that can answer this search and those that can't.
pub fn collect_indexes(embedding_model: &str) -> Result<(Vec<WorkspaceIndex>, Vec<SkippedIndex>), anyhow::Error> {
    let mut usable = Vec::new();
    let mut skipped = Vec::new();

    for index_dir in index_directories()? {
        let db_path = index_database_in(&index_dir);
        let connection = match open_index(&db_path) {
            Ok(connection) => connection,
            Err(err) => {
                skipped.push(SkippedIndex { project_root: index_dir.to_string_lossy().into_owned(), reason: err.to_string() });
                continue;
            }
        };
        let Some(project_root) = read_meta(&connection, PROJECT_ROOT_KEY)? else {
            // Indexes built before cbq recorded the project path can't have their results labelled.
            skipped.push(SkippedIndex {
                project_root: index_dir.to_string_lossy().into_owned(),
                reason: "built before cbq recorded project paths; re-index it".to_string(),
            });
            continue;
        };

        match read_embedding_model(&connection)? {
            // Vectors from different models mean different things, so their scores can't be compared.
            Some(model) if model != embedding_model => skipped.push(SkippedIndex {
                project_root,
                reason: format!("built with '{}', not '{}'", model, embedding_model),
            }),
            Some(_) => usable.push(WorkspaceIndex { project_root: PathBuf::from(project_root), db_path }),
            None => skipped.push(SkippedIndex { project_root, reason: "no embedding model recorded; re-index it".to_string() }),
        }
    }
    Ok((usable, skipped))
}

/// Rewrites a result's path to the full path on disk, so merged results say which project they came from.
pub fn qualify(result: &mut SearchResult, project_root: &Path) {
    result.chunk.file_path = project_root.join(&result.chunk.file_path);
}

/// Merges results from several projects into one ranking, best score first. Definitions pulled in
/// through call graphs stay behind the matches, as they do within a single project.
pub fn merge(results: Vec<SearchResult>, limit: usize) -> Vec<SearchResult> {
    let (mut matches, mut supporting): (Vec<_>, Vec<_>) =
        results.into_iter().partition(|result| !result.found_via_calls);

    for group in [&mut matches, &mut supporting] {
        group.sort_by(|left, right| right.score.total_cmp(&left.score));
    }
    matches.truncate(limit);
    supporting.truncate(MAX_SUPPORTING_RESULTS);
    matches.extend(supporting);
    matches
}

/// The last line each file was seen to have, keyed by full path so citations can be checked
/// against whichever project they name.
pub fn qualified_file_lengths(
    connection: &rusqlite::Connection,
    project_root: &Path,
) -> Result<HashMap<String, usize>, anyhow::Error> {
    Ok(file_lengths(connection)?
        .into_iter()
        .map(|(path, last_line)| (project_root.join(path).to_string_lossy().into_owned(), last_line))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::chunker::CodeChunk;

    fn result(name: &str, score: f64, found_via_calls: bool) -> SearchResult {
        SearchResult {
            chunk: CodeChunk {
                file_path: PathBuf::from(format!("src/{}.rs", name)),
                language: "rust".to_string(),
                name: name.to_string(),
                chunk_type: "function".to_string(),
                parent: None,
                content: String::new(),
                start_line: 1,
                end_line: 3,
            },
            score,
            matched_keywords: false,
            found_via_calls,
        }
    }

    fn names(results: &[SearchResult]) -> Vec<String> {
        results.iter().map(|result| result.chunk.name.clone()).collect()
    }

    #[test]
    fn merged_results_are_ordered_by_score_across_projects() {
        let merged = merge(vec![result("low", 0.2, false), result("high", 0.9, false), result("mid", 0.5, false)], 5);
        assert_eq!(names(&merged), vec!["high", "mid", "low"]);
    }

    #[test]
    fn supporting_definitions_stay_behind_the_matches() {
        let merged = merge(vec![result("called", 0.99, true), result("match", 0.3, false)], 5);
        assert_eq!(names(&merged), vec!["match", "called"]);
    }

    #[test]
    fn the_limit_counts_matches_not_supporting_definitions() {
        let merged = merge(vec![result("first", 0.9, false), result("second", 0.8, false), result("called", 0.7, true)], 1);
        assert_eq!(names(&merged), vec!["first", "called"]);
    }

    #[test]
    fn only_a_couple_of_supporting_definitions_survive_the_merge() {
        let supporting: Vec<SearchResult> = (0..6).map(|n| result(&format!("called{}", n), 0.5, true)).collect();
        assert_eq!(merge(supporting, 5).len(), MAX_SUPPORTING_RESULTS);
    }

    #[test]
    fn a_qualified_path_names_the_project_it_came_from() {
        let mut found = result("cart", 0.5, false);
        qualify(&mut found, Path::new("/work/shop"));
        assert_eq!(found.chunk.file_path, PathBuf::from("/work/shop/src/cart.rs"));
    }
}
