use std::collections::HashMap;
use rusqlite::Connection;
use crate::db::queries::keyword_matches;
use crate::services::vector_search::{score_all_chunks, SearchResult};

// Reciprocal Rank Fusion, with the constant from Cormack et al. (2009): a chunk's score is the sum
// of 1/(k + rank) over the rankings it appears in, so agreement between methods beats any single one.
const RANK_FUSION_CONSTANT: f64 = 60.0;
// How many candidates each method contributes before the two rankings are merged.
const CANDIDATES_PER_METHOD: usize = 50;
// One- and two-letter words match nearly everything, so they only dilute the keyword ranking.
const MIN_KEYWORD_LENGTH: usize = 3;
// Words that carry no signal about which code is wanted, and would otherwise match every file of prose.
const STOP_WORDS: [&str; 30] = [
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "does", "did", "how", "what",
    "when", "where", "which", "who", "why", "this", "that", "these", "those", "with", "from", "was",
    "were", "into", "its", "use",
];

/// Finds the chunks most relevant to a question, combining embedding similarity with keyword matching.
pub fn find_relevant_chunks(
    conn: &Connection,
    query_text: &str,
    query_vector: &[f32],
    limit: usize,
) -> Result<Vec<SearchResult>, anyhow::Error> {
    let scored = score_all_chunks(conn, query_vector)?;

    let mut by_similarity: Vec<(i64, f64)> = scored.iter().map(|chunk| (chunk.id, chunk.similarity)).collect();
    by_similarity.sort_by(|first, second| second.1.total_cmp(&first.1));
    let similar_ids: Vec<i64> = by_similarity.iter().take(CANDIDATES_PER_METHOD).map(|(id, _)| *id).collect();

    let keyword_ids = match to_keyword_query(query_text) {
        Some(keyword_query) => keyword_matches(conn, &keyword_query, CANDIDATES_PER_METHOD)?,
        None => Vec::new(),
    };

    let mut chunks_by_id: HashMap<i64, _> = scored.into_iter().map(|chunk| (chunk.id, chunk)).collect();
    Ok(fuse_rankings(&[&similar_ids, &keyword_ids], limit)
        .into_iter()
        .filter_map(|id| Some((id, chunks_by_id.remove(&id)?)))
        // The score stays the embedding similarity, which is comparable across queries; the ranking is fused.
        .map(|(id, chunk)| SearchResult {
            chunk: chunk.chunk,
            score: chunk.similarity,
            matched_keywords: keyword_ids.contains(&id),
        })
        .collect())
}

/// Merges rankings so that a chunk ranked well by either method rises, and one ranked well by both rises further.
pub fn fuse_rankings(rankings: &[&[i64]], limit: usize) -> Vec<i64> {
    let mut scores: Vec<(i64, f64)> = Vec::new();
    let mut position_of: HashMap<i64, usize> = HashMap::new();

    for ranking in rankings {
        for (index, id) in ranking.iter().enumerate() {
            let contribution = 1.0 / (RANK_FUSION_CONSTANT + (index + 1) as f64);
            match position_of.get(id) {
                Some(position) => scores[*position].1 += contribution,
                None => {
                    position_of.insert(*id, scores.len());
                    scores.push((*id, contribution));
                }
            }
        }
    }

    // A stable sort leaves chunks with equal scores in the order the first ranking found them.
    scores.sort_by(|first, second| second.1.total_cmp(&first.1));
    scores.into_iter().take(limit).map(|(id, _)| id).collect()
}

/// Turns a question into an FTS5 query, as quoted words so punctuation can't be read as query syntax.
pub fn to_keyword_query(text: &str) -> Option<String> {
    let words: Vec<String> = text
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .filter(|word| word.chars().count() >= MIN_KEYWORD_LENGTH)
        .filter(|word| !STOP_WORDS.contains(&word.to_lowercase().as_str()))
        .map(|word| format!("\"{}\"", word))
        .collect();
    (!words.is_empty()).then(|| words.join(" OR "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chunk_both_methods_rank_first_wins() {
        let fused = fuse_rankings(&[&[7, 1, 2], &[7, 3, 4]], 3);
        assert_eq!(fused[0], 7);
    }

    #[test]
    fn a_chunk_only_keyword_search_found_still_appears() {
        let fused = fuse_rankings(&[&[1, 2], &[9]], 3);
        assert!(fused.contains(&9));
    }

    #[test]
    fn one_empty_ranking_leaves_the_other_order_intact() {
        assert_eq!(fuse_rankings(&[&[1, 2, 3], &[]], 3), vec![1, 2, 3]);
    }

    #[test]
    fn fusion_returns_at_most_the_requested_number() {
        assert_eq!(fuse_rankings(&[&[1, 2, 3], &[4, 5, 6]], 2).len(), 2);
    }

    #[test]
    fn equal_scores_keep_the_first_rankings_order() {
        assert_eq!(fuse_rankings(&[&[5, 6], &[5, 6]], 2), vec![5, 6]);
    }

    #[test]
    fn punctuation_never_reaches_the_query_syntax() {
        assert_eq!(to_keyword_query("show compute_total() please?"), Some("\"show\" OR \"compute_total\" OR \"please\"".to_string()));
    }

    #[test]
    fn question_words_are_left_out() {
        assert_eq!(to_keyword_query("what does this do"), None);
    }

    #[test]
    fn short_words_are_left_out() {
        assert_eq!(to_keyword_query("is it a b c"), None);
    }

    #[test]
    fn a_query_of_only_punctuation_has_no_keywords() {
        assert_eq!(to_keyword_query("?! ... ()"), None);
    }
}
