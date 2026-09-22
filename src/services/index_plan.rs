use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use crate::services::file_discovery::{SkipReason, SkippedFile};
use crate::services::hashing::fnv1a_hash;
use crate::services::secrets::find_secret;

/// A project file read from disk, identified by its path relative to the project root.
pub struct SourceFile {
    pub absolute_path: PathBuf,
    pub relative_path: String,
    pub content: String,
    pub content_hash: String,
}

/// The work an index run has to do: files to embed, and files to drop from the index.
pub struct IndexPlan {
    pub new_files: Vec<SourceFile>,
    pub changed_files: Vec<SourceFile>,
    // Kept, not counted: the call graph is built from every file, not only the ones being embedded.
    pub unchanged_files: Vec<SourceFile>,
    pub removed_paths: Vec<String>,
}

/// Reads a file and hashes its content, failing for files that aren't UTF-8 text.
pub fn read_source_file(absolute_path: &Path, project_root: &Path) -> Result<SourceFile, anyhow::Error> {
    let bytes = fs::read(absolute_path)?;
    let content_hash = format!("{:016x}", fnv1a_hash(&bytes));
    let content = String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("not UTF-8 text"))?;
    let relative_path = absolute_path
        .strip_prefix(project_root)
        .unwrap_or(absolute_path)
        .to_string_lossy()
        .into_owned();
    Ok(SourceFile {
        absolute_path: absolute_path.to_path_buf(),
        relative_path,
        content,
        content_hash,
    })
}

// Reading and hashing each file is independent work, so it runs across cores.
/// Reads and hashes every file, leaving out any that hold credentials.
pub fn read_source_files(
    paths: &[PathBuf],
    project_root: &Path,
    allow_secrets: bool,
) -> (Vec<SourceFile>, Vec<SkippedFile>) {
    let read: Vec<Result<SourceFile, SkippedFile>> = paths
        .par_iter()
        .map(|path| {
            let source_file = read_source_file(path, project_root).map_err(|err| SkippedFile {
                path: path.clone(),
                reason: SkipReason::Unreadable(err.to_string()),
            })?;
            // A credential pasted into a source file would otherwise be embedded and shown in answers.
            match find_secret(&source_file.content) {
                Some(secret) if !allow_secrets => Err(SkippedFile {
                    path: path.clone(),
                    reason: SkipReason::LooksLikeSecret(secret.to_string()),
                }),
                _ => Ok(source_file),
            }
        })
        .collect();

    let mut source_files = Vec::new();
    let mut unreadable_files = Vec::new();
    for outcome in read {
        match outcome {
            Ok(source_file) => source_files.push(source_file),
            Err(skipped) => unreadable_files.push(skipped),
        }
    }
    (source_files, unreadable_files)
}

/// Compares the files on disk with the content hashes recorded by the last index run.
pub fn plan_index(files: Vec<SourceFile>, recorded_hashes: &HashMap<String, Option<String>>) -> IndexPlan {
    let current_paths: HashSet<&str> = files.iter().map(|file| file.relative_path.as_str()).collect();
    let mut removed_paths: Vec<String> = recorded_hashes
        .keys()
        .filter(|path| !current_paths.contains(path.as_str()))
        .cloned()
        .collect();
    removed_paths.sort();

    let mut plan = IndexPlan {
        new_files: Vec::new(),
        changed_files: Vec::new(),
        unchanged_files: Vec::new(),
        removed_paths,
    };
    for file in files {
        match recorded_hashes.get(&file.relative_path) {
            None => plan.new_files.push(file),
            Some(Some(hash)) if *hash == file.content_hash => plan.unchanged_files.push(file),
            Some(_) => plan.changed_files.push(file),
        }
    }
    plan
}

impl IndexPlan {
    /// Every file that needs embedding, new ones first.
    pub fn files_to_index(&self) -> impl Iterator<Item = &SourceFile> {
        self.new_files.iter().chain(self.changed_files.iter())
    }

    /// Every file in the project, whether or not it needs embedding.
    pub fn all_files(&self) -> impl Iterator<Item = &SourceFile> {
        self.files_to_index().chain(self.unchanged_files.iter())
    }

    pub fn unchanged_count(&self) -> usize {
        self.unchanged_files.len()
    }

    /// Reports whether the index already matches the files on disk.
    pub fn is_up_to_date(&self) -> bool {
        self.new_files.is_empty() && self.changed_files.is_empty() && self.removed_paths.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_file(path: &str, hash: &str) -> SourceFile {
        SourceFile {
            absolute_path: PathBuf::from("/project").join(path),
            relative_path: path.to_string(),
            content: String::new(),
            content_hash: hash.to_string(),
        }
    }

    fn recorded(entries: &[(&str, Option<&str>)]) -> HashMap<String, Option<String>> {
        entries.iter().map(|(path, hash)| (path.to_string(), hash.map(str::to_string))).collect()
    }

    #[test]
    fn file_with_same_hash_is_unchanged() {
        let plan = plan_index(vec![source_file("src/lib.rs", "aaa")], &recorded(&[("src/lib.rs", Some("aaa"))]));
        assert_eq!(plan.unchanged_count(), 1);
    }

    #[test]
    fn file_with_different_hash_is_changed() {
        let plan = plan_index(vec![source_file("src/lib.rs", "bbb")], &recorded(&[("src/lib.rs", Some("aaa"))]));
        assert_eq!(plan.changed_files.len(), 1);
    }

    #[test]
    fn unrecorded_file_is_new() {
        let plan = plan_index(vec![source_file("src/new.rs", "aaa")], &recorded(&[]));
        assert_eq!(plan.new_files.len(), 1);
    }

    #[test]
    fn recorded_file_missing_from_disk_is_removed() {
        let plan = plan_index(Vec::new(), &recorded(&[("src/gone.rs", Some("aaa"))]));
        assert_eq!(plan.removed_paths, vec!["src/gone.rs".to_string()]);
    }

    #[test]
    fn incompletely_indexed_file_is_retried() {
        let plan = plan_index(vec![source_file("src/lib.rs", "aaa")], &recorded(&[("src/lib.rs", None)]));
        assert_eq!(plan.changed_files.len(), 1);
    }

    #[test]
    fn matching_index_is_up_to_date() {
        let plan = plan_index(vec![source_file("src/lib.rs", "aaa")], &recorded(&[("src/lib.rs", Some("aaa"))]));
        assert!(plan.is_up_to_date());
    }
}
