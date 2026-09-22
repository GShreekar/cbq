use std::fs;
use std::path::{Path, PathBuf};
use crate::config::settings::cbq_home;
use crate::services::hashing::fnv1a_hash;

const INDEX_FILE_NAME: &str = "db.sqlite";

/// A project directory together with the index database that covers it.
pub struct IndexedProject {
    pub root: PathBuf,
    pub db_path: PathBuf,
}

/// Resolves a directory to the absolute path that identifies it as a project.
pub fn canonical_project_root(path: &Path) -> Result<PathBuf, anyhow::Error> {
    fs::canonicalize(path).map_err(|err| anyhow::anyhow!("Cannot open directory '{}': {}", path.display(), err))
}

/// Returns where a project's index database lives, without creating anything on disk.
pub fn index_path_for(project_root: &Path) -> Result<PathBuf, anyhow::Error> {
    Ok(codebases_dir()?.join(project_key(project_root)).join(INDEX_FILE_NAME))
}

/// Finds the nearest directory at or above `start_dir` that has been indexed.
pub fn find_indexed_project(start_dir: &Path) -> Result<Option<IndexedProject>, anyhow::Error> {
    let start = canonical_project_root(start_dir)?;
    for dir in start.ancestors() {
        let db_path = index_path_for(dir)?;
        if db_path.exists() {
            return Ok(Some(IndexedProject { root: dir.to_path_buf(), db_path }));
        }
    }
    Ok(None)
}

/// Finds an index from older cbq versions, which named index directories after the project folder alone.
pub fn find_legacy_index(start_dir: &Path) -> Result<Option<PathBuf>, anyhow::Error> {
    let start = canonical_project_root(start_dir)?;
    let codebases = codebases_dir()?;
    for dir in start.ancestors() {
        let Some(name) = dir.file_name() else { continue };
        let legacy_dir = codebases.join(name);
        if legacy_dir.join(INDEX_FILE_NAME).exists() {
            return Ok(Some(legacy_dir));
        }
    }
    Ok(None)
}

/// Lists the directory of every index cbq has built.
pub fn index_directories() -> Result<Vec<PathBuf>, anyhow::Error> {
    let codebases = codebases_dir()?;
    if !codebases.exists() {
        return Ok(Vec::new());
    }
    let mut directories: Vec<PathBuf> = fs::read_dir(codebases)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.join(INDEX_FILE_NAME).exists())
        .collect();
    directories.sort();
    Ok(directories)
}

/// Writes a relative path the way the index stores it: separated by forward slashes on every
/// platform, so a path typed with `/` finds a file the walker reported with `\`.
pub fn stored_path(relative: &Path) -> String {
    relative
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// Returns the index database inside an index directory.
pub fn index_database_in(index_dir: &Path) -> PathBuf {
    index_dir.join(INDEX_FILE_NAME)
}

fn codebases_dir() -> Result<PathBuf, anyhow::Error> {
    Ok(cbq_home()?.join("codebases"))
}

// The path hash keeps same-named projects in different places from sharing one index.
fn project_key(project_root: &Path) -> String {
    let name = project_root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".to_string());
    format!("{}-{:016x}", name, fnv1a_hash(project_root.as_os_str().as_encoded_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_named_projects_get_different_keys() {
        assert_ne!(project_key(Path::new("/work/api")), project_key(Path::new("/oss/api")));
    }

    #[test]
    fn project_key_starts_with_the_folder_name() {
        assert!(project_key(Path::new("/home/dev/cbq")).starts_with("cbq-"));
    }

    #[test]
    fn a_stored_path_is_separated_by_forward_slashes() {
        assert_eq!(stored_path(Path::new("src/services/cart.rs")), "src/services/cart.rs");
    }

    #[test]
    fn a_single_file_is_stored_unchanged() {
        assert_eq!(stored_path(Path::new("README.md")), "README.md");
    }

    // On Windows the walker reports `src\lib.rs`, while a person types `src/lib.rs`. Both are the
    // same file, so both have to reach the index as the same string.
    #[cfg(windows)]
    #[test]
    fn both_separators_reach_the_index_as_one_path() {
        assert_eq!(stored_path(Path::new("src\\lib.rs")), stored_path(Path::new("src/lib.rs")));
    }

    #[test]
    fn filesystem_root_gets_a_readable_key() {
        assert!(project_key(Path::new("/")).starts_with("root-"));
    }
}
