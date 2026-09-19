use std::fs;
use std::path::{Path, PathBuf};

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

fn codebases_dir() -> Result<PathBuf, anyhow::Error> {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| anyhow::anyhow!("Could not determine the home directory"))?;
    Ok(home_dir.join(".cbq").join("codebases"))
}

// The path hash keeps same-named projects in different places from sharing one index.
fn project_key(project_root: &Path) -> String {
    let name = project_root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".to_string());
    format!("{}-{:016x}", name, fnv1a_hash(project_root.as_os_str().as_encoded_bytes()))
}

// std's hasher may change between Rust releases, which would orphan every index on disk.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_hash_matches_the_published_test_vector() {
        assert_eq!(fnv1a_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
    }

    #[test]
    fn same_named_projects_get_different_keys() {
        assert_ne!(project_key(Path::new("/work/api")), project_key(Path::new("/oss/api")));
    }

    #[test]
    fn project_key_starts_with_the_folder_name() {
        assert!(project_key(Path::new("/home/dev/cbq")).starts_with("cbq-"));
    }

    #[test]
    fn filesystem_root_gets_a_readable_key() {
        assert!(project_key(Path::new("/")).starts_with("root-"));
    }
}
