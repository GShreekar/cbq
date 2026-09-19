use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ignore::WalkBuilder;
use crate::services::parser::extension_to_language_name;

// Indexed as plain line windows because they have no grammar in the parser.
const PLAIN_TEXT_EXTENSIONS: [&str; 1] = ["md"];
// Machine-generated and huge; they would bury real code in search results.
const LOCKFILE_NAMES: [&str; 2] = ["package-lock.json", "pnpm-lock.yaml"];

pub struct FileDiscoveryResult {
    pub files: Vec<PathBuf>,
    pub extension_counts: HashMap<String, usize>,
}

pub fn discover_files(root_path: &Path) -> Result<FileDiscoveryResult, anyhow::Error> {
    let mut files = Vec::new();
    let mut extension_counts = HashMap::new();

    let walker = WalkBuilder::new(root_path)
        .hidden(false)
        .git_ignore(true)
        .build();

    for result in walker {
        let entry = result?;
        let path = entry.path();

        if !path.is_file() {
            continue;
        }
        if let Some(extension) = indexable_extension(path) {
            files.push(path.to_path_buf());
            *extension_counts.entry(extension).or_insert(0) += 1;
        }
    }
    Ok(FileDiscoveryResult {
        files,
        extension_counts,
    })
}

fn indexable_extension(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_str()?;
    if LOCKFILE_NAMES.contains(&file_name) {
        return None;
    }
    let extension = path.extension()?.to_str()?.to_lowercase();
    let is_known = extension_to_language_name(&extension).is_some()
        || PLAIN_TEXT_EXTENSIONS.contains(&extension.as_str());
    is_known.then_some(extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsx_files_are_indexable() {
        assert_eq!(indexable_extension(Path::new("src/App.tsx")), Some("tsx".to_string()));
    }

    #[test]
    fn ruby_files_are_indexable() {
        assert_eq!(indexable_extension(Path::new("app/models/user.rb")), Some("rb".to_string()));
    }

    #[test]
    fn extension_matching_ignores_case() {
        assert_eq!(indexable_extension(Path::new("Main.JAVA")), Some("java".to_string()));
    }

    #[test]
    fn markdown_is_indexable_without_a_grammar() {
        assert_eq!(indexable_extension(Path::new("README.md")), Some("md".to_string()));
    }

    #[test]
    fn lockfiles_are_skipped() {
        assert_eq!(indexable_extension(Path::new("web/package-lock.json")), None);
    }

    #[test]
    fn unknown_extensions_are_skipped() {
        assert_eq!(indexable_extension(Path::new("logo.png")), None);
    }
}
