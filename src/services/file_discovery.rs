use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ignore::WalkBuilder;
use crate::services::parser::extension_to_language_name;

/// Files larger than this are skipped unless the caller raises the limit; they're almost always generated.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 512 * 1024;

/// Per-directory file where users list extra paths to skip, in .gitignore syntax.
pub const IGNORE_FILE_NAME: &str = ".cbqignore";

// Indexed as plain line windows because they have no grammar in the parser.
const PLAIN_TEXT_EXTENSIONS: [&str; 1] = ["md"];
// Machine-generated and huge; they would bury real code in search results.
const LOCKFILE_NAMES: [&str; 3] = ["package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml"];
// Version control internals, dependencies and build output: skipped even when a project doesn't gitignore them.
const SKIPPED_DIRECTORY_NAMES: [&str; 16] = [
    ".git", ".hg", ".svn", "node_modules", "bower_components", "vendor", "target", "dist", "build",
    "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", ".next",
];

pub struct FileDiscoveryResult {
    pub files: Vec<PathBuf>,
    pub extension_counts: HashMap<String, usize>,
    pub skipped: Vec<SkippedFile>,
}

/// A file left out of the index for a reason the user may want to act on.
pub struct SkippedFile {
    pub path: PathBuf,
    pub reason: SkipReason,
}

pub enum SkipReason {
    TooLarge { size_bytes: u64, limit_bytes: u64 },
    Unreadable(String),
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SkipReason::TooLarge { size_bytes, limit_bytes } => write!(
                formatter,
                "{} KB, over the {} KB limit",
                size_bytes / 1024,
                limit_bytes / 1024
            ),
            SkipReason::Unreadable(reason) => write!(formatter, "{}", reason),
        }
    }
}

/// Finds indexable source files under `root_path`, honoring .gitignore, .ignore and .cbqignore files.
pub fn discover_files(root_path: &Path, max_file_bytes: u64) -> Result<FileDiscoveryResult, anyhow::Error> {
    let mut result = FileDiscoveryResult {
        files: Vec::new(),
        extension_counts: HashMap::new(),
        skipped: Vec::new(),
    };

    let walker = WalkBuilder::new(root_path)
        .hidden(false)
        .git_ignore(true)
        .require_git(false)
        .add_custom_ignore_filename(IGNORE_FILE_NAME)
        .filter_entry(|entry| !is_skipped_directory(entry))
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                result.skipped.push(SkippedFile {
                    path: error_path(&err, root_path),
                    reason: SkipReason::Unreadable(err.to_string()),
                });
                continue;
            }
        };
        if !entry.file_type().is_some_and(|file_type| file_type.is_file()) {
            continue;
        }
        let Some(extension) = indexable_extension(entry.path()) else {
            continue;
        };

        let size = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if size > max_file_bytes {
            result.skipped.push(SkippedFile {
                path: entry.path().to_path_buf(),
                reason: SkipReason::TooLarge { size_bytes: size, limit_bytes: max_file_bytes },
            });
            continue;
        }
        result.files.push(entry.into_path());
        *result.extension_counts.entry(extension).or_insert(0) += 1;
    }
    Ok(result)
}

fn is_skipped_directory(entry: &ignore::DirEntry) -> bool {
    let is_directory = entry.file_type().is_some_and(|file_type| file_type.is_dir());
    // Depth 0 is the project root itself, which is indexed whatever it's called.
    is_directory
        && entry.depth() > 0
        && entry.file_name().to_str().is_some_and(|name| SKIPPED_DIRECTORY_NAMES.contains(&name))
}

fn indexable_extension(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_str()?;
    if LOCKFILE_NAMES.contains(&file_name) || file_name.contains(".min.") {
        return None;
    }
    let extension = path.extension()?.to_str()?.to_lowercase();
    let is_known = extension_to_language_name(&extension).is_some()
        || PLAIN_TEXT_EXTENSIONS.contains(&extension.as_str());
    is_known.then_some(extension)
}

fn error_path(error: &ignore::Error, root_path: &Path) -> PathBuf {
    match error {
        ignore::Error::WithPath { path, .. } => path.clone(),
        ignore::Error::WithDepth { err, .. } | ignore::Error::WithLineNumber { err, .. } => error_path(err, root_path),
        _ => root_path.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn project_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for (path, content) in files {
            let full_path = root.path().join(path);
            fs::create_dir_all(full_path.parent().unwrap()).unwrap();
            fs::write(full_path, content).unwrap();
        }
        root
    }

    fn discovered_names(root: &Path) -> Vec<String> {
        let mut names: Vec<String> = discover_files(root, DEFAULT_MAX_FILE_BYTES)
            .unwrap()
            .files
            .iter()
            .map(|path| path.strip_prefix(root).unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

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
    fn minified_files_are_skipped() {
        assert_eq!(indexable_extension(Path::new("static/app.min.js")), None);
    }

    #[test]
    fn unknown_extensions_are_skipped() {
        assert_eq!(indexable_extension(Path::new("logo.png")), None);
    }

    #[test]
    fn dependency_and_vcs_directories_are_not_walked() {
        let root = project_with(&[
            ("src/lib.rs", "fn a() {}"),
            ("node_modules/pkg/index.js", "x"),
            (".git/hooks/pre-commit.sh", "x"),
            ("target/debug/build.rs", "x"),
        ]);
        assert_eq!(discovered_names(root.path()), vec!["src/lib.rs"]);
    }

    #[test]
    fn hidden_source_directories_are_still_indexed() {
        let root = project_with(&[(".github/workflows/ci.yml", "on: push")]);
        assert_eq!(discovered_names(root.path()), vec![".github/workflows/ci.yml"]);
    }

    #[test]
    fn cbqignore_patterns_are_honored() {
        let root = project_with(&[
            (".cbqignore", "generated/\n"),
            ("src/lib.rs", "fn a() {}"),
            ("generated/schema.rs", "fn b() {}"),
        ]);
        assert_eq!(discovered_names(root.path()), vec!["src/lib.rs"]);
    }

    #[test]
    fn gitignore_is_honored_outside_a_git_repository() {
        let root = project_with(&[(".gitignore", "out/\n"), ("src/lib.rs", "fn a() {}"), ("out/gen.rs", "x")]);
        assert_eq!(discovered_names(root.path()), vec!["src/lib.rs"]);
    }

    #[test]
    fn files_over_the_size_limit_are_reported_as_skipped() {
        let root = project_with(&[("src/big.rs", &"x".repeat(2048))]);
        let result = discover_files(root.path(), 1024).unwrap();
        assert_eq!(result.skipped[0].path, root.path().join("src/big.rs"));
    }

    #[test]
    fn project_root_named_like_a_skipped_directory_is_still_indexed() {
        let root = project_with(&[("build/src/lib.rs", "fn a() {}")]);
        assert_eq!(discovered_names(&root.path().join("build")), vec!["src/lib.rs"]);
    }
}
