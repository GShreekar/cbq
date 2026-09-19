use std::path::{Path, PathBuf};
use tokio::process::Command;

/// Which changes `cbq analyze` asks git for when no diff is piped in.
pub enum DiffSource {
    // Every uncommitted change, staged or not.
    Uncommitted,
    Staged,
    // Everything on this branch since it diverged from the given ref, including uncommitted work.
    SinceBase(String),
}

/// One file's changes in a unified diff.
#[derive(Debug, PartialEq)]
pub struct FileDiff {
    pub path: String,
    pub change: FileChange,
    pub hunks: Vec<Hunk>,
}

#[derive(Debug, PartialEq)]
pub enum FileChange {
    Added,
    Deleted,
    Modified,
    Renamed { from: String },
}

/// A contiguous block of changes, with line ranges on both sides of the diff.
#[derive(Debug, PartialEq)]
pub struct Hunk {
    pub old_start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub new_count: usize,
    // The unified-diff body: context, removed and added lines, each with its prefix.
    pub text: String,
}

impl FileDiff {
    /// Counts added and removed lines across every hunk.
    pub fn line_counts(&self) -> (usize, usize) {
        let body_lines = self.hunks.iter().flat_map(|hunk| hunk.text.lines());
        body_lines.fold((0, 0), |(added, removed), line| match line.as_bytes().first() {
            Some(b'+') => (added + 1, removed),
            Some(b'-') => (added, removed + 1),
            _ => (added, removed),
        })
    }
}

/// Runs `git diff` in `directory` and returns its output.
pub async fn read_git_diff(directory: &Path, source: &DiffSource) -> Result<String, anyhow::Error> {
    let mut command = Command::new("git");
    command.arg("-C").arg(directory).args(["diff", "--no-color", "--no-ext-diff"]);
    match source {
        DiffSource::Uncommitted => command.arg("HEAD"),
        DiffSource::Staged => command.arg("--staged"),
        DiffSource::SinceBase(base) => command.args(["--merge-base", base]),
    };

    let output = command
        .output()
        .await
        .map_err(|err| anyhow::anyhow!("Could not run git: {}", err))?;
    if !output.status.success() {
        // git follows the reason with usage hints about its own CLI that don't apply to cbq.
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git diff failed: {}", stderr.lines().next().unwrap_or("unknown error"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Finds the top-level directory of the git repository containing `directory`, if there is one.
pub async fn find_repository_root(directory: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    // Canonical, so it compares cleanly with the canonical index root even through symlinks.
    let root = String::from_utf8(output.stdout).ok()?;
    std::fs::canonicalize(root.trim()).ok()
}

/// Parses a unified diff, as produced by `git diff` (with or without prefixes) or `diff -u`.
pub fn parse_diff(diff_text: &str) -> Vec<FileDiff> {
    let mut files = Vec::new();
    let mut builder = FileBuilder::default();
    let mut lines = diff_text.lines().peekable();

    while let Some(line) = lines.next() {
        if line.starts_with("diff --git ") {
            files.extend(std::mem::take(&mut builder).finish());
        } else if let Some(path) = line.strip_prefix("--- ") {
            if builder.new_path.is_some() {
                files.extend(std::mem::take(&mut builder).finish());
            }
            builder.old_path = Some(clean_path(path));
        } else if let Some(path) = line.strip_prefix("+++ ") {
            builder.new_path = Some(clean_path(path));
        } else if let Some(path) = line.strip_prefix("rename from ") {
            builder.renamed_from = Some(path.to_string());
        } else if let Some(path) = line.strip_prefix("rename to ") {
            builder.renamed_to = Some(path.to_string());
        } else if let Some(hunk) = parse_hunk_header(line) {
            builder.hunks.push(read_hunk_body(hunk, &mut lines));
        }
    }
    files.extend(builder.finish());
    files
}

#[derive(Default)]
struct FileBuilder {
    old_path: Option<String>,
    new_path: Option<String>,
    renamed_from: Option<String>,
    renamed_to: Option<String>,
    hunks: Vec<Hunk>,
}

impl FileBuilder {
    fn finish(self) -> Option<FileDiff> {
        let has_prefixes = has_side_prefixes(self.old_path.as_deref(), self.new_path.as_deref());
        let old_path = self.old_path.map(|path| strip_side_prefix(path, "a/", has_prefixes));
        let new_path = self.new_path.map(|path| strip_side_prefix(path, "b/", has_prefixes));
        let (path, change) = match (old_path, new_path) {
            (Some(old), Some(new)) if old == "/dev/null" => (new, FileChange::Added),
            (Some(old), Some(new)) if new == "/dev/null" => (old, FileChange::Deleted),
            (Some(old), Some(new)) if old != new => (new, FileChange::Renamed { from: old }),
            (_, Some(new)) => (new, FileChange::Modified),
            // Pure renames have no ---/+++ lines, only "rename from/to".
            _ => (self.renamed_to?, FileChange::Renamed { from: self.renamed_from? }),
        };
        Some(FileDiff { path, change, hunks: self.hunks })
    }
}

// `diff -u` appends a tab and a timestamp to each path.
fn clean_path(raw: &str) -> String {
    raw.split('\t').next().unwrap_or(raw).trim_end().to_string()
}

// git's a/ and b/ prefixes can't be told apart from real directories named "a" by the header line
// (`--no-prefix` gives "diff --git a/x a/x"), so both sides must carry their own prefix.
fn has_side_prefixes(old_path: Option<&str>, new_path: Option<&str>) -> bool {
    let (Some(old), Some(new)) = (old_path, new_path) else {
        return false;
    };
    let old_has_prefix = old == "/dev/null" || old.starts_with("a/");
    let new_has_prefix = new == "/dev/null" || new.starts_with("b/");
    old_has_prefix && new_has_prefix
}

fn strip_side_prefix(path: String, prefix: &str, has_prefixes: bool) -> String {
    if path == "/dev/null" || !has_prefixes {
        return path;
    }
    path.strip_prefix(prefix).map(str::to_string).unwrap_or(path)
}

fn parse_hunk_header(line: &str) -> Option<Hunk> {
    let ranges = line.strip_prefix("@@ -")?.split(" @@").next()?;
    let (old_range, new_range) = ranges.split_once(" +")?;
    let (old_start, old_count) = parse_range(old_range)?;
    let (new_start, new_count) = parse_range(new_range)?;
    Some(Hunk { old_start, old_count, new_start, new_count, text: String::new() })
}

// A range is "start,count", or just "start" when the count is 1.
fn parse_range(range: &str) -> Option<(usize, usize)> {
    match range.split_once(',') {
        Some((start, count)) => Some((start.parse().ok()?, count.parse().ok()?)),
        None => Some((range.parse().ok()?, 1)),
    }
}

// Reads exactly as many lines as the header promises, so a removed line like "--- x" isn't taken for a file header.
fn read_hunk_body<'a>(mut hunk: Hunk, lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>) -> Hunk {
    let mut old_remaining = hunk.old_count;
    let mut new_remaining = hunk.new_count;

    while old_remaining > 0 || new_remaining > 0 {
        let Some(line) = lines.next() else { break };
        match line.as_bytes().first() {
            Some(b'+') => new_remaining = new_remaining.saturating_sub(1),
            Some(b'-') => old_remaining = old_remaining.saturating_sub(1),
            Some(b'\\') => continue, // "\ No newline at end of file"
            _ => {
                old_remaining = old_remaining.saturating_sub(1);
                new_remaining = new_remaining.saturating_sub(1);
            }
        }
        hunk.text.push_str(line);
        hunk.text.push('\n');
    }
    while lines.peek().is_some_and(|line| line.starts_with('\\')) {
        lines.next();
    }
    hunk
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODIFIED: &str = "\
diff --git a/src/lib.rs b/src/lib.rs
index 111..222 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,3 +10,4 @@ fn total() {
     let a = 1;
-    let b = 2;
+    let b = 3;
+    let c = 4;
 }
";

    #[test]
    fn modified_file_path_has_prefix_stripped() {
        assert_eq!(parse_diff(MODIFIED)[0].path, "src/lib.rs");
    }

    #[test]
    fn hunk_ranges_are_parsed() {
        let hunk = &parse_diff(MODIFIED)[0].hunks[0];
        assert_eq!((hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count), (10, 3, 10, 4));
    }

    #[test]
    fn added_and_removed_lines_are_counted() {
        assert_eq!(parse_diff(MODIFIED)[0].line_counts(), (2, 1));
    }

    #[test]
    fn new_file_is_marked_added() {
        let diff = "diff --git a/new.rs b/new.rs\nnew file mode 100644\n--- /dev/null\n+++ b/new.rs\n@@ -0,0 +1 @@\n+fn f() {}\n";
        assert_eq!(parse_diff(diff)[0].change, FileChange::Added);
    }

    #[test]
    fn deleted_file_keeps_its_old_path() {
        let diff = "diff --git a/old.rs b/old.rs\ndeleted file mode 100644\n--- a/old.rs\n+++ /dev/null\n@@ -1 +0,0 @@\n-fn f() {}\n";
        assert_eq!(parse_diff(diff)[0].path, "old.rs");
    }

    #[test]
    fn no_prefix_diff_keeps_real_directory_named_a() {
        let diff = "diff --git a/x.rs a/x.rs\n--- a/x.rs\n+++ a/x.rs\n@@ -1 +1 @@\n-1\n+2\n";
        assert_eq!(parse_diff(diff)[0].path, "a/x.rs");
    }

    #[test]
    fn prefixed_diff_of_a_directory_named_a_strips_only_the_prefix() {
        let diff = "diff --git a/a/x.rs b/a/x.rs\n--- a/a/x.rs\n+++ b/a/x.rs\n@@ -1 +1 @@\n-1\n+2\n";
        assert_eq!(parse_diff(diff)[0].path, "a/x.rs");
    }

    #[test]
    fn removed_line_that_looks_like_a_header_stays_in_the_hunk() {
        let diff = "diff --git a/notes.md b/notes.md\n--- a/notes.md\n+++ b/notes.md\n@@ -1,2 +1 @@\n---- heading\n keep\n";
        let files = parse_diff(diff);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].line_counts(), (0, 1));
    }

    #[test]
    fn pure_rename_is_detected() {
        let diff = "diff --git a/old.rs b/new.rs\nsimilarity index 100%\nrename from old.rs\nrename to new.rs\n";
        assert_eq!(parse_diff(diff)[0].change, FileChange::Renamed { from: "old.rs".to_string() });
    }

    #[test]
    fn plain_diff_u_output_is_parsed() {
        let diff = "--- src/a.c\t2024-01-01 10:00:00\n+++ src/a.c\t2024-01-02 10:00:00\n@@ -1 +1 @@\n-int x;\n+int y;\n";
        assert_eq!(parse_diff(diff)[0].path, "src/a.c");
    }

    #[test]
    fn several_files_are_split_apart() {
        let two_files = format!("{}{}", MODIFIED, MODIFIED.replace("src/lib.rs", "src/main.rs"));
        assert_eq!(parse_diff(&two_files).len(), 2);
    }

    #[test]
    fn binary_file_has_no_hunks() {
        let diff = "diff --git a/logo.png b/logo.png\nindex 1..2 100644\nBinary files a/logo.png and b/logo.png differ\n";
        assert!(parse_diff(diff).is_empty());
    }
}
