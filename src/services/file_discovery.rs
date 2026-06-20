use std::collections::HashMap;
use std::path::{Path, PathBuf};
use ignore::WalkBuilder;

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

        if path.is_file() {
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                if file_name == "Cargo.lock" || file_name.ends_with(".png") || file_name.ends_with(".jpg") {
                    continue;
                }
            }

            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_lowercase();
                let is_code = matches!(
                    ext_lower.as_str(),
                    "rs" | "py" | "js" | "ts" | "c" | "cpp" | "h" | "hpp" | "go" | "java" | "html" | "css" | "json" | "md" | "toml" | "sh" | "swift" | "kt"
                );
                if is_code {
                    files.push(path.to_path_buf());
                    *extension_counts.entry(ext_lower).or_insert(0) += 1;
                }
            }
        }
    }
    Ok(FileDiscoveryResult {
        files,
        extension_counts,
    })
}