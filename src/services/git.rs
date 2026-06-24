#[derive(Debug)]
pub struct FileDiff {
    pub file_path: String,
    pub added_lines: Vec<String>,
}

pub fn parse_diff(diff_text: &str) -> Vec<FileDiff> {
    let mut files = Vec::new();
    let mut current_file: Option<FileDiff> = None;

    for line in diff_text.lines() {
        if line.starts_with("+++ b/") {
            if let Some(f) = current_file.take() {
                files.push(f);
            }
            let file_path = line.trim_start_matches("+++ b/").to_string();
            current_file = Some(FileDiff {
                file_path,
                added_lines: Vec::new(),
            });
        } else if line.starts_with('+') && !line.starts_with("+++") {
            if let Some(ref mut f) = current_file {
                let code = line.trim_start_matches('+').trim();
                if !code.is_empty() {
                    f.added_lines.push(code.to_string());
                }
            }
        }
    }

    if let Some(f) = current_file {
        files.push(f);
    }

    files
}