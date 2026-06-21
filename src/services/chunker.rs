use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: PathBuf,
    pub name: String,
    pub chunk_type: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

pub fn slice_by_lines(file_path: PathBuf, content: &str) -> Vec<CodeChunk> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let chunk_size = 30;
    let overlap = 5;

    let mut start = 0;
    while start < lines.len() {
        let end = std::cmp::min(start + chunk_size, lines.len());
        let chunk_lines = &lines[start..end];
        let chunk_content = chunk_lines.join("\n");

        chunks.push(CodeChunk {
            file_path: file_path.clone(),
            name: format!("lines-{}-{}", start + 1, end),
            chunk_type: "general".to_string(),
            content: chunk_content,
            start_line: start + 1,
            end_line: end,
        });

        if end == lines.len() {
            break;
        }
        start += chunk_size - overlap;
    }
    chunks
}