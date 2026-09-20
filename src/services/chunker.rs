use std::path::PathBuf;

/// How chunks are built. Raising it makes cbq rebuild indexes whose chunks predate the change.
pub const CHUNK_FORMAT_VERSION: u32 = 2;

/// Largest text sent to the embedding model; Ollama's nomic-embed-text fails past ~4.5 KB.
pub const MAX_CHUNK_BYTES: usize = 3_500;
// The header naming the file, symbol and lines is added before embedding, so code gets a little less.
const MAX_EMBED_HEADER_BYTES: usize = 400;
const MAX_CONTENT_BYTES: usize = MAX_CHUNK_BYTES - MAX_EMBED_HEADER_BYTES;
const MAX_CONTINUATION_HEADER_BYTES: usize = 300;

#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub file_path: PathBuf,
    pub language: String,
    pub name: String,
    pub chunk_type: String,
    // The type, class or module this belongs to, when it belongs to one.
    pub parent: Option<String>,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
}

impl CodeChunk {
    /// The text that gets embedded: the code, behind a header saying where it comes from.
    pub fn embed_text(&self) -> String {
        format!(
            "File: {}\nLanguage: {}\nSymbol: {} ({})\nParent: {}\nLines: {}-{}\n\n{}",
            self.file_path.display(),
            self.language,
            self.name,
            self.chunk_type,
            self.parent.as_deref().unwrap_or("<module>"),
            self.start_line,
            self.end_line,
            self.content
        )
    }

    /// Names the chunk the way a reader would refer to it, such as `Cart::add`.
    pub fn qualified_name(&self) -> String {
        match &self.parent {
            Some(parent) => format!("{}::{}", parent, self.name),
            None => self.name.clone(),
        }
    }
}

pub fn slice_by_lines(file_path: PathBuf, language: &str, content: &str) -> Vec<CodeChunk> {
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
            language: language.to_string(),
            name: format!("lines-{}-{}", start + 1, end),
            chunk_type: "general".to_string(),
            parent: None,
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

/// Splits a chunk whose code exceeds the budget into line-aligned pieces taken from the original source.
pub fn fit_to_byte_budget(chunk: CodeChunk, source_lines: &[&str]) -> Vec<CodeChunk> {
    if chunk.content.len() <= MAX_CONTENT_BYTES {
        return vec![chunk];
    }

    let last_line = chunk.end_line.min(source_lines.len());
    let lines = &source_lines[chunk.start_line - 1..last_line];
    let header = continuation_header(&chunk.name, lines);
    let body_budget = MAX_CONTENT_BYTES - header.len();

    let mut pieces = Vec::new();
    let mut body = String::new();
    let mut body_start_line = chunk.start_line;
    let mut body_end_line = chunk.start_line;

    for (line_number, segment) in split_into_segments(lines, chunk.start_line, body_budget) {
        if !body.is_empty() && body.len() + 1 + segment.len() > body_budget {
            pieces.push(build_piece(&chunk, &header, &body, body_start_line, body_end_line, pieces.is_empty()));
            body.clear();
        }
        if body.is_empty() {
            body_start_line = line_number;
        } else {
            body.push('\n');
        }
        body.push_str(segment);
        body_end_line = line_number;
    }
    if !body.is_empty() {
        pieces.push(build_piece(&chunk, &header, &body, body_start_line, body_end_line, pieces.is_empty()));
    }
    pieces
}

// Later pieces lose the chunk's opening line, so they repeat it to stay searchable by name.
fn continuation_header(name: &str, lines: &[&str]) -> String {
    let signature = lines.iter().map(|line| line.trim()).find(|line| !line.is_empty()).unwrap_or("");
    let header = format!("// ... continued from {}\n{}", name, signature);
    format!("{}\n", truncate_at_char_boundary(&header, MAX_CONTINUATION_HEADER_BYTES))
}

fn build_piece(chunk: &CodeChunk, header: &str, body: &str, start_line: usize, end_line: usize, is_first: bool) -> CodeChunk {
    let content = if is_first { body.to_string() } else { format!("{}{}", header, body) };
    CodeChunk {
        file_path: chunk.file_path.clone(),
        language: chunk.language.clone(),
        name: format!("{}-part-{}-{}", chunk.name, start_line, end_line),
        chunk_type: chunk.chunk_type.clone(),
        parent: chunk.parent.clone(),
        content,
        start_line,
        end_line,
    }
}

// Minified files can have single lines far over budget, so those are cut into several segments.
fn split_into_segments<'a>(lines: &[&'a str], first_line_number: usize, max_bytes: usize) -> Vec<(usize, &'a str)> {
    let mut segments = Vec::new();
    for (offset, line) in lines.iter().enumerate() {
        let mut rest = *line;
        while rest.len() > max_bytes {
            let segment = truncate_at_char_boundary(rest, max_bytes);
            segments.push((first_line_number + offset, segment));
            rest = &rest[segment.len()..];
        }
        segments.push((first_line_number + offset, rest));
    }
    segments
}

fn truncate_at_char_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut cut = max_bytes;
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    &text[..cut]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_spanning(lines: &[&str]) -> CodeChunk {
        CodeChunk {
            file_path: PathBuf::from("src/example.rs"),
            language: "rust".to_string(),
            name: "example".to_string(),
            chunk_type: "function".to_string(),
            parent: None,
            content: lines.join("\n"),
            start_line: 1,
            end_line: lines.len(),
        }
    }

    #[test]
    fn chunk_within_budget_is_unchanged() {
        let lines = ["fn example() {", "    work();", "}"];
        let pieces = fit_to_byte_budget(chunk_spanning(&lines), &lines);
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].content, lines.join("\n"));
    }

    #[test]
    fn oversized_chunk_pieces_all_fit_the_budget() {
        let line = "    let value = compute_something(first_argument, second_argument);";
        let lines = vec![line; 200];
        let pieces = fit_to_byte_budget(chunk_spanning(&lines), &lines);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|piece| piece.embed_text().len() <= MAX_CHUNK_BYTES));
    }

    #[test]
    fn oversized_chunk_pieces_cover_every_line_in_order() {
        let line = "    let value = compute_something(first_argument, second_argument);";
        let lines = vec![line; 200];
        let pieces = fit_to_byte_budget(chunk_spanning(&lines), &lines);
        let ranges: Vec<(usize, usize)> = pieces.iter().map(|piece| (piece.start_line, piece.end_line)).collect();
        assert_eq!(ranges.first().unwrap().0, 1);
        assert_eq!(ranges.last().unwrap().1, 200);
        assert!(ranges.windows(2).all(|pair| pair[1].0 == pair[0].1 + 1));
    }

    #[test]
    fn continuation_pieces_repeat_the_opening_line() {
        let mut lines = vec!["pub fn long_function() {"];
        lines.extend(vec!["    let value = compute_something(first_argument, second_argument);"; 200]);
        let pieces = fit_to_byte_budget(chunk_spanning(&lines), &lines);
        assert!(pieces[1].content.starts_with("// ... continued from example\npub fn long_function() {\n"));
    }

    #[test]
    fn minified_single_line_is_cut_to_fit() {
        let minified = "a=1;".repeat(5_000);
        let lines = [minified.as_str()];
        let pieces = fit_to_byte_budget(chunk_spanning(&lines), &lines);
        assert!(pieces.len() > 1);
        assert!(pieces.iter().all(|piece| piece.embed_text().len() <= MAX_CHUNK_BYTES));
    }

    #[test]
    fn multibyte_text_is_never_cut_mid_character() {
        let wide = "é".repeat(5_000);
        let lines = [wide.as_str()];
        let pieces = fit_to_byte_budget(chunk_spanning(&lines), &lines);
        let rejoined: String = pieces.iter().map(|piece| piece.content.lines().last().unwrap()).collect();
        assert_eq!(rejoined, wide);
    }

    #[test]
    fn embed_text_names_the_file_symbol_and_parent() {
        let mut chunk = chunk_spanning(&["fn add() {}"]);
        chunk.name = "add".to_string();
        chunk.parent = Some("Cart".to_string());
        let embedded = chunk.embed_text();
        assert!(embedded.starts_with("File: src/example.rs\nLanguage: rust\nSymbol: add (function)\nParent: Cart\n"));
        assert!(embedded.ends_with("fn add() {}"));
    }

    #[test]
    fn a_chunk_outside_any_type_has_no_parent_in_its_header() {
        assert!(chunk_spanning(&["fn free() {}"]).embed_text().contains("Parent: <module>"));
    }

    #[test]
    fn qualified_name_includes_the_parent_type() {
        let mut chunk = chunk_spanning(&["fn add() {}"]);
        chunk.name = "add".to_string();
        chunk.parent = Some("Cart".to_string());
        assert_eq!(chunk.qualified_name(), "Cart::add");
    }
}
