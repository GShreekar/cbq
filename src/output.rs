use serde_json::{json, Value};
use crate::services::vector_search::SearchResult;

/// How a command reports what it did: for a person, or for whatever called cbq.
#[derive(Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    pub fn is_json(self) -> bool {
        self == OutputFormat::Json
    }

    /// True when a command may print progress, spinners and prose.
    pub fn is_text(self) -> bool {
        self == OutputFormat::Text
    }
}

/// Prints a command's result as one JSON object on stdout.
pub fn emit(value: Value) {
    println!("{}", serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string()));
}

/// Prints a failure as JSON on stderr, keeping stdout free of anything but results.
pub fn emit_error(message: &str) {
    eprintln!("{}", json!({ "error": message }));
}

/// Describes a search hit, including the code itself so a caller needn't read the file.
pub fn search_result_json(result: &SearchResult) -> Value {
    json!({
        "path": result.chunk.file_path.to_string_lossy(),
        "start_line": result.chunk.start_line,
        "end_line": result.chunk.end_line,
        "name": result.chunk.name,
        "kind": result.chunk.chunk_type,
        "parent": result.chunk.parent,
        "language": result.chunk.language,
        "score": round_score(result.score),
        "matched_keywords": result.matched_keywords,
        "found_via_calls": result.found_via_calls,
        "content": result.chunk.content,
    })
}

// Scores carry two useful digits; the rest is noise that makes output diffs jump around.
pub fn round_score(score: f64) -> f64 {
    (score * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::services::chunker::CodeChunk;

    fn result() -> SearchResult {
        SearchResult {
            chunk: CodeChunk {
                file_path: PathBuf::from("src/cart.rs"),
                language: "rust".to_string(),
                name: "add".to_string(),
                chunk_type: "function".to_string(),
                parent: Some("Cart".to_string()),
                content: "pub fn add() {}".to_string(),
                start_line: 3,
                end_line: 9,
            },
            score: 0.756_3,
            matched_keywords: true,
            found_via_calls: false,
        }
    }

    #[test]
    fn a_search_hit_carries_its_location_and_code() {
        let value = search_result_json(&result());
        assert_eq!(value["path"], "src/cart.rs");
        assert_eq!(value["start_line"], 3);
        assert_eq!(value["content"], "pub fn add() {}");
    }

    #[test]
    fn a_search_hit_says_how_it_was_found() {
        let value = search_result_json(&result());
        assert_eq!(value["matched_keywords"], true);
        assert_eq!(value["found_via_calls"], false);
    }

    #[test]
    fn a_hit_inside_a_type_names_that_type() {
        assert_eq!(search_result_json(&result())["parent"], "Cart");
    }

    #[test]
    fn scores_are_rounded_to_two_digits() {
        assert_eq!(round_score(0.756_3), 0.76);
    }
}
