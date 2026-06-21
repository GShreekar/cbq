use rusqlite::Connection;
use std::path::PathBuf;
use crate::services::chunker::CodeChunk;
use crate::services::embeddings::get_search_tokens;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f64,
}

pub fn search_codebase(conn: &Connection, query: &str, top_k: usize) -> Result<Vec<SearchResult>, anyhow::Error> {
    let query_tokens = get_search_tokens(query);
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let mut stmt = conn.prepare(
        "SELECT file_path, name, chunk_type, content, start_line, end_line FROM chunks"
    )?;

    let chunk_rows = stmt.query_map([], |row| {
        let file_path_str: String = row.get(0)?;
        Ok(CodeChunk {
            file_path: PathBuf::from(file_path_str),
            name: row.get(1)?,
            chunk_type: row.get(2)?,
            content: row.get(3)?,
            start_line: row.get::<_, i64>(4)? as usize,
            end_line: row.get::<_, i64>(5)? as usize,
        })
    })?;

    let mut results = Vec::new();
    for chunk_res in chunk_rows {
        let chunk = chunk_res?;
        let chunk_tokens = get_search_tokens(&chunk.content);

        let overlap_count = query_tokens
            .iter()
            .filter(|token| chunk_tokens.contains(*token))
            .count();

        if overlap_count > 0 {
            let score = overlap_count as f64 / query_tokens.len() as f64;
            results.push(SearchResult { chunk, score });
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    results.truncate(top_k);
    Ok(results)
}