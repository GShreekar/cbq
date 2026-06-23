use crate::services::chunker::CodeChunk;
use crate::services::embeddings::bytes_to_vector;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f64,
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (val_a, val_b) in a.iter().zip(b.iter()) {
        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    (dot_product / (norm_a.sqrt() * norm_b.sqrt())) as f64
}

pub fn search_codebase(
    conn: &rusqlite::Connection,
    query_vector: &[f32],
    limit: usize,
) -> Result<Vec<SearchResult>, anyhow::Error> {
    let mut stmt = conn.prepare(
        "SELECT file_path, name, chunk_type, content, start_line, end_line, embedding FROM chunks"
    )?;

    let chunk_iter = stmt.query_map([], |row| {
        let file_path_str: String = row.get(0)?;
        let name: String = row.get(1)?;
        let chunk_type: String = row.get(2)?;
        let content: String = row.get(3)?;
        let start_line: usize = row.get(4)?;
        let end_line: usize = row.get(5)?;
        let emb_bytes: Vec<u8> = row.get(6)?;

        Ok((
            CodeChunk {
                file_path: PathBuf::from(file_path_str),
                name,
                chunk_type,
                content,
                start_line,
                end_line,
            },
            emb_bytes,
        ))
    })?;

    let mut results = Vec::new();
    for item in chunk_iter {
        if let Ok((chunk, bytes)) = item {
            let chunk_vector = bytes_to_vector(&bytes);
            let score = cosine_similarity(query_vector, &chunk_vector);
            results.push(SearchResult { chunk, score });
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }
}