use crate::services::chunker::CodeChunk;
use crate::services::embeddings::bytes_to_vector;
use std::path::PathBuf;

/// A chunk found by search, scored from 0.0 (unrelated) to 1.0 (a perfect match).
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f64,
    // True when keyword search found it, which is why results aren't ordered by score alone.
    pub matched_keywords: bool,
}

// Tests describe behaviour rather than implement it, so they lose to source files at equal similarity.
const TEST_FILE_PENALTY: f64 = 0.8;

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

/// A chunk scored against the query, with the row id that ranks it alongside keyword matches.
pub struct ScoredChunk {
    pub id: i64,
    pub chunk: CodeChunk,
    pub similarity: f64,
}

/// Scores every chunk in the index against the query vector.
pub fn score_all_chunks(
    conn: &rusqlite::Connection,
    query_vector: &[f32],
) -> Result<Vec<ScoredChunk>, anyhow::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, file_path, language, name, chunk_type, parent, content, start_line, end_line, embedding FROM chunks"
    )?;

    let chunk_iter = stmt.query_map([], |row| {
        let chunk_id: i64 = row.get(0)?;
        let file_path: String = row.get(1)?;
        let embedding: Vec<u8> = row.get(9)?;

        Ok((
            chunk_id,
            CodeChunk {
                file_path: PathBuf::from(file_path),
                language: row.get(2)?,
                name: row.get(3)?,
                chunk_type: row.get(4)?,
                parent: row.get(5)?,
                content: row.get(6)?,
                start_line: row.get(7)?,
                end_line: row.get(8)?,
            },
            embedding,
        ))
    })?;

    let mut scored = Vec::new();
    for item in chunk_iter {
        let (id, chunk, bytes) = item?;
        let chunk_vector = bytes_to_vector(&bytes);
        if chunk_vector.len() != query_vector.len() {
            return Err(anyhow::anyhow!(
                "The index stores {}-dimension vectors, but the query embedding has {}. \
                 The embedding model changed since indexing; rebuild the index with `cbq index`.",
                chunk_vector.len(),
                query_vector.len()
            ));
        }
        // Clamped first: penalising a negative similarity would move it towards zero, raising it.
        let similarity = cosine_similarity(query_vector, &chunk_vector).clamp(0.0, 1.0);
        let similarity = match is_test_file(&chunk.file_path) {
            true => similarity * TEST_FILE_PENALTY,
            false => similarity,
        };
        scored.push(ScoredChunk { id, chunk, similarity });
    }
    Ok(scored)
}

/// Reports whether a path looks like test code rather than the implementation.
pub fn is_test_file(path: &std::path::Path) -> bool {
    let path_str = path.to_string_lossy();
    path_str.contains("/test")
        || path_str.contains("\\test")
        || path_str.contains("test_")
        || path_str.starts_with("test")
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

    fn index_with(chunks: &[(&str, [f32; 3])]) -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        crate::db::schema::create_tables(&conn).unwrap();
        for (path, vector) in chunks {
            conn.execute(
                "INSERT INTO chunks (file_path, name, chunk_type, content, start_line, end_line, embedding)
                 VALUES (?1, 'chunk', 'function', 'fn chunk() {}', 1, 1, ?2)",
                rusqlite::params![path, crate::services::embeddings::vector_to_bytes(vector)],
            ).unwrap();
        }
        conn
    }

    #[test]
    fn weak_matches_are_returned_rather_than_dropped() {
        let conn = index_with(&[("src/lib.rs", [0.30, 0.95, 0.0])]);
        let results = score_all_chunks(&conn, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity < 0.5, "expected a weak score, got {}", results[0].similarity);
    }

    #[test]
    fn opposite_vectors_score_zero_rather_than_negative() {
        let conn = index_with(&[("tests/lib_test.rs", [-1.0, 0.0, 0.0])]);
        let results = score_all_chunks(&conn, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(results[0].similarity, 0.0);
    }

    #[test]
    fn test_files_rank_below_source_files_that_match_equally_well() {
        let conn = index_with(&[("tests/cart_test.rs", [1.0, 0.0, 0.0]), ("src/cart.rs", [1.0, 0.0, 0.0])]);
        let mut results = score_all_chunks(&conn, &[1.0, 0.0, 0.0]).unwrap();
        results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        assert_eq!(results[0].chunk.file_path.to_string_lossy(), "src/cart.rs");
        assert!(results[0].similarity > results[1].similarity);
    }

    #[test]
    fn search_fails_when_query_and_index_dimensions_differ() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        crate::db::schema::create_tables(&conn).unwrap();
        conn.execute(
            "INSERT INTO chunks (file_path, name, chunk_type, content, start_line, end_line, embedding)
             VALUES ('src/lib.rs', 'run', 'function', 'fn run() {}', 1, 1, ?1)",
            [crate::services::embeddings::vector_to_bytes(&[0.5; 768])],
        ).unwrap();
        assert!(score_all_chunks(&conn, &[0.5; 1024]).is_err());
    }
}