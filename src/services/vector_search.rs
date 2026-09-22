use rayon::prelude::*;
use crate::services::chunker::CodeChunk;
use crate::services::embeddings::{bytes_to_vector, normalise};

/// A chunk found by search, scored from 0.0 (unrelated) to 1.0 (a perfect match).
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: CodeChunk,
    pub score: f64,
    // True when keyword search found it, which is why results aren't ordered by score alone.
    pub matched_keywords: bool,
    // True when it was pulled in because the results above call it.
    pub found_via_calls: bool,
}

// Tests describe behaviour rather than implement it, so they lose to source files at equal similarity.
const TEST_FILE_PENALTY: f32 = 0.8;

/// Every chunk's vector, laid out end to end so scoring walks memory in one pass.
pub struct ChunkVectors {
    ids: Vec<i64>,
    penalties: Vec<f32>,
    values: Vec<f32>,
    dimensions: usize,
}

/// Loads every chunk's vector, leaving the source text in the database until the winners are known.
pub fn load_chunk_vectors(conn: &rusqlite::Connection) -> Result<ChunkVectors, anyhow::Error> {
    let mut stmt = conn.prepare("SELECT id, file_path, embedding FROM chunks")?;
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let file_path: String = row.get(1)?;
        let embedding: Vec<u8> = row.get(2)?;
        Ok((id, file_path, embedding))
    })?;

    let mut vectors = ChunkVectors { ids: Vec::new(), penalties: Vec::new(), values: Vec::new(), dimensions: 0 };
    for row in rows {
        let (id, file_path, embedding) = row?;
        let vector = bytes_to_vector(&embedding);
        if vectors.dimensions == 0 {
            vectors.dimensions = vector.len();
        }
        if vector.len() != vectors.dimensions {
            anyhow::bail!(
                "The index mixes {}- and {}-dimension vectors; rebuild it with `cbq index --force`.",
                vectors.dimensions,
                vector.len()
            );
        }
        vectors.ids.push(id);
        vectors.penalties.push(match is_test_file(std::path::Path::new(&file_path)) {
            true => TEST_FILE_PENALTY,
            false => 1.0,
        });
        vectors.values.extend_from_slice(&vector);
    }
    Ok(vectors)
}

impl ChunkVectors {
    /// Scores every chunk against the query and returns the best `count` of them, best first.
    pub fn best_matches(&self, query_vector: &[f32], count: usize) -> Result<Vec<(i64, f64)>, anyhow::Error> {
        let mut scored = self.score_all(query_vector)?;
        if scored.len() > count {
            // Only the leaders need ordering, so the rest are merely partitioned away.
            scored.select_nth_unstable_by(count, |first, second| second.1.total_cmp(&first.1));
            scored.truncate(count);
        }
        scored.sort_by(|first, second| second.1.total_cmp(&first.1));
        Ok(scored)
    }

    /// Scores every chunk against the query, in the order they were loaded.
    pub fn score_all(&self, query_vector: &[f32]) -> Result<Vec<(i64, f64)>, anyhow::Error> {
        if self.ids.is_empty() {
            return Ok(Vec::new());
        }
        if query_vector.len() != self.dimensions {
            anyhow::bail!(
                "The index stores {}-dimension vectors, but the query embedding has {}. \
                 The embedding model changed since indexing; rebuild the index with `cbq index`.",
                self.dimensions,
                query_vector.len()
            );
        }

        // Both sides are unit length, so their dot product is the cosine similarity.
        let query = normalise(query_vector);
        Ok(self
            .values
            .par_chunks(self.dimensions)
            .enumerate()
            .map(|(position, vector)| {
                let similarity: f32 = vector.iter().zip(query.iter()).map(|(stored, asked)| stored * asked).sum();
                let score = similarity.clamp(0.0, 1.0) * self.penalties[position];
                (self.ids[position], score as f64)
            })
            .collect())
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// How many numbers each stored vector has, or None when the index is empty.
    pub fn dimensions(&self) -> Option<usize> {
        (!self.is_empty()).then_some(self.dimensions)
    }
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

    fn index_with(chunks: &[(&str, [f32; 3])]) -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        crate::db::schema::create_tables(&conn).unwrap();
        for (path, vector) in chunks {
            conn.execute(
                "INSERT INTO chunks (file_path, language, name, chunk_type, content, start_line, end_line, embedding)
                 VALUES (?1, 'rust', 'chunk', 'function', 'fn chunk() {}', 1, 1, ?2)",
                rusqlite::params![path, crate::services::embeddings::vector_to_bytes(&normalise(vector))],
            ).unwrap();
        }
        conn
    }

    fn scores_of(conn: &rusqlite::Connection, query: &[f32]) -> Vec<(i64, f64)> {
        load_chunk_vectors(conn).unwrap().best_matches(query, 10).unwrap()
    }

    #[test]
    fn weak_matches_are_returned_rather_than_dropped() {
        let conn = index_with(&[("src/lib.rs", [0.30, 0.95, 0.0])]);
        let scored = scores_of(&conn, &[1.0, 0.0, 0.0]);
        assert_eq!(scored.len(), 1);
        assert!(scored[0].1 < 0.5, "expected a weak score, got {}", scored[0].1);
    }

    #[test]
    fn opposite_vectors_score_zero_rather_than_negative() {
        let conn = index_with(&[("tests/lib_test.rs", [-1.0, 0.0, 0.0])]);
        assert_eq!(scores_of(&conn, &[1.0, 0.0, 0.0])[0].1, 0.0);
    }

    #[test]
    fn test_files_rank_below_source_files_that_match_equally_well() {
        let conn = index_with(&[("tests/cart_test.rs", [1.0, 0.0, 0.0]), ("src/cart.rs", [1.0, 0.0, 0.0])]);
        let scored = scores_of(&conn, &[1.0, 0.0, 0.0]);
        assert_eq!(scored[0].0, 2, "the source file should rank first");
        assert!(scored[0].1 > scored[1].1);
    }

    #[test]
    fn an_unnormalised_query_still_scores_as_cosine_similarity() {
        let conn = index_with(&[("src/cart.rs", [1.0, 0.0, 0.0])]);
        assert!((scores_of(&conn, &[7.0, 0.0, 0.0])[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn only_the_requested_number_of_matches_comes_back() {
        let conn = index_with(&[("a.rs", [1.0, 0.0, 0.0]), ("b.rs", [0.0, 1.0, 0.0]), ("c.rs", [0.0, 0.0, 1.0])]);
        assert_eq!(load_chunk_vectors(&conn).unwrap().best_matches(&[1.0, 1.0, 1.0], 2).unwrap().len(), 2);
    }

    #[test]
    fn search_fails_when_query_and_index_dimensions_differ() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        crate::db::schema::create_tables(&conn).unwrap();
        conn.execute(
            "INSERT INTO chunks (file_path, language, name, chunk_type, content, start_line, end_line, embedding)
             VALUES ('src/lib.rs', 'rust', 'run', 'function', 'fn run() {}', 1, 1, ?1)",
            [crate::services::embeddings::vector_to_bytes(&[0.5; 768])],
        ).unwrap();
        assert!(load_chunk_vectors(&conn).unwrap().best_matches(&[0.5; 1024], 5).is_err());
    }

    #[test]
    fn an_empty_index_returns_no_matches() {
        let conn = index_with(&[]);
        assert!(load_chunk_vectors(&conn).unwrap().is_empty());
        assert!(scores_of(&conn, &[1.0, 0.0, 0.0]).is_empty());
    }
}
