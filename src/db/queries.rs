use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use rusqlite::{params, Connection};
use crate::db::index_metadata::write_embedding_model;
use crate::services::chunker::CodeChunk;

/// One file's freshly embedded chunks, ready to replace whatever the index held for it.
pub struct FileUpdate<'a> {
    pub path: &'a str,
    // None when some chunks failed, so the next run retries the file.
    pub content_hash: Option<&'a str>,
    pub chunks: &'a [CodeChunk],
    pub embeddings: &'a [Vec<f32>],
}

/// Reads the content hash recorded for each indexed file; None marks a file whose indexing was incomplete.
pub fn read_file_hashes(conn: &Connection) -> Result<HashMap<String, Option<String>>, anyhow::Error> {
    let mut stmt = conn.prepare("SELECT path, content_hash FROM files")?;
    let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Reports whether the index holds any chunks at all.
pub fn has_chunks(conn: &Connection) -> Result<bool, anyhow::Error> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM chunks)", [], |row| row.get(0))?)
}

/// Replaces one file's chunks and records its content hash, in one transaction.
pub fn replace_file_chunks(conn: &mut Connection, update: &FileUpdate) -> Result<(), anyhow::Error> {
    anyhow::ensure!(
        update.chunks.len() == update.embeddings.len(),
        "{} chunks but {} embeddings",
        update.chunks.len(),
        update.embeddings.len()
    );
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM chunks WHERE file_path = ?1", [update.path])?;

    {
        let mut stmt = tx.prepare(
            "INSERT INTO chunks (file_path, name, chunk_type, content, start_line, end_line, embedding)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
        )?;

        for (chunk, emb) in update.chunks.iter().zip(update.embeddings.iter()) {
            let emb_bytes = crate::services::embeddings::vector_to_bytes(emb);
            stmt.execute(params![
                update.path,
                chunk.name,
                chunk.chunk_type,
                chunk.content,
                chunk.start_line as i64,
                chunk.end_line as i64,
                emb_bytes,
            ])?;
        }
    }

    tx.execute(
        "INSERT INTO files (path, content_hash) VALUES (?1, ?2)
         ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash",
        params![update.path, update.content_hash],
    )?;
    tx.commit()?;
    Ok(())
}

/// Deletes files that no longer exist in the project, along with their chunks.
pub fn remove_files(conn: &mut Connection, paths: &[String]) -> Result<(), anyhow::Error> {
    let tx = conn.transaction()?;
    for path in paths {
        tx.execute("DELETE FROM chunks WHERE file_path = ?1", [path])?;
        tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    }
    tx.commit()?;
    Ok(())
}

/// Empties the index and records the embedding model the rebuilt index will use.
pub fn reset_index(conn: &mut Connection, embedding_model: &str) -> Result<(), anyhow::Error> {
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM chunks", [])?;
    tx.execute("DELETE FROM files", [])?;
    write_embedding_model(&tx, embedding_model)?;
    tx.commit()?;
    Ok(())
}

/// Deletes chunks that belong to no tracked file, such as those left by indexes built before file tracking.
pub fn delete_untracked_chunks(conn: &Connection) -> Result<usize, anyhow::Error> {
    Ok(conn.execute("DELETE FROM chunks WHERE file_path NOT IN (SELECT path FROM files)", [])?)
}

/// A summary of what an index currently holds.
pub struct DbStats {
    pub total_chunks: usize,
    pub indexed_files: usize,
    pub file_extensions: Vec<String>,
}

/// Summarises the index for display.
pub fn get_db_stats(conn: &Connection) -> Result<DbStats, anyhow::Error> {
    let total_chunks = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;
    let indexed_files = conn.query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))?;

    let mut stmt = conn.prepare("SELECT DISTINCT file_path FROM chunks")?;
    let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut file_extensions = BTreeSet::new();
    for path in paths {
        if let Some(extension) = Path::new(&path?).extension().and_then(|name| name.to_str()) {
            file_extensions.insert(extension.to_lowercase());
        }
    }

    Ok(DbStats {
        total_chunks,
        indexed_files,
        file_extensions: file_extensions.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::db::schema::create_tables;
    use crate::services::chunker::CodeChunk;

    fn chunk_in(path: &str) -> CodeChunk {
        CodeChunk {
            file_path: PathBuf::from(path),
            name: "chunk".to_string(),
            chunk_type: "function".to_string(),
            content: "fn chunk() {}".to_string(),
            start_line: 1,
            end_line: 1,
        }
    }

    fn index_with(files: &[(&str, &str)]) -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        create_tables(&conn).unwrap();
        for (path, hash) in files {
            replace_file_chunks(&mut conn, &FileUpdate {
                path,
                content_hash: Some(hash),
                chunks: &[chunk_in(path)],
                embeddings: &[vec![0.5; 4]],
            }).unwrap();
        }
        conn
    }

    #[test]
    fn replacing_a_file_leaves_other_files_alone() {
        let mut conn = index_with(&[("src/a.rs", "aaa"), ("src/b.rs", "bbb")]);
        replace_file_chunks(&mut conn, &FileUpdate {
            path: "src/a.rs",
            content_hash: Some("ccc"),
            chunks: &[chunk_in("src/a.rs"), chunk_in("src/a.rs")],
            embeddings: &[vec![0.5; 4], vec![0.5; 4]],
        }).unwrap();

        let stats = get_db_stats(&conn).unwrap();
        assert_eq!(stats.indexed_files, 2);
        assert_eq!(stats.total_chunks, 3);
    }

    #[test]
    fn an_incomplete_file_is_recorded_without_a_hash() {
        let mut conn = index_with(&[("src/a.rs", "aaa")]);
        replace_file_chunks(&mut conn, &FileUpdate {
            path: "src/b.rs",
            content_hash: None,
            chunks: &[chunk_in("src/b.rs")],
            embeddings: &[vec![0.5; 4]],
        }).unwrap();
        assert_eq!(read_file_hashes(&conn).unwrap()["src/b.rs"], None);
    }

    #[test]
    fn removing_a_file_drops_its_chunks_too() {
        let mut conn = index_with(&[("src/a.rs", "aaa"), ("src/b.rs", "bbb")]);
        remove_files(&mut conn, &["src/a.rs".to_string()]).unwrap();

        let stats = get_db_stats(&conn).unwrap();
        assert_eq!(stats.indexed_files, 1);
        assert_eq!(stats.total_chunks, 1);
    }

    #[test]
    fn chunks_left_by_older_versions_are_deleted() {
        let conn = index_with(&[("src/a.rs", "aaa")]);
        conn.execute(
            "INSERT INTO chunks (file_path, name, chunk_type, content, start_line, end_line, embedding)
             VALUES ('src/untracked.rs', 'c', 'function', 'fn c() {}', 1, 1, x'00')",
            [],
        ).unwrap();

        assert_eq!(delete_untracked_chunks(&conn).unwrap(), 1);
        assert_eq!(get_db_stats(&conn).unwrap().total_chunks, 1);
    }

    #[test]
    fn a_reset_empties_the_index_and_records_the_model() {
        let mut conn = index_with(&[("src/a.rs", "aaa")]);
        reset_index(&mut conn, "nomic-embed-text").unwrap();

        assert_eq!(get_db_stats(&conn).unwrap().total_chunks, 0);
        assert!(read_file_hashes(&conn).unwrap().is_empty());
        assert!(crate::db::index_metadata::ensure_index_model_matches(&conn, "nomic-embed-text").is_ok());
    }

    #[test]
    fn extensions_are_listed_once_each() {
        let conn = index_with(&[("src/a.rs", "a"), ("src/b.rs", "b"), ("web/c.TS", "c")]);
        assert_eq!(get_db_stats(&conn).unwrap().file_extensions, vec!["rs".to_string(), "ts".to_string()]);
    }
}
