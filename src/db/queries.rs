use std::collections::HashMap;
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

pub struct DbStats {
    pub total_chunks: usize,
    pub distinct_types: Vec<(String, usize)>,
}

pub fn get_db_stats(conn: &Connection) -> Result<DbStats, anyhow::Error> {
    let total_chunks: usize = conn.query_row(
        "SELECT COUNT(*) FROM chunks",
        [],
        |row| row.get(0),
    )?;

    let mut stmt = conn.prepare(
        "SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type ORDER BY COUNT(*) DESC",
    )?;

    let type_rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
    })?;

    let mut distinct_types = Vec::new();
    for row in type_rows {
        distinct_types.push(row?);
    }

    Ok(DbStats {
        total_chunks,
        distinct_types,
    })
}