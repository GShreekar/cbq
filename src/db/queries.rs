use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use rusqlite::{params, Connection, OptionalExtension};
use crate::services::symbols::{self, Reference};
use crate::db::index_metadata::write_embedding_model;
use crate::services::chunker::CodeChunk;

/// One file's freshly embedded chunks, ready to replace whatever the index held for it.
pub struct FileUpdate<'a> {
    pub path: &'a str,
    // None when some chunks failed, so the next run retries the file.
    pub content_hash: Option<&'a str>,
    pub chunks: &'a [CodeChunk],
    pub embeddings: &'a [Vec<f32>],
    pub references: &'a [Reference],
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
            "INSERT INTO chunks (file_path, language, name, chunk_type, parent, content, start_line, end_line, embedding)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
        )?;

        for (chunk, emb) in update.chunks.iter().zip(update.embeddings.iter()) {
            // Stored ready to compare: scoring a query is then one multiply-add per dimension.
            let emb_bytes = crate::services::embeddings::vector_to_bytes(&crate::services::embeddings::normalise(emb));
            stmt.execute(params![
                update.path,
                chunk.language,
                chunk.name,
                chunk.chunk_type,
                chunk.parent,
                chunk.content,
                chunk.start_line as i64,
                chunk.end_line as i64,
                emb_bytes,
            ])?;
        }
    }

    replace_file_references(&tx, update.path, update.references)?;
    tx.execute(
        "INSERT INTO files (path, content_hash) VALUES (?1, ?2)
         ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash",
        params![update.path, update.content_hash],
    )?;
    tx.commit()?;
    Ok(())
}

/// Replaces the calls and imports recorded for one file.
pub fn replace_file_references(conn: &Connection, path: &str, references: &[Reference]) -> Result<(), anyhow::Error> {
    conn.execute("DELETE FROM refs WHERE file_path = ?1", [path])?;
    let mut stmt = conn.prepare("INSERT INTO refs (symbol_name, file_path, line, kind) VALUES (?1, ?2, ?3, ?4)")?;
    for reference in references {
        stmt.execute(params![reference.symbol_name, path, reference.line as i64, reference.kind])?;
    }
    Ok(())
}

/// Rebuilds the whole call and import graph, for indexes built before cbq recorded one.
pub fn replace_all_references(conn: &mut Connection, by_file: &[(String, Vec<Reference>)]) -> Result<(), anyhow::Error> {
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM refs", [])?;
    {
        let mut stmt = tx.prepare("INSERT INTO refs (symbol_name, file_path, line, kind) VALUES (?1, ?2, ?3, ?4)")?;
        for (path, references) in by_file {
            for reference in references {
                stmt.execute(params![reference.symbol_name, path, reference.line as i64, reference.kind])?;
            }
        }
    }
    tx.commit()?;
    Ok(())
}

/// Finds where a symbol is defined. A qualified name such as `Cart::add` narrows it to that type.
pub fn find_definitions(conn: &Connection, symbol: &str) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let (parent, name) = match symbol.rsplit_once("::") {
        Some((parent, name)) => (Some(parent), name),
        None => (None, symbol),
    };
    let mut stmt = conn.prepare(
        "SELECT id, file_path, language, name, chunk_type, parent, content, start_line, end_line
         FROM chunks
         WHERE name = ?1 AND chunk_type NOT IN ('module', 'general') AND (?2 IS NULL OR parent = ?2)
         ORDER BY file_path, start_line",
    )?;
    let rows = stmt.query_map(params![name, parent], |row| Ok(read_chunk(row)?.1))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Finds every place a symbol is used, optionally only the calls.
pub fn find_references(conn: &Connection, symbol: &str, only_calls: bool) -> Result<Vec<Reference>, anyhow::Error> {
    let name = symbol.rsplit("::").next().unwrap_or(symbol);
    let mut stmt = conn.prepare(
        "SELECT symbol_name, file_path, line, kind FROM refs
         WHERE symbol_name = ?1 AND (?2 = 0 OR kind = 'call')
         ORDER BY file_path, line",
    )?;
    let rows = stmt.query_map(params![name, only_calls as i64], |row| {
        let kind: String = row.get(3)?;
        Ok(Reference {
            symbol_name: row.get(0)?,
            file_path: PathBuf::from(row.get::<_, String>(1)?),
            line: row.get(2)?,
            // The column only ever holds the two kinds the parser records.
            kind: if kind == symbols::CALL { symbols::CALL } else { symbols::IMPORT },
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Names the symbols called from within a range of lines, which is what that code depends on.
pub fn called_symbols_in_range(
    conn: &Connection,
    file_path: &str,
    start_line: usize,
    end_line: usize,
) -> Result<Vec<String>, anyhow::Error> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT symbol_name FROM refs
         WHERE file_path = ?1 AND line BETWEEN ?2 AND ?3 AND kind = 'call'",
    )?;
    let rows = stmt.query_map(params![file_path, start_line as i64, end_line as i64], |row| row.get(0))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Finds the ids of the chunks defining a symbol.
pub fn find_definition_ids(conn: &Connection, symbol: &str) -> Result<Vec<i64>, anyhow::Error> {
    let mut stmt = conn.prepare(
        "SELECT id FROM chunks WHERE name = ?1 AND chunk_type NOT IN ('module', 'general') ORDER BY id",
    )?;
    let rows = stmt.query_map([symbol], |row| row.get(0))?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Names the symbols whose bodies overlap a range of lines, which is what a diff hunk touches.
pub fn find_symbols_in_range(
    conn: &Connection,
    file_path: &str,
    start_line: usize,
    end_line: usize,
) -> Result<Vec<String>, anyhow::Error> {
    let mut stmt = conn.prepare(
        "SELECT name, parent FROM chunks
         WHERE file_path = ?1 AND start_line <= ?3 AND end_line >= ?2
           AND chunk_type NOT IN ('module', 'general')
         ORDER BY start_line",
    )?;
    let rows = stmt.query_map(params![file_path, start_line as i64, end_line as i64], |row| {
        let name: String = row.get(0)?;
        let parent: Option<String> = row.get(1)?;
        Ok(match parent {
            Some(parent) => format!("{}::{}", parent, name),
            None => name,
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Names the smallest symbol whose body contains a line, which is the symbol doing the calling.
pub fn enclosing_symbol(conn: &Connection, file_path: &str, line: usize) -> Result<Option<String>, anyhow::Error> {
    let name = conn
        .query_row(
            "SELECT name, parent FROM chunks
             WHERE file_path = ?1 AND start_line <= ?2 AND end_line >= ?2 AND chunk_type NOT IN ('module', 'general')
             ORDER BY (end_line - start_line) ASC LIMIT 1",
            params![file_path, line as i64],
            |row| {
                let name: String = row.get(0)?;
                let parent: Option<String> = row.get(1)?;
                Ok(match parent {
                    Some(parent) => format!("{}::{}", parent, name),
                    None => name,
                })
            },
        )
        .optional()?;
    Ok(name)
}

/// Deletes files that no longer exist in the project, along with their chunks.
pub fn remove_files(conn: &mut Connection, paths: &[String]) -> Result<(), anyhow::Error> {
    let tx = conn.transaction()?;
    for path in paths {
        tx.execute("DELETE FROM chunks WHERE file_path = ?1", [path])?;
        tx.execute("DELETE FROM refs WHERE file_path = ?1", [path])?;
        tx.execute("DELETE FROM files WHERE path = ?1", [path])?;
    }
    tx.commit()?;
    Ok(())
}

/// Empties the index and records the embedding model the rebuilt index will use.
pub fn reset_index(conn: &mut Connection, embedding_model: &str) -> Result<(), anyhow::Error> {
    let tx = conn.transaction()?;
    tx.execute("DELETE FROM chunks", [])?;
    tx.execute("DELETE FROM refs", [])?;
    tx.execute("DELETE FROM files", [])?;
    write_embedding_model(&tx, embedding_model)?;
    tx.commit()?;
    Ok(())
}

/// Deletes chunks that belong to no tracked file, such as those left by indexes built before file tracking.
pub fn delete_untracked_chunks(conn: &Connection) -> Result<usize, anyhow::Error> {
    conn.execute("DELETE FROM refs WHERE file_path NOT IN (SELECT path FROM files)", [])?;
    Ok(conn.execute("DELETE FROM chunks WHERE file_path NOT IN (SELECT path FROM files)", [])?)
}

/// Fetches whole chunks by id, which is how search avoids loading source text it won't return.
pub fn chunks_by_ids(conn: &Connection, ids: &[i64]) -> Result<HashMap<i64, CodeChunk>, anyhow::Error> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = vec!["?"; ids.len()].join(",");
    let mut stmt = conn.prepare(&format!(
        "SELECT id, file_path, language, name, chunk_type, parent, content, start_line, end_line
         FROM chunks WHERE id IN ({})",
        placeholders
    ))?;

    let rows = stmt.query_map(rusqlite::params_from_iter(ids), read_chunk)?;
    Ok(rows.collect::<Result<_, _>>()?)
}

fn read_chunk(row: &rusqlite::Row) -> rusqlite::Result<(i64, CodeChunk)> {
    let file_path: String = row.get(1)?;
    Ok((
        row.get(0)?,
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
    ))
}

/// Returns the ids of chunks matching the keywords, best first, ranked by BM25.
pub fn keyword_matches(conn: &Connection, keyword_query: &str, limit: usize) -> Result<Vec<i64>, anyhow::Error> {
    // Names outweigh bodies: a query naming a symbol should find where it is defined.
    let mut stmt = conn.prepare(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY bm25(chunks_fts, 10.0, 1.0, 2.0) LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![keyword_query, limit], |row| row.get(0))?;
    Ok(rows.collect::<Result<_, _>>()?)
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
    use crate::db::schema::create_tables;
    use crate::services::chunker::CodeChunk;

    fn chunk_in(path: &str) -> CodeChunk {
        CodeChunk {
            file_path: PathBuf::from(path),
            language: "rust".to_string(),
            name: "chunk".to_string(),
            chunk_type: "function".to_string(),
            parent: None,
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
                references: &[],
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
            references: &[],
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
            references: &[],
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
