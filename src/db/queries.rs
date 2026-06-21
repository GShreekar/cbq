use rusqlite::{params, Connection};
use crate::services::chunker::CodeChunk;

pub fn clear_chunks(conn: &Connection) -> Result<(), anyhow::Error> {
    conn.execute("DELETE FROM chunks", [])?;
    Ok(())
}

pub fn insert_chunks(conn: &mut Connection, chunks: &[CodeChunk]) -> Result<(), anyhow::Error> {
    let tx = conn.transaction()?;

    {
        let mut stmt = tx.prepare(
            "INSERT INTO chunks (file_path, name, chunk_type, content, start_line, end_line)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)"
        )?;

        for chunk in chunks {
            stmt.execute(params![
                chunk.file_path.to_string_lossy().to_string(),
                chunk.name,
                chunk.chunk_type,
                chunk.content,
                chunk.start_line as i64,
                chunk.end_line as i64,
            ])?;
        }
    }

    tx.commit()?;
    Ok(())
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