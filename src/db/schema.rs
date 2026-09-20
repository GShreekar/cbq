use std::fs;
use std::path::Path;
use rusqlite::{Connection, OpenFlags};

/// Creates the index database and its directory if needed, ready for indexing.
pub fn init_db(db_path: &Path) -> Result<Connection, anyhow::Error> {
    if let Some(db_dir) = db_path.parent() {
        fs::create_dir_all(db_dir)?;
    }
    let conn = Connection::open(db_path)?;
    create_tables(&conn)?;
    Ok(conn)
}

/// Opens an existing index database for querying, adding any tables it predates.
pub fn open_index(db_path: &Path) -> Result<Connection, anyhow::Error> {
    // Read-write because questions and answers are recorded back into the project's own index.
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    create_tables(&conn)?;
    Ok(conn)
}

/// Creates the index tables if they don't exist yet.
pub fn create_tables(conn: &Connection) -> Result<(), anyhow::Error> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            name TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            content TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            embedding BLOB
        );
        CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_path);
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            content_hash TEXT
        );
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            asked_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            citations TEXT NOT NULL
        );",
    )?;
    Ok(())
}
