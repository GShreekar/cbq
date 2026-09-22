use std::fs;
use std::path::Path;
use rusqlite::{Connection, OpenFlags};
use crate::config::settings::{cbq_home, restrict_to_owner};
use crate::db::index_metadata::{read_meta, write_meta, KEYWORD_INDEX_KEY};

// Bumping this rebuilds the keyword index, which also repairs one left inconsistent.
const KEYWORD_INDEX_VERSION: &str = "2-porter";

/// Creates the index database and its directory if needed, ready for indexing.
pub fn init_db(db_path: &Path) -> Result<Connection, anyhow::Error> {
    if let Some(db_dir) = db_path.parent() {
        fs::create_dir_all(db_dir)?;
        // An index holds the project's source code, so only its owner may read it.
        restrict_to_owner(&cbq_home()?)?;
        restrict_to_owner(db_dir)?;
    }
    let conn = Connection::open(db_path)?;
    restrict_to_owner(db_path)?;
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

/// Creates the index tables if they don't exist yet, and adds columns older indexes lack.
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
        CREATE TABLE IF NOT EXISTS refs (
            symbol_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line INTEGER NOT NULL,
            kind TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS refs_by_symbol ON refs(symbol_name);
        CREATE INDEX IF NOT EXISTS refs_by_file ON refs(file_path);
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            asked_at TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            citations TEXT NOT NULL
        );",
    )?;

    add_missing_column(conn, "chunks", "language", "TEXT NOT NULL DEFAULT ''")?;
    add_missing_column(conn, "chunks", "parent", "TEXT")?;
    ensure_keyword_index(conn)?;
    Ok(())
}

/// Creates the keyword index, rebuilding it when chunks predate it or its tokenizer changed.
pub fn ensure_keyword_index(conn: &Connection) -> Result<(), anyhow::Error> {
    // Counting rows of an external-content table counts the chunks table, so the state is recorded instead.
    if read_meta(conn, KEYWORD_INDEX_KEY)?.as_deref() == Some(KEYWORD_INDEX_VERSION) {
        return Ok(());
    }

    // Keyword search complements the embeddings, which are weak on exact identifiers.
    // Porter stemming lets "skipped" answer a question asking about "skip".
    // The triggers keep the index in step with chunks, which are only ever inserted and deleted.
    conn.execute_batch(
        "DROP TABLE IF EXISTS chunks_fts;
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            name, content, file_path,
            content='chunks', content_rowid='id',
            tokenize=\"porter unicode61 remove_diacritics 0\"
        );
        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, name, content, file_path)
            VALUES (new.id, new.name, new.content, new.file_path);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, name, content, file_path)
            VALUES ('delete', old.id, old.name, old.content, old.file_path);
        END;
        INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');",
    )?;
    write_meta(conn, KEYWORD_INDEX_KEY, KEYWORD_INDEX_VERSION)?;
    Ok(())
}

fn add_missing_column(conn: &Connection, table: &str, column: &str, definition: &str) -> Result<(), anyhow::Error> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table))?;
    let mut columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
    if columns.any(|name| name.as_deref() == Ok(column)) {
        return Ok(());
    }
    conn.execute(&format!("ALTER TABLE {} ADD COLUMN {} {}", table, column, definition), [])?;
    Ok(())
}
