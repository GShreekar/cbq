use std::fs;
use std::path::{Path, PathBuf};
use rusqlite::Connection;

pub fn get_db_path(project_path: &Path) -> Result<PathBuf, anyhow::Error> {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| anyhow::anyhow!("Could not determine the home directory"))?;

    let project_name = project_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("default-project");

    let db_dir = home_dir.join(".cbq").join("codebases").join(project_name);

    fs::create_dir_all(&db_dir)?;

    Ok(db_dir.join("db.sqlite"))
}

pub fn init_db(db_path: &Path) -> Result<Connection, anyhow::Error> {
    let conn = Connection::open(db_path)?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            name TEXT NOT NULL,
            chunk_type TEXT NOT NULL,
            content TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            embedding BLOB
        )",
        [],
    )?;

    Ok(conn)
}