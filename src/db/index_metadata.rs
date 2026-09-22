use rusqlite::{Connection, OptionalExtension, params};
use crate::services::ollama::is_same_model;

pub const EMBEDDING_MODEL_KEY: &str = "embedding_model";
/// The prefix the embedded text carried; changing it makes stored vectors incomparable.
pub const DOCUMENT_PREFIX_KEY: &str = "document_prefix";
/// How chunks were built; changing it means the stored text no longer matches what cbq would produce.
pub const CHUNK_FORMAT_KEY: &str = "chunk_format";
/// The directory the index was built from, so `cbq list` can name it.
pub const PROJECT_ROOT_KEY: &str = "project_root";
/// How the call and import graph was built; changing it rebuilds the graph without re-embedding.
pub const GRAPH_VERSION_KEY: &str = "graph_version";
/// Which build of the keyword index the chunks were fed into.
pub const KEYWORD_INDEX_KEY: &str = "keyword_index";

/// Records a fact about how the index was built.
pub fn write_meta(conn: &Connection, key: &str, value: &str) -> Result<(), anyhow::Error> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

/// Reads a fact about how the index was built, if it was recorded.
pub fn read_meta(conn: &Connection, key: &str) -> Result<Option<String>, anyhow::Error> {
    let has_meta_table: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta')",
        [],
        |row| row.get(0),
    )?;
    if !has_meta_table {
        return Ok(None);
    }
    let value = conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |row| row.get(0))
        .optional()?;
    Ok(value)
}

/// Records which embedding model produced the vectors stored in an index.
pub fn write_embedding_model(conn: &Connection, model: &str) -> Result<(), anyhow::Error> {
    write_meta(conn, EMBEDDING_MODEL_KEY, model)
}

/// Fails with re-index instructions if the index was built with a different embedding model.
pub fn ensure_index_model_matches(conn: &Connection, configured_model: &str) -> Result<(), anyhow::Error> {
    let Some(indexed_model) = read_embedding_model(conn)? else {
        return Ok(()); // built before cbq recorded models; search still catches mismatched vector sizes
    };
    if is_same_model(&indexed_model, configured_model) {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "This index was built with embedding model '{indexed_model}', but the configured model is \
         '{configured_model}'.\nVectors from different models can't be compared. Rebuild the index \
         with `cbq index`, or switch back with:\n  cbq config set ollama.embedding_model {indexed_model}"
    ))
}

/// Reads the embedding model an index was built with, if one was recorded.
pub fn read_embedding_model(conn: &Connection) -> Result<Option<String>, anyhow::Error> {
    read_meta(conn, EMBEDDING_MODEL_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::schema::create_tables;

    fn index_built_with(model: &str) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create_tables(&conn).unwrap();
        write_embedding_model(&conn, model).unwrap();
        conn
    }

    #[test]
    fn matching_model_is_accepted() {
        let conn = index_built_with("nomic-embed-text");
        assert!(ensure_index_model_matches(&conn, "nomic-embed-text").is_ok());
    }

    #[test]
    fn different_model_is_rejected() {
        let conn = index_built_with("nomic-embed-text");
        assert!(ensure_index_model_matches(&conn, "mxbai-embed-large").is_err());
    }

    #[test]
    fn latest_tag_counts_as_the_same_model() {
        let conn = index_built_with("nomic-embed-text");
        assert!(ensure_index_model_matches(&conn, "nomic-embed-text:latest").is_ok());
    }

    #[test]
    fn rewriting_the_model_replaces_the_old_one() {
        let conn = index_built_with("nomic-embed-text");
        write_embedding_model(&conn, "mxbai-embed-large").unwrap();
        assert!(ensure_index_model_matches(&conn, "mxbai-embed-large").is_ok());
    }

    #[test]
    fn index_without_meta_table_is_accepted() {
        let conn = Connection::open_in_memory().unwrap();
        assert!(ensure_index_model_matches(&conn, "nomic-embed-text").is_ok());
    }
}
