use rusqlite::{Connection, OptionalExtension, params};

const EMBEDDING_MODEL_KEY: &str = "embedding_model";

/// Records which embedding model produced the vectors stored in an index.
pub fn write_embedding_model(conn: &Connection, model: &str) -> Result<(), anyhow::Error> {
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![EMBEDDING_MODEL_KEY, model],
    )?;
    Ok(())
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

fn read_embedding_model(conn: &Connection) -> Result<Option<String>, anyhow::Error> {
    let has_meta_table: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta')",
        [],
        |row| row.get(0),
    )?;
    if !has_meta_table {
        return Ok(None);
    }
    let model = conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [EMBEDDING_MODEL_KEY], |row| row.get(0))
        .optional()?;
    Ok(model)
}

// Ollama resolves a bare model name to its ":latest" tag.
fn is_same_model(first: &str, second: &str) -> bool {
    first.trim_end_matches(":latest") == second.trim_end_matches(":latest")
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
