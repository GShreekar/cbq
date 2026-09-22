use std::path::Path;
use crate::config::settings::{cbq_home, load_config, Config};
use crate::db::index_metadata::{
    read_embedding_model, read_meta, CHUNK_FORMAT_KEY, DOCUMENT_PREFIX_KEY, GRAPH_VERSION_KEY, PROJECT_ROOT_KEY,
};
use crate::db::location::{canonical_project_root, find_indexed_project};
use crate::db::queries::{get_db_stats, read_file_hashes};
use crate::db::schema::open_index;
use crate::services::chunker::CHUNK_FORMAT_VERSION;
use crate::services::embeddings::{document_prefix, embed_query};
use crate::services::file_discovery::{discover_files, DiscoveryOptions};
use crate::services::index_plan::{plan_index, read_source_files};
use crate::services::ollama::{is_same_model, Ollama};
use crate::services::vector_search::load_chunk_vectors;

/// How a single check turned out.
#[derive(PartialEq)]
pub enum Status {
    Ok,
    Warning,
    Failed,
}

/// One thing cbq needs in order to work, and whether it holds.
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
}

impl Check {
    fn ok(name: &str, detail: impl Into<String>) -> Self {
        Self { name: name.to_string(), status: Status::Ok, detail: detail.into() }
    }

    fn warn(name: &str, detail: impl Into<String>) -> Self {
        Self { name: name.to_string(), status: Status::Warning, detail: detail.into() }
    }

    fn failed(name: &str, detail: impl Into<String>) -> Self {
        Self { name: name.to_string(), status: Status::Failed, detail: detail.into() }
    }
}

/// What the index holds for one project, and how far it has drifted from the files on disk.
pub struct IndexSummary {
    pub project_root: String,
    pub indexed_files: usize,
    pub total_chunks: usize,
    pub languages: Vec<String>,
    pub embedding_model: Option<String>,
    pub size_bytes: u64,
    pub indexed_at: Option<String>,
    pub new_files: usize,
    pub changed_files: usize,
    pub removed_files: usize,
}

impl IndexSummary {
    pub fn is_stale(&self) -> bool {
        self.new_files + self.changed_files + self.removed_files > 0
    }
}

/// Reads what the index holds and compares it with the files on disk. Touches no network.
pub fn summarise_index(directory: &Path) -> Result<Option<IndexSummary>, anyhow::Error> {
    let Some(project) = find_indexed_project(directory)? else {
        return Ok(None);
    };
    let conn = open_index(&project.db_path)?;
    let stats = get_db_stats(&conn)?;

    let discovery = discover_files(&project.root, &DiscoveryOptions::default())?;
    let (source_files, _) = read_source_files(&discovery.files, &project.root, false);
    let plan = plan_index(source_files, &read_file_hashes(&conn)?);

    let metadata = std::fs::metadata(&project.db_path);
    Ok(Some(IndexSummary {
        project_root: read_meta(&conn, PROJECT_ROOT_KEY)?.unwrap_or_else(|| project.root.display().to_string()),
        indexed_files: stats.indexed_files,
        total_chunks: stats.total_chunks,
        languages: stats.file_extensions,
        embedding_model: read_embedding_model(&conn)?,
        size_bytes: metadata.as_ref().map(|file| file.len()).unwrap_or(0),
        indexed_at: metadata.ok().and_then(|file| file.modified().ok()).map(format_age),
        new_files: plan.new_files.len(),
        changed_files: plan.changed_files.len(),
        removed_files: plan.removed_paths.len(),
    }))
}

/// Checks everything that has to hold for a question to be answered, and reports what doesn't.
pub async fn run_checks(directory: &Path) -> Vec<Check> {
    let mut checks = Vec::new();
    let config = match load_config() {
        Ok(config) => {
            checks.push(Check::ok("Configuration", describe_config_source()));
            config
        }
        Err(err) => {
            checks.push(Check::warn("Configuration", format!("unreadable, using defaults: {}", err)));
            Config::default()
        }
    };

    let ollama = check_ollama(&config, &mut checks).await;
    check_index(directory, &config, ollama.as_ref(), &mut checks).await;
    check_permissions(&mut checks);
    checks
}

fn describe_config_source() -> String {
    match crate::config::settings::get_config_path() {
        Ok(path) if path.exists() => format!("{}", path.display()),
        Ok(path) => format!("{} (not written yet, using defaults)", path.display()),
        Err(err) => format!("cannot locate: {}", err),
    }
}

async fn check_ollama(config: &Config, checks: &mut Vec<Check>) -> Option<Ollama> {
    let ollama = match Ollama::new(&config.ollama.host, config.ollama.port) {
        Ok(ollama) => ollama,
        Err(err) => {
            checks.push(Check::failed("Ollama address", err.to_string()));
            return None;
        }
    };

    let location = match ollama.is_local() {
        true => "on this machine".to_string(),
        false => match config.ollama.allow_remote {
            true => "NOT on this machine; your code is sent there".to_string(),
            false => "not this machine, and ollama.allow_remote is false".to_string(),
        },
    };
    match ollama.is_local() || config.ollama.allow_remote {
        true => checks.push(Check::ok("Ollama address", format!("{} ({})", ollama.address(), location))),
        false => checks.push(Check::failed("Ollama address", format!("{} is {}", ollama.address(), location))),
    }

    let installed = match ollama.installed_models().await {
        Ok(installed) => {
            checks.push(Check::ok("Ollama server", format!("reachable, {} models installed", installed.len())));
            installed
        }
        Err(err) => {
            checks.push(Check::failed("Ollama server", err.to_string()));
            return None;
        }
    };

    for (role, model) in [("Embedding model", &config.ollama.embedding_model), ("Chat model", &config.ollama.chat_model)] {
        match installed.iter().any(|name| is_same_model(name, model)) {
            true => checks.push(Check::ok(role, model.clone())),
            false => checks.push(Check::warn(role, format!("'{}' is not installed; cbq will pull it", model))),
        }
    }
    Some(ollama)
}

async fn check_index(directory: &Path, config: &Config, ollama: Option<&Ollama>, checks: &mut Vec<Check>) {
    let summary = match summarise_index(directory) {
        Ok(Some(summary)) => summary,
        Ok(None) => {
            let shown = canonical_project_root(directory).unwrap_or_else(|_| directory.to_path_buf());
            checks.push(Check::failed("Index", format!("none covers {}; run `cbq index`", shown.display())));
            return;
        }
        Err(err) => {
            checks.push(Check::failed("Index", err.to_string()));
            return;
        }
    };

    checks.push(Check::ok(
        "Index",
        format!(
            "{} files, {} chunks, {:.1} MB{}",
            summary.indexed_files,
            summary.total_chunks,
            summary.size_bytes as f64 / 1024.0 / 1024.0,
            summary.indexed_at.as_ref().map(|age| format!(", built {}", age)).unwrap_or_default()
        ),
    ));

    match summary.is_stale() {
        true => checks.push(Check::warn(
            "Index freshness",
            format!(
                "{} new, {} changed, {} removed since indexing; run `cbq index`",
                summary.new_files, summary.changed_files, summary.removed_files
            ),
        )),
        false => checks.push(Check::ok("Index freshness", "matches the files on disk")),
    }

    // Opened once here: the remaining checks all read from the same index.
    let conn = find_indexed_project(directory)
        .ok()
        .flatten()
        .and_then(|project| open_index(&project.db_path).ok());
    let Some(conn) = conn else {
        return;
    };
    check_index_model(&summary, config, ollama, &conn, checks).await;
    check_index_format(config, &conn, checks);
}

// The failure this catches is silent: vectors from another model score near zero against every query.
async fn check_index_model(
    summary: &IndexSummary,
    config: &Config,
    ollama: Option<&Ollama>,
    conn: &rusqlite::Connection,
    checks: &mut Vec<Check>,
) {
    let configured = &config.ollama.embedding_model;
    match &summary.embedding_model {
        Some(indexed) if !is_same_model(indexed, configured) => {
            checks.push(Check::failed(
                "Embedding model match",
                format!("index was built with '{}' but '{}' is configured; rebuild with `cbq index`", indexed, configured),
            ));
            return;
        }
        Some(_) => checks.push(Check::ok("Embedding model match", "index and config agree")),
        None => checks.push(Check::warn("Embedding model match", "index predates model tracking; rebuild to record it")),
    }

    let Some(ollama) = ollama else {
        return;
    };
    let dimensions = load_chunk_vectors(conn).map(|vectors| vectors.dimensions());

    match (dimensions, embed_query(ollama, configured, "cbq doctor").await) {
        (Ok(Some(stored)), Ok(query)) if stored != query.len() => checks.push(Check::failed(
            "Vector size match",
            format!("index stores {}-dimension vectors but '{}' now returns {}", stored, configured, query.len()),
        )),
        (Ok(Some(stored)), Ok(_)) => checks.push(Check::ok("Vector size match", format!("{} dimensions", stored))),
        (_, Err(err)) => checks.push(Check::failed("Vector size match", format!("could not embed a test query: {}", err))),
        _ => {}
    }
}

// Each of these means the next `cbq index` rebuilds something, which is worth knowing in advance.
fn check_index_format(config: &Config, conn: &rusqlite::Connection, checks: &mut Vec<Check>) {
    let current_format = CHUNK_FORMAT_VERSION.to_string();
    let chunk_format = read_meta(conn, CHUNK_FORMAT_KEY).ok().flatten();
    let graph_version = read_meta(conn, GRAPH_VERSION_KEY).ok().flatten();
    let stored_prefix = read_meta(conn, DOCUMENT_PREFIX_KEY).ok().flatten();

    let mut outdated = Vec::new();
    if chunk_format.as_deref() != Some(&current_format) {
        outdated.push("chunks");
    }
    if graph_version.as_deref() != Some(&current_format) {
        outdated.push("call graph");
    }
    if stored_prefix.as_deref() != Some(document_prefix(&config.ollama.embedding_model)) {
        outdated.push("embedding prefix");
    }

    match outdated.is_empty() {
        true => checks.push(Check::ok("Index format", "current")),
        false => checks.push(Check::warn(
            "Index format",
            format!(
                "{} {} this version of cbq; the next `cbq index` rebuilds {}",
                outdated.join(" and "),
                if outdated.len() == 1 { "predates" } else { "predate" },
                if outdated.len() == 1 { "it" } else { "them" }
            ),
        )),
    }
}

fn check_permissions(checks: &mut Vec<Check>) {
    let Ok(home) = cbq_home() else {
        return;
    };
    if !home.exists() {
        return;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&home).map(|data| data.permissions().mode() & 0o777);
        match mode {
            Ok(0o700) => checks.push(Check::ok("Stored data", format!("{} is private to you", home.display()))),
            Ok(mode) => checks.push(Check::warn(
                "Stored data",
                format!("{} is mode {:o}; indexes hold your source code", home.display(), mode),
            )),
            Err(err) => checks.push(Check::warn("Stored data", err.to_string())),
        }
    }
    #[cfg(not(unix))]
    checks.push(Check::ok("Stored data", format!("{}", home.display())));
}

fn format_age(modified: std::time::SystemTime) -> String {
    let Ok(elapsed) = modified.elapsed() else {
        return "just now".to_string();
    };
    let minutes = elapsed.as_secs() / 60;
    match minutes {
        0 => "just now".to_string(),
        1..=59 => format!("{} minutes ago", minutes),
        60..=1439 => format!("{} hours ago", minutes / 60),
        _ => format!("{} days ago", minutes / 1440),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn summary_with(new: usize, changed: usize, removed: usize) -> IndexSummary {
        IndexSummary {
            project_root: "/project".to_string(),
            indexed_files: 10,
            total_chunks: 100,
            languages: vec!["rust".to_string()],
            embedding_model: Some("nomic-embed-text".to_string()),
            size_bytes: 1024,
            indexed_at: None,
            new_files: new,
            changed_files: changed,
            removed_files: removed,
        }
    }

    #[test]
    fn an_index_matching_disk_is_not_stale() {
        assert!(!summary_with(0, 0, 0).is_stale());
    }

    #[test]
    fn any_difference_from_disk_makes_an_index_stale() {
        assert!(summary_with(1, 0, 0).is_stale());
        assert!(summary_with(0, 1, 0).is_stale());
        assert!(summary_with(0, 0, 1).is_stale());
    }

    #[test]
    fn a_fresh_timestamp_reads_as_just_now() {
        assert_eq!(format_age(SystemTime::now()), "just now");
    }

    #[test]
    fn an_older_timestamp_reads_in_hours() {
        assert_eq!(format_age(SystemTime::now() - Duration::from_secs(3 * 3600)), "3 hours ago");
    }

    #[test]
    fn a_much_older_timestamp_reads_in_days() {
        assert_eq!(format_age(SystemTime::now() - Duration::from_secs(50 * 3600)), "2 days ago");
    }
}
