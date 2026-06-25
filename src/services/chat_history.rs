use serde::{Serialize, Deserialize};
use std::fs;
use std::path::PathBuf;
use chrono::{DateTime, Local};
use crate::services::vector_search::SearchResult;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatEntry {
    pub timestamp: String,
    pub query: String,
    pub results: Vec<ChatResultSnippet>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatResultSnippet {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f64,
}

pub fn get_chats_dir() -> Result<PathBuf, anyhow::Error> {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| anyhow::anyhow!("Could not determine home directory"))?;

    let chats_dir = home_dir.join(".cbq").join("chats");
    fs::create_dir_all(&chats_dir)?;
    Ok(chats_dir)
}

pub fn save_chat(query: &str, results: &[SearchResult]) -> Result<(), anyhow::Error> {
    let chats_dir = get_chats_dir()?;
    let now: DateTime<Local> = Local::now();

    let snippet_results = results
        .iter()
        .map(|r| ChatResultSnippet {
            file_path: r.chunk.file_path.to_string_lossy().to_string(),
            start_line: r.chunk.start_line,
            end_line: r.chunk.end_line,
            score: r.score,
        })
        .collect();
    
    let entry = ChatEntry {
        timestamp: now.format("%Y-%m-%d %H:%M:%S").to_string(),
        query: query.to_string(),
        results: snippet_results,
    };

    let filename = format!("{}.json", now.timestamp_millis());
    let filepath = chats_dir.join(filename);

    let content = serde_json::to_string_pretty(&entry)?;
    fs::write(filepath, content)?;

    Ok(())
}

pub fn get_history() -> Result<Vec<ChatEntry>, anyhow::Error> {
    let chats_dir = get_chats_dir()?;
    let mut entries = Vec::new();

    if chats_dir.exists() {
        for entry in fs::read_dir(chats_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let content = fs::read_to_string(path)?;
                if let Ok(chat_entry) = serde_json::from_str::<ChatEntry>(&content) {
                    entries.push(chat_entry);
                }
            }
        }
    }

    entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    Ok(entries)
}

pub fn export_history_to_markdown() -> Result<PathBuf, anyhow::Error> {
    let history = get_history()?;
    if history.is_empty() {
        return Err(anyhow::anyhow!("No chat history to export"));
    }

    let chats_dir = get_chats_dir()?;
    let now: DateTime<Local> = Local::now();
    let file_name = format!("cbq-session-{}.md", now.format("%Y-%m-%d_%H-%M-%S"));
    let export_path = chats_dir.join(file_name);

    let mut markdown = String::new();
    markdown.push_str("# CBQ Search & Query Session History\n\n");
    markdown.push_str(&format!("Generated on: {}\n\n", now.format("%Y-%m-%d %H:%M:%S")));
    markdown.push_str("---\n\n");

    for (idx, entry) in history.iter().enumerate() {
        markdown.push_str(&format!("## {}. Query: \"{}\"\n", idx + 1, entry.query));
        markdown.push_str(&format!("*Timestamp: {}*\n\n", entry.timestamp));

        if entry.results.is_empty() {
            markdown.push_str("No relevant chunks retrieved.\n\n");
        } else {
            markdown.push_str("### Retrieved Chunks:\n");
            for res in &entry.results {
                markdown.push_str(&format!("- **{}** (Lines {}-{}) - Match Score: `{:.2}`\n", res.file_path, res.start_line, res.end_line, res.score));
            }
            markdown.push_str("\n");
        }
        markdown.push_str("---\n\n");
    }

    fs::write(&export_path, markdown)?;
    Ok(export_path)
}