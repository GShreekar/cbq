use serde::{Serialize, Deserialize};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub ollama: OllamaConfig,
    pub search: SearchConfig,
}

fn default_chat_model() -> String {
    "qwen2.5:1.5b".to_string()
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
    pub embedding_model: String,
    #[serde(default = "default_chat_model")]
    pub chat_model: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchConfig {
    pub top_k: usize,
    pub similarity_threshold: f64,
}

pub fn get_config_path() -> Result<PathBuf, anyhow::Error> {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| anyhow::anyhow!("Could not determine the home directory"))?;

    let qb_dir = home_dir.join(".cbq");
    fs::create_dir_all(&qb_dir)?;

    Ok(qb_dir.join("config.toml"))
}

pub fn load_config() -> Result<Config, anyhow::Error> {
    let path = get_config_path()?;
    if !path.exists() {
        return Ok(Config::default());
    }
    
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

pub fn save_config(config: &Config) -> Result<(), anyhow::Error> {
    let path = get_config_path()?;
    let content = toml::to_string_pretty(config)?;
    fs::write(path, content)?;
    Ok(())
}