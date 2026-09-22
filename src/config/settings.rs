use serde::{Serialize, Deserialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub ollama: OllamaConfig,
    pub search: SearchConfig,
}

fn default_chat_model() -> String {
    "qwen2.5:1.5b".to_string()
}

// One request at a time suits a single-GPU or CPU server, which processes them serially anyway.
fn default_parallelism() -> usize {
    1
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
    pub embedding_model: String,
    #[serde(default = "default_chat_model")]
    pub chat_model: String,
    /// How many embedding requests may be in flight at once; raise it only if OLLAMA_NUM_PARALLEL is above 1.
    #[serde(default = "default_parallelism")]
    pub parallelism: usize,
    /// Whether cbq may send code to an Ollama server that isn't on this machine.
    #[serde(default)]
    pub allow_remote: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchConfig {
    pub top_k: usize,
    pub similarity_threshold: f64,
    /// Have the chat model re-order candidates before answering; costs one extra model call per question.
    #[serde(default)]
    pub rerank: bool,
}

/// Returns the directory holding every index, config file and export cbq writes.
pub fn cbq_home() -> Result<PathBuf, anyhow::Error> {
    let home_dir = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .map_err(|_| anyhow::anyhow!("Could not determine the home directory"))?;
    Ok(home_dir.join(".cbq"))
}

pub fn get_config_path() -> Result<PathBuf, anyhow::Error> {
    let cbq_dir = cbq_home()?;
    fs::create_dir_all(&cbq_dir)?;
    restrict_to_owner(&cbq_dir)?;
    Ok(cbq_dir.join("config.toml"))
}

/// Makes a file or directory readable only by the user, since indexes hold their source code.
pub fn restrict_to_owner(path: &Path) -> Result<(), anyhow::Error> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let owner_only = match path.is_dir() {
            true => 0o700,
            false => 0o600,
        };
        fs::set_permissions(path, fs::Permissions::from_mode(owner_only))?;
    }
    // Windows inherits the user profile's permissions, which are already per-user.
    let _ = path;
    Ok(())
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
    fs::write(&path, content)?;
    restrict_to_owner(&path)?;
    Ok(())
}