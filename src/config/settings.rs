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

/// The file a project can commit to share settings with everyone who works on it.
pub const PROJECT_CONFIG_NAME: &str = ".cbq.toml";

/// Settings a project may override. Deliberately not the Ollama address: a repository you cloned
/// must not be able to redirect where your code is sent.
#[derive(Deserialize, Debug, Default)]
pub struct ProjectOverrides {
    #[serde(default)]
    pub ollama: OllamaOverrides,
    #[serde(default)]
    pub search: SearchOverrides,
}

#[derive(Deserialize, Debug, Default)]
pub struct OllamaOverrides {
    pub embedding_model: Option<String>,
    pub chat_model: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
pub struct SearchOverrides {
    pub top_k: Option<usize>,
    pub similarity_threshold: Option<f64>,
    pub rerank: Option<bool>,
}

/// Loads the global settings, then applies whatever the project overrides.
pub fn load_config_for(directory: &Path) -> Result<Config, anyhow::Error> {
    let mut config = load_config()?;
    let Some(project_config) = find_project_config(directory) else {
        return Ok(config);
    };

    let overrides: ProjectOverrides = toml::from_str(&fs::read_to_string(&project_config)?)
        .map_err(|err| anyhow::anyhow!("{} is not valid: {}", project_config.display(), err))?;
    apply_overrides(&mut config, overrides);
    Ok(config)
}

/// Finds the project settings file covering a directory, looking upwards from it.
pub fn find_project_config(directory: &Path) -> Option<PathBuf> {
    let start = fs::canonicalize(directory).ok()?;
    start
        .ancestors()
        .map(|ancestor| ancestor.join(PROJECT_CONFIG_NAME))
        .find(|candidate| candidate.is_file())
}

fn apply_overrides(config: &mut Config, overrides: ProjectOverrides) {
    if let Some(model) = overrides.ollama.embedding_model {
        config.ollama.embedding_model = model;
    }
    if let Some(model) = overrides.ollama.chat_model {
        config.ollama.chat_model = model;
    }
    if let Some(top_k) = overrides.search.top_k {
        config.search.top_k = top_k;
    }
    if let Some(threshold) = overrides.search.similarity_threshold {
        config.search.similarity_threshold = threshold;
    }
    if let Some(rerank) = overrides.search.rerank {
        config.search.rerank = rerank;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn overrides_from(toml_text: &str) -> ProjectOverrides {
        toml::from_str(toml_text).unwrap()
    }

    #[test]
    fn a_project_can_choose_its_own_chat_model() {
        let mut config = Config::default();
        apply_overrides(&mut config, overrides_from("[ollama]\nchat_model = \"llama3.1\""));
        assert_eq!(config.ollama.chat_model, "llama3.1");
    }

    #[test]
    fn a_project_can_change_search_settings() {
        let mut config = Config::default();
        apply_overrides(&mut config, overrides_from("[search]\ntop_k = 9\nrerank = true"));
        assert_eq!(config.search.top_k, 9);
        assert!(config.search.rerank);
    }

    #[test]
    fn settings_a_project_leaves_out_keep_their_global_value() {
        let mut config = Config::default();
        let global_host = config.ollama.host.clone();
        apply_overrides(&mut config, overrides_from("[search]\ntop_k = 9"));
        assert_eq!(config.ollama.host, global_host);
        assert_eq!(config.search.similarity_threshold, Config::default().search.similarity_threshold);
    }

    #[test]
    fn a_project_cannot_redirect_where_code_is_sent() {
        // The override type has no host, port or allow_remote, so such keys are simply ignored.
        let mut config = Config::default();
        apply_overrides(&mut config, overrides_from("[ollama]\nhost = \"http://evil.example\"\nchat_model = \"x\""));
        assert_eq!(config.ollama.host, Config::default().ollama.host);
        assert!(!config.ollama.allow_remote);
    }

    #[test]
    fn an_empty_project_file_changes_nothing() {
        let mut config = Config::default();
        apply_overrides(&mut config, overrides_from(""));
        assert_eq!(config.search.top_k, Config::default().search.top_k);
    }
}
