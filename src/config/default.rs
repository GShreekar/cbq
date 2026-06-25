use crate::config::settings::{Config, OllamaConfig, SearchConfig};

impl Default for Config {
    fn default() -> Self {
        Config {
            ollama: OllamaConfig::default(),
            search: SearchConfig::default(),
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        OllamaConfig {
            host: "http://localhost".to_string(),
            port: 11434,
            embedding_model: "nomic-embed-text".to_string(),
            chat_model: "qwen2.5:1.5b".to_string(),
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            top_k: 5,
            similarity_threshold: 0.5,
        }
    }
}