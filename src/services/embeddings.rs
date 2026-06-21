use std::collections::HashSet;

pub fn get_search_tokens(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .filter(|s| s.len() > 1)
        .collect()
}
