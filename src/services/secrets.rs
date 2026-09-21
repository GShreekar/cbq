// Only markers that are secrets by construction are listed here. Anything fuzzier, such as "a long
// random-looking string", would skip real source files far more often than it would catch a leak.

/// Names the credential a file appears to contain, if any.
pub fn find_secret(content: &str) -> Option<&'static str> {
    if content.contains("-----BEGIN") && content.contains("PRIVATE KEY-----") {
        return Some("a private key");
    }
    if contains_token(content, "AKIA", 16, |character| character.is_ascii_uppercase() || character.is_ascii_digit()) {
        return Some("an AWS access key");
    }
    if contains_token(content, "ghp_", 36, char::is_alphanumeric)
        || contains_token(content, "github_pat_", 22, |character| character.is_alphanumeric() || character == '_')
    {
        return Some("a GitHub token");
    }
    if contains_token(content, "xoxb-", 10, |character| character.is_alphanumeric() || character == '-') {
        return Some("a Slack token");
    }
    if content.contains("PRIVATE KEY BLOCK-----") {
        return Some("a PGP private key");
    }
    None
}

/// Reports whether a file's name marks it as holding credentials rather than code.
pub fn is_secret_file_name(file_name: &str) -> bool {
    let name = file_name.to_lowercase();
    if name.starts_with(".env") || matches!(name.as_str(), ".netrc" | ".npmrc" | ".pgpass" | "credentials") {
        return true;
    }
    if ["id_rsa", "id_dsa", "id_ecdsa", "id_ed25519"].iter().any(|key| name.starts_with(key)) {
        return true;
    }
    if [".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"].iter().any(|suffix| name.ends_with(suffix)) {
        return true;
    }

    // Source files are judged by their contents instead, so that `secrets.rs` still gets indexed.
    let is_data_file = [".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg", ".xml"]
        .iter()
        .any(|suffix| name.ends_with(suffix));
    is_data_file && ["secret", "credential", "password"].iter().any(|word| name.contains(word))
}

// Checks that `length` characters of the right shape follow the prefix, so prose mentioning it doesn't match.
fn contains_token(content: &str, prefix: &str, length: usize, is_allowed: fn(char) -> bool) -> bool {
    content.match_indices(prefix).any(|(position, _)| {
        let rest = &content[position + prefix.len()..];
        rest.chars().take(length).filter(|character| is_allowed(*character)).count() == length
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_private_key_block_is_found() {
        let content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----";
        assert_eq!(find_secret(content), Some("a private key"));
    }

    #[test]
    fn an_aws_access_key_is_found() {
        assert_eq!(find_secret("aws_access_key_id = AKIAIOSFODNN7EXAMPLE"), Some("an AWS access key"));
    }

    #[test]
    fn a_github_token_is_found() {
        assert_eq!(find_secret("token: ghp_012345678901234567890123456789012345"), Some("a GitHub token"));
    }

    #[test]
    fn ordinary_code_holds_no_secret() {
        assert_eq!(find_secret("pub fn load_key(path: &Path) -> Key { Key::from_file(path) }"), None);
    }

    #[test]
    fn prose_mentioning_a_prefix_is_not_a_secret() {
        assert_eq!(find_secret("// AWS keys start with AKIA and are rejected here"), None);
    }

    #[test]
    fn credential_files_are_named_as_such() {
        assert!(is_secret_file_name(".env.local"));
        assert!(is_secret_file_name("server.pem"));
        assert!(is_secret_file_name("id_rsa"));
        assert!(is_secret_file_name("gcp-credentials.json"));
    }

    #[test]
    fn source_files_are_judged_by_their_contents_not_their_name() {
        assert!(!is_secret_file_name("secrets.rs"));
        assert!(!is_secret_file_name("password_strength.py"));
        assert!(!is_secret_file_name("main.rs"));
    }
}
