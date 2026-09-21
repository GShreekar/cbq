use std::future::Future;
use std::time::Duration;
use futures_util::StreamExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

// Ollama's default 4,096-token window silently drops the start of longer prompts, rules included.
// Fixed rather than per-prompt, because changing it forces Ollama to reload the model.
const CHAT_CONTEXT_TOKENS: u32 = 16_384;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const STATUS_TIMEOUT: Duration = Duration::from_secs(10);
// Generous because a busy server queues requests, e.g. an embedding behind another client's long prompt.
const EMBED_TIMEOUT: Duration = Duration::from_secs(300);
// Ollama sends nothing until the prompt is evaluated: 82s for 6k tokens on a 1.5B model on CPU.
const FIRST_RESPONSE_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// A client for one Ollama server, sharing a connection pool across every request cbq makes.
pub struct Ollama {
    http: reqwest::Client,
    base_url: String,
}

/// One progress update streamed while Ollama downloads a model.
#[derive(Debug, Deserialize)]
pub struct PullProgress {
    pub status: String,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Deserialize)]
struct TagsResponse {
    models: Vec<InstalledModel>,
}

#[derive(Deserialize)]
struct InstalledModel {
    name: String,
}

// Streamed lines either carry data or, even on HTTP 200, an error that ends the stream.
#[derive(Deserialize)]
struct StreamLine<T> {
    #[serde(flatten)]
    data: Option<T>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct GenerateChunk {
    response: String,
}

impl Ollama {
    /// Creates a client for the server at `host`:`port`, rejecting addresses that aren't http(s) URLs.
    pub fn new(host: &str, port: u16) -> Result<Self, anyhow::Error> {
        let base_url = format!("{}:{}", host.trim_end_matches('/'), port);
        let scheme = reqwest::Url::parse(&base_url)
            .ok()
            .filter(|url| url.path() == "/")
            .map(|url| url.scheme().to_string());

        match scheme.as_deref() {
            Some("http") => {}
            // Building without TLS keeps OpenSSL out of the install; say so rather than fail at request time.
            Some("https") if !cfg!(feature = "https") => anyhow::bail!(
                "This build of cbq speaks plain HTTP only, so it cannot reach '{}'. \
                 Reinstall with `cargo install cbq --features https` to allow https addresses.",
                base_url
            ),
            Some("https") => {}
            _ => anyhow::bail!(
                "Invalid Ollama address '{}'. Set ollama.host to something like http://localhost",
                base_url
            ),
        }
        let http = reqwest::Client::builder().connect_timeout(CONNECT_TIMEOUT).build()?;
        Ok(Self { http, base_url })
    }

    /// Returns the server address, for messages shown to the user.
    pub fn address(&self) -> &str {
        &self.base_url
    }

    /// Reports whether the server runs on this machine, so nothing sent to it leaves the machine.
    pub fn is_local(&self) -> bool {
        let Ok(url) = reqwest::Url::parse(&self.base_url) else {
            return false;
        };
        let Some(host) = url.host_str() else {
            return false;
        };
        // A URL writes IPv6 addresses inside brackets.
        let host = host.trim_start_matches('[').trim_end_matches(']');
        if host.eq_ignore_ascii_case("localhost") {
            return true;
        }
        host.parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback() || address.is_unspecified())
    }

    /// Reports whether traffic to the server is encrypted.
    pub fn is_encrypted(&self) -> bool {
        self.base_url.starts_with("https://")
    }

    /// Lists the models on the server, which also confirms an Ollama server is answering here.
    pub async fn installed_models(&self) -> Result<Vec<String>, anyhow::Error> {
        let response = self
            .http
            .get(self.url("/api/tags"))
            .timeout(STATUS_TIMEOUT)
            .send()
            .await
            .map_err(|err| anyhow::anyhow!("Ollama is not reachable at {}: {}", self.base_url, describe(&err)))?;
        if !response.status().is_success() {
            anyhow::bail!(
                "{} answered with HTTP {}; is ollama.host pointing at an Ollama server?",
                self.base_url,
                response.status()
            );
        }

        let tags = response.json::<TagsResponse>().await.map_err(|_| {
            anyhow::anyhow!("{} answered, but not like an Ollama server; check ollama.host", self.base_url)
        })?;
        Ok(tags.models.into_iter().map(|model| model.name).collect())
    }

    /// Downloads `model` onto the server, reporting progress as it streams in.
    pub async fn pull_model(&self, model: &str, mut on_progress: impl FnMut(&PullProgress)) -> Result<(), anyhow::Error> {
        let request = self.http.post(self.url("/api/pull")).json(&serde_json::json!({ "model": model, "stream": true }));
        let response = send_within(request.send(), STREAM_IDLE_TIMEOUT).await?.error_for_status()?;

        let mut succeeded = false;
        read_json_lines(response, |progress: PullProgress| {
            succeeded = progress.status == "success";
            on_progress(&progress);
        })
        .await
        .map_err(|err| err.context(format!("Failed to pull model '{}'", model)))?;

        if !succeeded {
            anyhow::bail!("Pulling model '{}' stopped before it finished", model);
        }
        Ok(())
    }

    /// Embeds several texts in one request, truncating input longer than the model's context instead of failing.
    pub async fn embed_all(&self, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let response = self
            .http
            .post(self.url("/api/embed"))
            .timeout(EMBED_TIMEOUT)
            .json(&EmbedRequest { model, input: texts })
            .send()
            .await?
            .error_for_status()?
            .json::<EmbedResponse>()
            .await?;

        anyhow::ensure!(
            response.embeddings.len() == texts.len(),
            "Asked Ollama to embed {} texts but got {} vectors back",
            texts.len(),
            response.embeddings.len()
        );
        Ok(response.embeddings)
    }

    /// Embeds one text with `model`.
    pub async fn embed(&self, model: &str, text: &str) -> Result<Vec<f32>, anyhow::Error> {
        self.embed_all(model, std::slice::from_ref(&text.to_string()))
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Ollama returned no embedding for model '{}'", model))
    }

    /// Streams a response to `prompt`, calling `on_chunk` with each piece of text as it arrives.
    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        mut on_chunk: impl FnMut(&str),
    ) -> Result<(), anyhow::Error> {
        let payload = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
            "options": { "num_ctx": CHAT_CONTEXT_TOKENS },
        });
        let request = self.http.post(self.url("/api/generate")).json(&payload);
        let response = send_within(request.send(), FIRST_RESPONSE_TIMEOUT).await?.error_for_status()?;

        read_json_lines(response, |chunk: GenerateChunk| on_chunk(&chunk.response)).await
    }

    /// Generates a complete response with no sampling randomness, for internal steps like rewriting a question.
    pub async fn generate_deterministic(&self, model: &str, prompt: &str) -> Result<String, anyhow::Error> {
        let payload = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": { "num_ctx": CHAT_CONTEXT_TOKENS, "temperature": 0 },
        });
        let response = self
            .http
            .post(self.url("/api/generate"))
            .timeout(FIRST_RESPONSE_TIMEOUT)
            .json(&payload)
            .send()
            .await?
            .error_for_status()?
            .json::<GenerateResponse>()
            .await?;
        Ok(response.response)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

/// Reports whether two model names refer to the same model; Ollama resolves a bare name to its ":latest" tag.
pub fn is_same_model(first: &str, second: &str) -> bool {
    first.trim_end_matches(":latest") == second.trim_end_matches(":latest")
}

async fn send_within(
    request: impl Future<Output = Result<reqwest::Response, reqwest::Error>>,
    limit: Duration,
) -> Result<reqwest::Response, anyhow::Error> {
    match tokio::time::timeout(limit, request).await {
        Ok(response) => Ok(response?),
        Err(_) => anyhow::bail!("Ollama did not respond within {} seconds", limit.as_secs()),
    }
}

// Ollama streams one JSON object per line, and a line can be split across network chunks.
async fn read_json_lines<T: DeserializeOwned>(
    response: reqwest::Response,
    mut on_line: impl FnMut(T),
) -> Result<(), anyhow::Error> {
    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();
    // Where the search for the next newline resumes, so a long line isn't rescanned on every chunk.
    let mut scanned_to = 0;

    loop {
        let next = tokio::time::timeout(STREAM_IDLE_TIMEOUT, stream.next())
            .await
            .map_err(|_| anyhow::anyhow!("Ollama stopped responding for {} seconds", STREAM_IDLE_TIMEOUT.as_secs()))?;
        let Some(bytes) = next else { break };
        buffer.extend_from_slice(&bytes?);

        while let Some(offset) = buffer[scanned_to..].iter().position(|&byte| byte == b'\n') {
            let line: Vec<u8> = buffer.drain(..=scanned_to + offset).collect();
            scanned_to = 0;
            if let Some(data) = parse_stream_line(&line)? {
                on_line(data);
            }
        }
        scanned_to = buffer.len();
    }
    if let Some(data) = parse_stream_line(&buffer)? {
        on_line(data);
    }
    Ok(())
}

fn parse_stream_line<T: DeserializeOwned>(line: &[u8]) -> Result<Option<T>, anyhow::Error> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let parsed: StreamLine<T> = serde_json::from_slice(line)?;
    if let Some(error) = parsed.error {
        anyhow::bail!("Ollama reported an error: {}", error);
    }
    Ok(parsed.data)
}

fn describe(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "the connection timed out"
    } else if error.is_connect() {
        "nothing is listening there"
    } else {
        "the request failed"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_model_name_matches_its_latest_tag() {
        assert!(is_same_model("nomic-embed-text", "nomic-embed-text:latest"));
    }

    #[test]
    fn model_prefix_is_not_a_match() {
        assert!(!is_same_model("llama3", "llama3.1:latest"));
    }

    #[test]
    fn host_with_trailing_slash_is_accepted() {
        assert_eq!(Ollama::new("http://localhost/", 11434).unwrap().address(), "http://localhost:11434");
    }

    #[test]
    fn localhost_and_loopback_addresses_count_as_local() {
        assert!(Ollama::new("http://localhost", 11434).unwrap().is_local());
        assert!(Ollama::new("http://127.0.0.1", 11434).unwrap().is_local());
        assert!(Ollama::new("http://[::1]", 11434).unwrap().is_local());
    }

    #[test]
    fn another_machine_does_not_count_as_local() {
        assert!(!Ollama::new("http://192.168.1.50", 11434).unwrap().is_local());
        assert!(!Ollama::new("http://ollama.example.com", 11434).unwrap().is_local());
    }

    #[test]
    fn host_without_scheme_is_rejected() {
        assert!(Ollama::new("localhost", 11434).is_err());
    }

    #[test]
    fn host_with_a_path_is_rejected() {
        assert!(Ollama::new("http://localhost/api", 11434).is_err());
    }

    #[test]
    fn an_https_host_is_accepted_only_when_built_with_tls() {
        assert_eq!(Ollama::new("https://ollama.example", 11434).is_ok(), cfg!(feature = "https"));
    }

    #[test]
    fn error_line_becomes_an_error() {
        let result = parse_stream_line::<PullProgress>(br#"{"error":"pull model manifest: file does not exist"}"#);
        assert!(result.unwrap_err().to_string().contains("file does not exist"));
    }

    #[test]
    fn progress_line_is_parsed() {
        let line = br#"{"status":"pulling 970aa74c0a90","total":274290656,"completed":1024}"#;
        let progress = parse_stream_line::<PullProgress>(line).unwrap().unwrap();
        assert_eq!(progress.completed, Some(1024));
    }

    #[test]
    fn blank_line_is_skipped() {
        assert!(parse_stream_line::<PullProgress>(b"  \n").unwrap().is_none());
    }
}
