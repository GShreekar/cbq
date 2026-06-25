use serde::{Serialize, Deserialize};
use futures_util::StreamExt;

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

pub async fn generate_embedding(
    host: &str,
    port: u16,
    model: &str,
    prompt: &str,
) -> Result<Vec<f32>, anyhow::Error> {
    let client = reqwest::Client::new();
    let url = format!("{}:{}/api/embeddings", host, port);

    let response = client
        .post(&url)
        .json(&EmbedRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
        })
        .send()
        .await?
        .error_for_status()?
        .json::<EmbedResponse>()
        .await?;

    Ok(response.embedding)
}

pub async fn check_ollama_status(host: &str, port: u16) -> bool {
    let client = reqwest::Client::new();
    let url = format!("{}:{}/api/tags", host, port);
    client.get(&url).send().await.is_ok()
}

pub async fn generate_response_stream<F>(
    host: &str,
    port: u16,
    model: &str,
    prompt: &str,
    mut on_chunk: F,
) -> Result<(), anyhow::Error> where F: FnMut(&str), {
    let client = reqwest::Client::new();
    let url = format!("{}:{}/api/generate", host, port);

    let payload = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "stream": true,
    });

    let response = client.post(&url)
        .json(&payload)
        .send()
        .await?
        .error_for_status()?;

    let mut stream = response.bytes_stream();
    let mut buffer = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        buffer.extend_from_slice(&chunk);

        while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
            let line_bytes = buffer.drain(..=pos).collect::<Vec<u8>>();
            let line = String::from_utf8_lossy(&line_bytes);
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
                if let Some(response_chunk) = json.get("response").and_then(|v| v.as_str()) {
                    on_chunk(response_chunk);
                }
            }
        }
    }

    Ok(())
}