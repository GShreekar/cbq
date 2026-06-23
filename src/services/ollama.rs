use serde::{Serialize, Deserialize};

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