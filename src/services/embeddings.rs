use crate::services::ollama::Ollama;

/// Embeds chunks for storage, with whatever prefix the model expects for documents.
pub async fn embed_documents(ollama: &Ollama, model: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
    let prefixed: Vec<String> = texts.iter().map(|text| format!("{}{}", document_prefix(model), text)).collect();
    ollama.embed_all(model, &prefixed).await
}

/// Embeds a question, with whatever prefix the model expects for queries.
pub async fn embed_query(ollama: &Ollama, model: &str, text: &str) -> Result<Vec<f32>, anyhow::Error> {
    ollama.embed(model, &format!("{}{}", query_prefix(model), text)).await
}

/// The prefix stored text needs for this model; recorded in the index, since changing it invalidates vectors.
pub fn document_prefix(model: &str) -> &'static str {
    match expects_task_prefixes(model) {
        true => "search_document: ",
        false => "",
    }
}

fn query_prefix(model: &str) -> &'static str {
    match expects_task_prefixes(model) {
        true => "search_query: ",
        false => "",
    }
}

// nomic-embed-text is trained with task prefixes and retrieves worse without them.
fn expects_task_prefixes(model: &str) -> bool {
    model.to_lowercase().contains("nomic-embed-text")
}

/// Scales a vector to length 1, so comparing two of them is a plain dot product.
pub fn normalise(vector: &[f32]) -> Vec<f32> {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|value| value / length).collect()
}

pub fn vector_to_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for val in vector {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

pub fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    let mut vector = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
        vector.push(f32::from_le_bytes(arr));
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_normalised_vector_has_length_one() {
        let normalised = normalise(&[3.0, 4.0]);
        let length: f32 = normalised.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalising_keeps_the_direction() {
        assert_eq!(normalise(&[3.0, 4.0]), vec![0.6, 0.8]);
    }

    #[test]
    fn an_all_zero_vector_is_left_alone() {
        assert_eq!(normalise(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn nomic_documents_and_queries_get_their_task_prefixes() {
        assert_eq!(document_prefix("nomic-embed-text:latest"), "search_document: ");
        assert_eq!(query_prefix("nomic-embed-text"), "search_query: ");
    }

    #[test]
    fn other_models_get_no_prefix() {
        assert_eq!(document_prefix("mxbai-embed-large"), "");
        assert_eq!(query_prefix("mxbai-embed-large"), "");
    }

    #[test]
    fn test_vector_bytes_roundtrip() {
        let original = vec![0.15, -0.982, 1.2345, 42.0];
        let bytes = vector_to_bytes(&original);
        let reconstructed = bytes_to_vector(&bytes);
        assert_eq!(original, reconstructed);
    }
}
