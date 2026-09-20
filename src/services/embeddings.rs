use crate::services::ollama::Ollama;

/// Embeds a chunk for storage, with whatever prefix the model expects for documents.
pub async fn embed_document(ollama: &Ollama, model: &str, text: &str) -> Result<Vec<f32>, anyhow::Error> {
    ollama.embed(model, &format!("{}{}", document_prefix(model), text)).await
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
