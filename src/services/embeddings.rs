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
    fn test_vector_bytes_roundtrip() {
        let original = vec![0.15, -0.982, 1.2345, 42.0];
        let bytes = vector_to_bytes(&original);
        let reconstructed = bytes_to_vector(&bytes);
        assert_eq!(original, reconstructed);
    }
}
