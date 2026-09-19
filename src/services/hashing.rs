/// Hashes bytes with 64-bit FNV-1a, which unlike std's hasher never changes between Rust releases.
pub fn fnv1a_hash(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(PRIME))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_hash_matches_the_published_test_vector() {
        assert_eq!(fnv1a_hash(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
