use crate::core::fingerprint::ImageFingerprint;
use subtle::ConstantTimeEq;

/// Pre-computed population count lookup table for 8-bit values.
/// This provides faster Hamming distance computation than calling count_ones()
/// for 64-bit values, especially in hot loops.
const POPCOUNT_TABLE: [u8; 256] = compute_popcount_table();

const fn compute_popcount_table() -> [u8; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = (i as u8).count_ones() as u8;
        i += 1;
    }
    table
}

/// Computes similarity from a Hamming distance.
///
/// Returns a value from 0.0 (completely different, distance >= 64)
/// to 1.0 (identical, distance = 0).
///
/// This is a shared utility used by both single and multi-hash comparison.
#[inline(always)]
pub fn hash_similarity(distance: u32) -> f32 {
    if distance >= 64 {
        0.0
    } else {
        1.0 - (distance as f32 / 64.0)
    }
}

/// Computes Hamming distance between two 64-bit hashes using lookup table.
///
/// Uses a pre-computed population count table for faster computation
/// than the built-in count_ones() on 64-bit values.
#[inline(always)]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    let xor = a ^ b;
    // Sum population count of each byte using lookup table
    POPCOUNT_TABLE[xor as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 8) as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 16) as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 24) as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 32) as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 40) as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 48) as usize & 0xFF] as u32
        + POPCOUNT_TABLE[(xor >> 56) as usize & 0xFF] as u32
}

/// Similarity score between two image fingerprints.
///
/// The score combines exact hashing with perceptual hashing to provide
/// a robust measure of visual similarity.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity {
    /// Similarity score from 0.0 (completely different) to 1.0 (identical).
    ///
    /// Values above 0.7 generally indicate the same image with minor modifications.
    /// Values below 0.3 indicate substantially different images.
    pub score: f32,

    /// True if the images have identical BLAKE3 hashes (exact byte match).
    pub exact_match: bool,

    /// Hamming distance between global perceptual hashes (0-64).
    ///
    /// Lower values indicate higher similarity. Distance of 0 means identical
    /// perceptual hashes.
    pub perceptual_distance: u32,
}

impl Similarity {
    /// Returns a perfect similarity score (1.0, exact match).
    #[inline]
    #[must_use]
    pub fn perfect() -> Self {
        Self {
            score: 1.0,
            exact_match: true,
            perceptual_distance: 0,
        }
    }
}

/// Computes similarity between two fingerprints.
///
/// Uses a weighted combination:
/// - 40% weight on global perceptual hash similarity
/// - 60% weight on block-level hash similarity (crop resistance)
///
/// These weights were chosen empirically to balance overall image similarity
/// (global hash) with regional matching (block hashes) for crop resistance.
/// Block hashes receive higher weight because partial region matches are
/// more reliable indicators of similarity than global structure alone.
///
/// Uses constant-time comparison to prevent timing attacks on exact match.
#[must_use]
pub fn compute_similarity(a: &ImageFingerprint, b: &ImageFingerprint) -> Similarity {
    let exact_match = a.exact.ct_eq(&b.exact).into();

    let global_distance = hamming_distance(a.global_hash, b.global_hash);
    let global_similarity = hash_similarity(global_distance);
    let block_similarity = compute_block_similarity(&a.block_hashes, &b.block_hashes);

    // Weighted combination: 40% global + 60% block-level
    // This weighting emphasizes crop resistance while maintaining overall similarity
    let combined_score = 0.4 * global_similarity + 0.6 * block_similarity;

    if exact_match {
        Similarity {
            score: 1.0,
            exact_match: true,
            perceptual_distance: 0,
        }
    } else {
        Similarity {
            score: combined_score,
            exact_match: false,
            perceptual_distance: global_distance,
        }
    }
}

/// Computes block-level similarity with crop resistance.
///
/// Compares corresponding blocks from the 4x4 grid and filters out blocks
/// with Hamming distance > 32. This threshold handles cropped images where
/// some regions may not overlap between the two images.
///
/// For example, if image A is cropped to show only the top-left quadrant,
/// blocks in the bottom-right of A won't match B, but the top-left blocks
/// will still contribute to the similarity score.
fn compute_block_similarity(a: &[u64; 16], b: &[u64; 16]) -> f32 {
    let mut total_similarity = 0.0f32;
    let mut valid_comparisons = 0u32;

    for i in 0..16 {
        let distance = hamming_distance(a[i], b[i]);
        if distance <= 32 {
            total_similarity += hash_similarity(distance);
            valid_comparisons += 1;
        }
    }

    if valid_comparisons == 0 {
        0.0
    } else {
        total_similarity / valid_comparisons as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_similarity_distance_zero() {
        assert_eq!(hash_similarity(0), 1.0);
    }

    #[test]
    fn test_hash_similarity_distance_32() {
        assert_eq!(hash_similarity(32), 0.5);
    }

    #[test]
    fn test_hash_similarity_distance_64() {
        assert_eq!(hash_similarity(64), 0.0);
    }

    #[test]
    fn test_hash_similarity_distance_greater_than_64() {
        assert_eq!(hash_similarity(100), 0.0);
    }

    #[test]
    fn test_hash_similarity_distance_16() {
        assert_eq!(hash_similarity(16), 0.75);
    }

    #[test]
    fn test_hamming_distance_identical() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(u64::MAX, u64::MAX), 0);
        assert_eq!(hamming_distance(0x1234567890ABCDEF, 0x1234567890ABCDEF), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        assert_eq!(hamming_distance(0, u64::MAX), 64);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        assert_eq!(hamming_distance(0, 1), 1);
        assert_eq!(hamming_distance(0, 2), 1);
        assert_eq!(hamming_distance(0, 4), 1);
    }

    #[test]
    fn test_hamming_distance_known_values() {
        assert_eq!(hamming_distance(0b10101010, 0b01010101), 8);
        assert_eq!(hamming_distance(0xFF00FF00, 0x00FF00FF), 32);
    }

    #[test]
    fn test_hamming_distance_symmetric() {
        let a: u64 = 0x1234567890ABCDEF;
        let b: u64 = 0xFEDCBA0987654321;
        assert_eq!(hamming_distance(a, b), hamming_distance(b, a));
    }

    #[test]
    fn test_compute_block_similarity_identical() {
        let blocks_a = [1u64; 16];
        let blocks_b = [1u64; 16];
        let sim = compute_block_similarity(&blocks_a, &blocks_b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_block_similarity_all_different() {
        let blocks_a = [0u64; 16];
        let blocks_b = [u64::MAX; 16];
        let sim = compute_block_similarity(&blocks_a, &blocks_b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_compute_block_similarity_partial_match() {
        let mut blocks_a = [0u64; 16];
        let mut blocks_b = [0u64; 16];

        for i in 0..8 {
            blocks_a[i] = 0x1234567890ABCDEF;
            blocks_b[i] = 0x1234567890ABCDEF;
        }

        let sim = compute_block_similarity(&blocks_a, &blocks_b);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Expected 1.0 for half matching blocks (others are identical 0s), got {}",
            sim
        );
    }

    #[test]
    fn test_compute_block_similarity_all_above_threshold() {
        let blocks_a = [0u64; 16];
        let mut blocks_b = [0u64; 16];
        for i in 0..16 {
            blocks_b[i] = blocks_a[i] ^ u64::MAX;
        }
        let sim = compute_block_similarity(&blocks_a, &blocks_b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_compute_similarity_identical_fingerprints() {
        let fp = ImageFingerprint::new([1u8; 32], 0x1234567890ABCDEF, [0xABCDEF; 16]);
        let sim = compute_similarity(&fp, &fp);
        assert_eq!(sim.score, 1.0);
        assert!(sim.exact_match);
        assert_eq!(sim.perceptual_distance, 0);
    }

    #[test]
    fn test_compute_similarity_different_fingerprints() {
        let fp1 = ImageFingerprint::new([1u8; 32], 0x0000000000000000, [0u64; 16]);
        let fp2 = ImageFingerprint::new([2u8; 32], 0xFFFFFFFFFFFFFFFF, [u64::MAX; 16]);
        let sim = compute_similarity(&fp1, &fp2);
        assert!(!sim.exact_match);
        assert!(sim.score < 0.1);
        assert_eq!(sim.perceptual_distance, 64);
    }

    #[test]
    fn test_compute_similarity_same_exact_hash_different_blocks() {
        let fp1 = ImageFingerprint::new([1u8; 32], 0x1234567890ABCDEF, [0u64; 16]);
        let fp2 = ImageFingerprint::new([1u8; 32], 0x1234567890ABCDEF, [u64::MAX; 16]);
        let sim = compute_similarity(&fp1, &fp2);
        assert!(sim.exact_match);
        assert_eq!(sim.score, 1.0);
    }

    #[test]
    fn test_compute_similarity_similar_global_different_blocks() {
        let fp1 = ImageFingerprint::new([1u8; 32], 0x0000000000000000, [0u64; 16]);
        let fp2 = ImageFingerprint::new([2u8; 32], 0x0000000000000001, [u64::MAX; 16]);
        let sim = compute_similarity(&fp1, &fp2);
        assert!(!sim.exact_match);
        assert!(sim.perceptual_distance == 1);
    }

    #[test]
    fn test_similarity_perfect() {
        let sim = Similarity::perfect();
        assert_eq!(sim.score, 1.0);
        assert!(sim.exact_match);
        assert_eq!(sim.perceptual_distance, 0);
    }

    #[test]
    fn test_similarity_clone_copy() {
        let sim = Similarity {
            score: 0.75,
            exact_match: false,
            perceptual_distance: 16,
        };
        let sim2 = sim;
        assert_eq!(sim.score, sim2.score);
        assert_eq!(sim.exact_match, sim2.exact_match);
        assert_eq!(sim.perceptual_distance, sim2.perceptual_distance);
    }

    #[test]
    fn test_similarity_partial_eq() {
        let sim1 = Similarity {
            score: 0.75,
            exact_match: false,
            perceptual_distance: 16,
        };
        let sim2 = Similarity {
            score: 0.75,
            exact_match: false,
            perceptual_distance: 16,
        };
        let sim3 = Similarity {
            score: 0.80,
            exact_match: false,
            perceptual_distance: 12,
        };
        assert_eq!(sim1, sim2);
        assert_ne!(sim1, sim3);
    }

    #[test]
    fn test_weighted_combination_formula() {
        let global_hash1 = 0x0000000000000000;
        let global_hash2 = 0x0000000000000000;

        let mut blocks1 = [0u64; 16];
        let mut blocks2 = [0u64; 16];

        for i in 0..16 {
            blocks1[i] = 0xAAAAAAAAAAAAAAAA;
            blocks2[i] = 0xAAAAAAAAAAAAAAAA;
        }

        let fp1 = ImageFingerprint::new([1u8; 32], global_hash1, blocks1);
        let fp2 = ImageFingerprint::new([2u8; 32], global_hash2, blocks2);

        let sim = compute_similarity(&fp1, &fp2);

        assert!(!sim.exact_match);
        assert_eq!(sim.perceptual_distance, 0);
        assert!((sim.score - 1.0).abs() < 1e-5);
    }
}
