use crate::core::fingerprint::ImageFingerprint;

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
pub fn compute_similarity(a: &ImageFingerprint, b: &ImageFingerprint) -> Similarity {
    if a.exact == b.exact {
        return Similarity::perfect();
    }

    let global_distance = hamming_distance(a.global_hash, b.global_hash);
    let global_similarity = hash_similarity(global_distance);
    let block_similarity = compute_block_similarity(&a.block_hashes, &b.block_hashes);
    let combined_score = 0.4 * global_similarity + 0.6 * block_similarity;

    Similarity {
        score: combined_score,
        exact_match: false,
        perceptual_distance: global_distance,
    }
}

#[inline(always)]
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[inline(always)]
fn hash_similarity(distance: u32) -> f32 {
    if distance >= 64 {
        0.0
    } else {
        1.0 - (distance as f32 / 64.0)
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
