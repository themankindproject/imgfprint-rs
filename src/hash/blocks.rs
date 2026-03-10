//! Block-level perceptual hashing for crop resistance.

use crate::hash::phash::compute_phash_from_64x64;

/// Computes pHash for all 16 blocks in a 4x4 grid.
///
/// Each block is 64x64 pixels. Block hashes enable matching of cropped images
/// by comparing corresponding regions independently.
pub fn compute_block_hashes(blocks: &[[f32; 64 * 64]; 16]) -> [u64; 16] {
    let mut hashes = [0u64; 16];

    for (i, block) in blocks.iter().enumerate() {
        hashes[i] = compute_phash_from_64x64(block);
    }

    hashes
}
