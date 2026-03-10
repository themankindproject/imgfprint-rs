//! Perceptual hash (pHash) implementation using 2D DCT with cached plans.

use rustdct::{DctPlanner, TransformType2And3};
use std::sync::{Arc, OnceLock};

const HASH_SIZE: usize = 8;
const DCT_SIZE: usize = 32;
const TOTAL_HASH_ELEMENTS: usize = HASH_SIZE * HASH_SIZE;

// Global cached DCT plan - computed once, shared across all threads
// This eliminates per-call planner allocation and planning overhead
static DCT_PLAN_32: OnceLock<Arc<dyn TransformType2And3<f32>>> = OnceLock::new();

/// Gets or creates the cached 32-point DCT plan.
fn get_dct_plan() -> Arc<dyn TransformType2And3<f32>> {
    DCT_PLAN_32
        .get_or_init(|| {
            let mut planner = DctPlanner::new();
            planner.plan_dct2(DCT_SIZE)
        })
        .clone()
}

/// Computes perceptual hash from a 32x32 grayscale buffer.
///
/// Uses 2D DCT to extract frequency components, then applies median thresholding
/// to generate a 64-bit hash. The algorithm is robust to minor visual changes.
///
/// Uses a globally cached DCT plan for optimal performance.
///
/// Pixel values should be in range [0.0, 1.0].
pub fn compute_phash(pixels: &[f32; DCT_SIZE * DCT_SIZE]) -> u64 {
    let dct = get_dct_plan();
    let mut row_buffer = [0.0f32; DCT_SIZE];
    let mut col_buffer = [0.0f32; DCT_SIZE * DCT_SIZE];

    // Row DCTs with transposition for cache-efficient column access
    for row in 0..DCT_SIZE {
        let start = row * DCT_SIZE;
        row_buffer.copy_from_slice(&pixels[start..start + DCT_SIZE]);
        dct.process_dct2(&mut row_buffer);

        // Write transposed (column-major) for better cache locality
        for (col, val) in row_buffer.iter().enumerate() {
            col_buffer[col * DCT_SIZE + row] = *val;
        }
    }

    // Column DCTs and hash coefficient extraction
    let mut hash_matrix = [0.0f32; TOTAL_HASH_ELEMENTS];
    for col in 0..HASH_SIZE {
        let start = col * DCT_SIZE;
        row_buffer.copy_from_slice(&col_buffer[start..start + DCT_SIZE]);
        dct.process_dct2(&mut row_buffer);

        for row in 0..HASH_SIZE {
            hash_matrix[row * HASH_SIZE + col] = row_buffer[row];
        }
    }

    // Compute median and generate hash
    compute_hash_from_coeffs(&hash_matrix)
}

/// Computes pHash from a 64x64 block by downsampling to 32x32 first.
///
/// Uses 2x2 averaging for downsampling, then applies standard pHash.
pub fn compute_phash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; DCT_SIZE * DCT_SIZE];

    // Optimized 2x2 downsampling
    for y in 0..32 {
        let src_y = y * 2;
        for x in 0..32 {
            let src_x = x * 2;
            let idx = y * 32 + x;
            let base = src_y * 64 + src_x;

            downsampled[idx] =
                (block[base] + block[base + 1] + block[base + 64] + block[base + 65]) * 0.25;
        }
    }

    compute_phash(&downsampled)
}

/// Computes hash from DCT coefficients using median thresholding.
///
/// Uses deterministic sorting with index-based tie-breaking to ensure
/// consistent hash generation across platforms and compiler optimizations.
#[inline]
fn compute_hash_from_coeffs(coeffs: &[f32; TOTAL_HASH_ELEMENTS]) -> u64 {
    // Copy coefficients with their original indices for tie-breaking
    let mut indexed: [(usize, f32); TOTAL_HASH_ELEMENTS] = std::array::from_fn(|i| (i, coeffs[i]));

    // Stable sort by value, using index as tie-breaker for deterministic ordering
    // This ensures consistent median selection even when coefficients are equal
    indexed.sort_by(|(idx_a, val_a), (idx_b, val_b)| {
        // Handle NaN values - treat as greater than all valid numbers
        match (val_a.is_nan(), val_b.is_nan()) {
            (true, true) => idx_a.cmp(idx_b), // Both NaN: use index ordering
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => {
                // Compare values first, then use index as tie-breaker
                match val_a.partial_cmp(val_b) {
                    Some(std::cmp::Ordering::Equal) => idx_a.cmp(idx_b),
                    Some(ordering) => ordering,
                    None => idx_a.cmp(idx_b), // Shouldn't happen for non-NaN
                }
            }
        }
    });

    // Extract median value
    let median = indexed[TOTAL_HASH_ELEMENTS / 2].1;

    // Build 64-bit hash using median thresholding
    // Bit 63 (MSB) corresponds to coefficient 0 (DC component)
    // Bit 0 (LSB) corresponds to coefficient 63
    coeffs
        .iter()
        .enumerate()
        .take(TOTAL_HASH_ELEMENTS)
        .fold(0u64, |hash, (i, &coeff)| {
            if coeff >= median {
                hash | (1u64 << (63 - i))
            } else {
                hash
            }
        })
}
