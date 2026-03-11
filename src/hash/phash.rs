//! Perceptual hash (pHash) implementation using 2D DCT with cached plans.

use rustdct::{DctPlanner, TransformType2And3};
use std::sync::{Arc, OnceLock};

const HASH_SIZE: usize = 8;
const DCT_SIZE: usize = 32;
const TOTAL_HASH_ELEMENTS: usize = HASH_SIZE * HASH_SIZE;

// Global cached DCT plan - computed once, shared across all threads
// This eliminates per-call planner allocation and planning overhead.
// Note: Arc::clone() is cheap (atomic increment) and only happens 17 times per image
// (1 for global hash + 16 for block hashes), which is negligible compared to DCT computation.
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
/// Pixel values should be in range [0.0, 1.0].
pub fn compute_phash(pixels: &[f32; DCT_SIZE * DCT_SIZE]) -> u64 {
    let dct = get_dct_plan();
    let mut row_buffer = [0.0f32; DCT_SIZE];
    let mut col_buffer = [0.0f32; DCT_SIZE * DCT_SIZE];

    for row in 0..DCT_SIZE {
        let start = row * DCT_SIZE;
        row_buffer.copy_from_slice(&pixels[start..start + DCT_SIZE]);
        dct.process_dct2(&mut row_buffer);

        for (col, val) in row_buffer.iter().enumerate() {
            col_buffer[col * DCT_SIZE + row] = *val;
        }
    }

    let mut hash_matrix = [0.0f32; TOTAL_HASH_ELEMENTS];
    for col in 0..HASH_SIZE {
        let start = col * DCT_SIZE;
        row_buffer.copy_from_slice(&col_buffer[start..start + DCT_SIZE]);
        dct.process_dct2(&mut row_buffer);

        for row in 0..HASH_SIZE {
            hash_matrix[row * HASH_SIZE + col] = row_buffer[row];
        }
    }

    compute_hash_from_coeffs(&hash_matrix)
}

/// Computes pHash from a 64x64 block by downsampling to 32x32 first.
pub fn compute_phash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; DCT_SIZE * DCT_SIZE];

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
///
/// # Bit Ordering
/// The resulting 64-bit hash uses the following bit layout:
/// - Bit 63 (MSB): DC component (average brightness)
/// - Bits 62-0: AC components (frequency information)
///
/// This ordering places the most significant frequency component at the
/// highest bit position for intuitive comparison.
#[inline]
fn compute_hash_from_coeffs(coeffs: &[f32; TOTAL_HASH_ELEMENTS]) -> u64 {
    let mut indexed: [(usize, f32); TOTAL_HASH_ELEMENTS] = std::array::from_fn(|i| (i, coeffs[i]));

    indexed.sort_by(
        |(idx_a, val_a), (idx_b, val_b)| match (val_a.is_nan(), val_b.is_nan()) {
            (true, true) => idx_a.cmp(idx_b),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => match val_a.partial_cmp(val_b) {
                Some(std::cmp::Ordering::Equal) => idx_a.cmp(idx_b),
                Some(ordering) => ordering,
                None => idx_a.cmp(idx_b),
            },
        },
    );

    let median = indexed[TOTAL_HASH_ELEMENTS / 2].1;

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
