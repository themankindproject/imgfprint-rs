//! Perceptual hash (pHash) implementation using 2D DCT with FFT acceleration.

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex32;
use std::sync::{Arc, OnceLock};

const HASH_SIZE: usize = 8;
const DCT_SIZE: usize = 32;
const TOTAL_HASH_ELEMENTS: usize = HASH_SIZE * HASH_SIZE;

// Cached FFT planner for 32-point real FFT
static FFT_PLAN_32: OnceLock<Arc<dyn realfft::RealToComplex<f32>>> = OnceLock::new();

/// Gets or creates the cached 32-point real FFT plan.
fn get_fft_plan() -> Arc<dyn realfft::RealToComplex<f32>> {
    FFT_PLAN_32
        .get_or_init(|| {
            let mut planner = RealFftPlanner::<f32>::new();
            planner.plan_fft_forward(DCT_SIZE)
        })
        .clone()
}

/// Computes DCT-II of length 32 using real FFT.
///
/// Uses the algorithm: DCT-II(x) = 2 * Re{FFT(y)} where y is a permuted version of x.
/// This is more efficient than direct DCT and leverages SIMD-accelerated FFT.
#[inline(always)]
fn dct2_32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), 32);
    debug_assert_eq!(output.len(), 32);

    let fft = get_fft_plan();
    let mut buffer = [0.0f32; 32];
    let mut complex_buffer = [Complex32::new(0.0, 0.0); 17]; // N/2 + 1 for real FFT

    // Input permutation for DCT-II via FFT
    for i in 0..16 {
        buffer[i] = input[i * 2];
        buffer[31 - i] = input[i * 2 + 1];
    }

    // Forward FFT
    fft.process(&mut buffer, &mut complex_buffer).unwrap();

    // Extract real part and scale
    const SCALE: f32 = 2.0 / 32.0;
    output[0] = complex_buffer[0].re * SCALE;
    #[allow(clippy::needless_range_loop)]
    for i in 1..32 {
        let k = i.min(32 - i);
        output[i] = complex_buffer[k].re * SCALE;
    }
}

/// Computes perceptual hash from a 32x32 grayscale buffer.
///
/// Uses 2D DCT to extract frequency components, then applies median thresholding
/// to generate a 64-bit hash. The algorithm is robust to minor visual changes.
///
/// Pixel values should be in range [0.0, 1.0].
pub fn compute_phash(pixels: &[f32; DCT_SIZE * DCT_SIZE]) -> u64 {
    let mut row_buffer = [0.0f32; DCT_SIZE];
    let mut col_buffer = [0.0f32; DCT_SIZE * DCT_SIZE];

    // Row-wise DCT
    for row in 0..DCT_SIZE {
        let start = row * DCT_SIZE;
        row_buffer.copy_from_slice(&pixels[start..start + DCT_SIZE]);
        dct2_32(&row_buffer, &mut col_buffer[start..start + DCT_SIZE]);
    }

    // Column-wise DCT (transpose then DCT rows)
    let mut hash_matrix = [0.0f32; TOTAL_HASH_ELEMENTS];
    let mut col_input = [0.0f32; DCT_SIZE];
    let mut col_output = [0.0f32; DCT_SIZE];

    for col in 0..HASH_SIZE {
        // Extract column
        for row in 0..DCT_SIZE {
            col_input[row] = col_buffer[row * DCT_SIZE + col];
        }

        dct2_32(&col_input, &mut col_output);

        // Store only top HASH_SIZE rows
        for row in 0..HASH_SIZE {
            hash_matrix[row * HASH_SIZE + col] = col_output[row];
        }
    }

    compute_hash_from_coeffs(&hash_matrix)
}

/// Computes pHash from a 64x64 block by downsampling to 32x32 first.
#[inline]
pub fn compute_phash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; DCT_SIZE * DCT_SIZE];
    const DOWNSAMPLE_FACTOR: f32 = 1.0 / 4.0;

    for y in 0..32 {
        let src_y = y * 2;
        for x in 0..32 {
            let src_x = x * 2;
            let idx = y * 32 + x;
            let base = src_y * 64 + src_x;

            downsampled[idx] =
                (block[base] + block[base + 1] + block[base + 64] + block[base + 65])
                    * DOWNSAMPLE_FACTOR;
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
#[inline(always)]
fn compute_hash_from_coeffs(coeffs: &[f32; TOTAL_HASH_ELEMENTS]) -> u64 {
    let mut indexed: [(usize, f32); TOTAL_HASH_ELEMENTS] = std::array::from_fn(|i| (i, coeffs[i]));

    indexed.sort_unstable_by(|(idx_a, val_a), (idx_b, val_b)| {
        match (val_a.is_nan(), val_b.is_nan()) {
            (true, true) => idx_a.cmp(idx_b),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => match val_a.partial_cmp(val_b) {
                Some(std::cmp::Ordering::Equal) => idx_a.cmp(idx_b),
                Some(ordering) => ordering,
                None => idx_a.cmp(idx_b),
            },
        }
    });

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
