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
#[inline]
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
///
/// Uses stack-allocated buffers to avoid heap allocation and RefCell overhead.
#[inline(always)]
fn dct2_32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), 32);
    debug_assert_eq!(output.len(), 32);

    let fft = get_fft_plan();

    // Stack-allocated buffers - faster than thread-local with RefCell
    let mut buffer = [0.0f32; 32];
    let mut complex_buffer = [Complex32::new(0.0, 0.0); 17];

    // Input permutation for DCT-II via FFT
    // Unroll for better performance
    for i in 0..16 {
        buffer[i] = input[i * 2];
        buffer[31 - i] = input[i * 2 + 1];
    }

    // Forward FFT (requires mutable input buffer)
    fft.process(&mut buffer, &mut complex_buffer).unwrap();

    // Extract real part and scale
    const SCALE: f32 = 2.0 / 32.0;
    output[0] = complex_buffer[0].re * SCALE;
    for (i, item) in output.iter_mut().enumerate().take(32).skip(1) {
        let k = i.min(32 - i);
        *item = complex_buffer[k].re * SCALE;
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
/// Uses linear-time selection algorithm (nth_element) for optimal performance.
/// Falls back to sorting for NaN handling to ensure deterministic results.
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
    // Fast path: check if any NaN values exist
    let has_nan = coeffs.iter().any(|v| v.is_nan());

    let median = if has_nan {
        // Fallback: sort to handle NaN values deterministically
        let mut indexed: [(usize, f32); TOTAL_HASH_ELEMENTS] =
            std::array::from_fn(|i| (i, coeffs[i]));

        indexed.sort_unstable_by(|(idx_a, val_a), (idx_b, val_b)| {
            match (val_a.is_nan(), val_b.is_nan()) {
                (true, true) => idx_a.cmp(idx_b),
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => val_a.total_cmp(val_b),
            }
        });

        indexed[TOTAL_HASH_ELEMENTS / 2].1
    } else {
        // Fast path: use selection algorithm (O(n) instead of O(n log n))
        let mut coeffs_copy = *coeffs;
        let median_idx = TOTAL_HASH_ELEMENTS / 2;

        // Use total_cmp for faster f32 comparison (no partial_cmp overhead)
        *coeffs_copy
            .select_nth_unstable_by(median_idx, |a, b| a.total_cmp(b))
            .1
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phash_deterministic() {
        let img: [f32; 32 * 32] = std::array::from_fn(|i| ((i % 256) as f32) / 255.0);
        let h1 = compute_phash(&img);
        let h2 = compute_phash(&img);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_phash_different_images() {
        let img1: [f32; 32 * 32] = std::array::from_fn(|i| {
            let x = i % 32;
            let y = i / 32;
            ((x * x + y * y) % 256) as f32 / 255.0
        });
        let img2: [f32; 32 * 32] = std::array::from_fn(|i| {
            let x = i % 32;
            let y = i / 32;
            ((x + y) * 7 % 256) as f32 / 255.0
        });
        let h1 = compute_phash(&img1);
        let h2 = compute_phash(&img2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_phash_uniform_image() {
        let img: [f32; 32 * 32] = [0.5; 32 * 32];
        let hash = compute_phash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_all_zeros() {
        let img: [f32; 32 * 32] = [0.0; 32 * 32];
        let hash = compute_phash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_all_ones() {
        let img: [f32; 32 * 32] = [1.0; 32 * 32];
        let hash = compute_phash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_gradient_horizontal() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = x as f32 / 31.0;
            }
        }
        let hash = compute_phash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_gradient_vertical() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = y as f32 / 31.0;
            }
        }
        let hash = compute_phash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_from_64x64_downsampling() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| {
            let x = i % 64;
            let y = i / 64;
            ((x.wrapping_mul(y)) % 256) as f32 / 255.0
        });
        let hash = compute_phash_from_64x64(&block);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_from_64x64_deterministic() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| (i % 256) as f32 / 255.0);
        let h1 = compute_phash_from_64x64(&block);
        let h2 = compute_phash_from_64x64(&block);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_phash_with_nan_values() {
        let mut img = [0.5f32; 32 * 32];
        img[0] = f32::NAN;
        img[10] = f32::NAN;
        let hash = compute_phash(&img);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_phash_with_infinity_values() {
        let mut img = [0.5f32; 32 * 32];
        img[0] = f32::INFINITY;
        let hash = compute_phash(&img);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_phash_checkerboard_pattern() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        let hash = compute_phash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_similar_images_similar_hashes() {
        let mut img1 = [0.5f32; 32 * 32];
        let mut img2 = [0.5f32; 32 * 32];
        
        for i in 0..img1.len() {
            img1[i] = (i % 128) as f32 / 255.0;
            img2[i] = (i % 128 + 2) as f32 / 255.0;
        }
        
        let h1 = compute_phash(&img1);
        let h2 = compute_phash(&img2);
        let distance = (h1 ^ h2).count_ones();
        assert!(distance < 32, "Similar images should have low Hamming distance, got {}", distance);
    }

    #[test]
    fn test_dct2_32_symmetric_input() {
        let input: [f32; 32] = std::array::from_fn(|i| (i % 256) as f32 / 255.0);
        let mut output = [0.0f32; 32];
        dct2_32(&input, &mut output);
        
        let mut output2 = [0.0f32; 32];
        dct2_32(&input, &mut output2);
        
        assert_eq!(output, output2);
    }

    #[test]
    fn test_compute_hash_from_coeffs_all_same() {
        let coeffs = [0.5f32; TOTAL_HASH_ELEMENTS];
        let hash = compute_hash_from_coeffs(&coeffs);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_compute_hash_from_coeffs_ascending() {
        let mut coeffs = [0.0f32; TOTAL_HASH_ELEMENTS];
        for i in 0..TOTAL_HASH_ELEMENTS {
            coeffs[i] = i as f32;
        }
        let hash = compute_hash_from_coeffs(&coeffs);
        assert_ne!(hash, 0);
    }
}
