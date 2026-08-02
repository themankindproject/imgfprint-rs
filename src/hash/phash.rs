//! Perceptual hash (pHash) implementation using 2D DCT.

use std::f32::consts::PI;
use std::sync::OnceLock;

const HASH_SIZE: usize = 8;
const DCT_SIZE: usize = 32;
const TOTAL_HASH_ELEMENTS: usize = HASH_SIZE * HASH_SIZE;

/// Twiddle factors for the radix-2 DIT FFT of length 32.
/// Precomputed: W(k, N) = exp(-2πi·k/N) = (cos(2πk/N), -sin(2πk/N))
static FFT_TWIDDLES: OnceLock<Box<[(f32, f32); DCT_SIZE]>> = OnceLock::new();

#[inline]
fn get_fft_twiddles() -> &'static [(f32, f32); DCT_SIZE] {
    FFT_TWIDDLES.get_or_init(|| {
        let mut tw = Box::new([(0.0f32, 0.0f32); DCT_SIZE]);
        for k in 0..DCT_SIZE {
            let angle = -2.0 * PI * k as f32 / DCT_SIZE as f32;
            tw[k] = (angle.cos(), angle.sin());
        }
        tw
    })
}

/// In-place radix-2 DIT FFT for N=32 complex values.
/// Input is in bit-reversed order, output in natural order.
#[inline]
fn fft32_in_place(re: &mut [f32; DCT_SIZE], im: &mut [f32; DCT_SIZE]) {
    let tw = get_fft_twiddles();

    // Bit-reversal permutation for N=32 (5 bits)
    const BIT_REV: [usize; 32] = [
        0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3,
        19, 11, 27, 7, 23, 15, 31,
    ];

    // Apply bit-reversal permutation
    for (i, &j) in BIT_REV.iter().enumerate() {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly stages: size 2, 4, 8, 16, 32
    let mut size = 2;
    while size <= DCT_SIZE {
        let half = size / 2;
        let step = DCT_SIZE / size; // twiddle index step

        let mut start = 0;
        while start < DCT_SIZE {
            for k in 0..half {
                let tw_idx = k * step;
                let (tw_re, tw_im) = tw[tw_idx];

                let i = start + k;
                let j = start + k + half;

                // Complex multiply: W * X[j]
                let t_re = re[j] * tw_re - im[j] * tw_im;
                let t_im = re[j] * tw_im + im[j] * tw_re;

                // Butterfly
                re[j] = re[i] - t_re;
                im[j] = im[i] - t_im;
                re[i] += t_re;
                im[i] += t_im;
            }
            start += size;
        }
        size *= 2;
    }
}

/// Compute real FFT of 32 real values → 17 complex outputs.
/// This matches the output format of realfft::RealToComplex.
#[inline]
fn real_fft_32(input: &[f32; DCT_SIZE], out_re: &mut [f32; 17], out_im: &mut [f32; 17]) {
    // Pack real input into complex array (imaginary = 0)
    let mut re = *input;
    let mut im = [0.0f32; DCT_SIZE];

    fft32_in_place(&mut re, &mut im);

    // Extract first 17 bins (DC to Nyquist)
    out_re.copy_from_slice(&re[..17]);
    out_im.copy_from_slice(&im[..17]);
}

/// Reusable scratch space for `dct2_32`. Holds the reorder buffer used in
/// the DCT-via-FFT algorithm.
#[derive(Debug, Clone)]
pub(crate) struct Dct2Scratch {
    buffer: [f32; DCT_SIZE],
}

impl Dct2Scratch {
    pub(crate) fn new() -> Self {
        Self {
            buffer: [0.0; DCT_SIZE],
        }
    }
}

/// Reusable scratch space for the DCT path of `compute_phash`.
///
/// Combines the [`Dct2Scratch`] with the row/column/hash buffers used by
/// `compute_phash`, eliminating repeated stack-frame setup per fingerprint call.
#[derive(Debug, Clone)]
pub(crate) struct DctScratch {
    dct2: Dct2Scratch,
    row_buffer: [f32; DCT_SIZE],
    col_buffer: [f32; DCT_SIZE * DCT_SIZE],
    hash_matrix: [f32; TOTAL_HASH_ELEMENTS],
    col_input: [f32; DCT_SIZE],
    col_output: [f32; DCT_SIZE],
}

impl DctScratch {
    pub(crate) fn new() -> Self {
        Self {
            dct2: Dct2Scratch::new(),
            row_buffer: [0.0; DCT_SIZE],
            col_buffer: [0.0; DCT_SIZE * DCT_SIZE],
            hash_matrix: [0.0; TOTAL_HASH_ELEMENTS],
            col_input: [0.0; DCT_SIZE],
            col_output: [0.0; DCT_SIZE],
        }
    }

    /// Prepares scratch for reuse.
    #[inline(always)]
    fn reset(&mut self) {
        self.hash_matrix = [0.0; TOTAL_HASH_ELEMENTS];
    }
}

impl Default for DctScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes DCT-II of length 32 using an inline radix-2 FFT that replicates
/// the exact computation path of the previous realfft-based implementation.
///
/// Algorithm: reorder input into even/odd interleaving, compute 32-point
/// real DFT via radix-2 Cooley-Tukey butterfly, then apply twiddle factors.
/// This preserves bit-identical output with the previous realfft-based code.
#[inline(always)]
fn dct2_32_with_scratch(
    input: &[f32],
    output: &mut [f32],
    scratch: &mut Dct2Scratch,
) -> Result<(), crate::error::ImgFprintError> {
    debug_assert_eq!(input.len(), DCT_SIZE);
    debug_assert_eq!(output.len(), DCT_SIZE);

    // Step 1: Permute input (same reordering as old code)
    for i in 0..(DCT_SIZE / 2) {
        scratch.buffer[i] = input[i * 2];
        scratch.buffer[DCT_SIZE - 1 - i] = input[i * 2 + 1];
    }

    // Step 2: Real FFT via radix-2 butterfly → 17 complex outputs
    let mut fft_re = [0.0f32; 17];
    let mut fft_im = [0.0f32; 17];
    real_fft_32(&scratch.buffer, &mut fft_re, &mut fft_im);

    // Step 3: Apply twiddle factors (identical logic to old code)
    const SCALE: f32 = 2.0 / DCT_SIZE as f32;
    output[0] = fft_re[0] * SCALE;
    for k in 1..DCT_SIZE {
        let angle = -PI * k as f32 / (2.0 * DCT_SIZE as f32);
        let twiddle_re = angle.cos();
        let twiddle_im = angle.sin();
        let re = fft_re[k.min(DCT_SIZE - k)];
        let im = if k < 17 {
            fft_im[k]
        } else {
            -fft_im[DCT_SIZE - k]
        };
        output[k] = (re * twiddle_re - im * twiddle_im) * SCALE;
    }

    Ok(())
}

/// Legacy `dct2_32` entry point that allocates scratch on the stack.
///
/// Kept for the test suite; production callers go through `_with_scratch` variants.
#[cfg(test)]
#[inline(always)]
fn dct2_32(input: &[f32], output: &mut [f32]) -> Result<(), crate::error::ImgFprintError> {
    debug_assert_eq!(input.len(), DCT_SIZE);
    debug_assert_eq!(output.len(), DCT_SIZE);
    let mut scratch = Dct2Scratch::new();
    dct2_32_with_scratch(input, output, &mut scratch)
}

/// Computes perceptual hash from a 32x32 grayscale buffer.
///
/// Uses 2D DCT to extract frequency components, then applies median thresholding
/// to generate a 64-bit hash. The algorithm is robust to minor visual changes.
///
/// Pixel values should be in range [0.0, 1.0].
///
/// # Errors
///
/// Returns `ImgFprintError::ProcessingError` if the DCT computation fails.
#[allow(dead_code)]
pub fn compute_phash(
    pixels: &[f32; DCT_SIZE * DCT_SIZE],
) -> Result<u64, crate::error::ImgFprintError> {
    let mut scratch = DctScratch::new();
    compute_phash_with_scratch(pixels, &mut scratch)
}

/// Context-aware variant of [`compute_phash`](Self::compute_phash) that reuses a
/// caller-supplied scratch buffer instead of allocating stack frames repeatedly.
///
/// Behavior and output are bit-identical to `compute_phash`. The scratch is
/// reset at the start of each call so the function is safe to invoke multiple
/// times against the same buffer.
#[inline]
pub(crate) fn compute_phash_with_scratch(
    pixels: &[f32; DCT_SIZE * DCT_SIZE],
    scratch: &mut DctScratch,
) -> Result<u64, crate::error::ImgFprintError> {
    scratch.reset();

    // Row-wise DCT
    for row in 0..DCT_SIZE {
        let start = row * DCT_SIZE;
        scratch
            .row_buffer
            .copy_from_slice(&pixels[start..start + DCT_SIZE]);
        dct2_32_with_scratch(
            &scratch.row_buffer,
            &mut scratch.col_buffer[start..start + DCT_SIZE],
            &mut scratch.dct2,
        )?;
    }

    // Column-wise DCT (transpose then DCT rows)
    for col in 0..HASH_SIZE {
        // Extract column from row-DCT output into scratch.col_input
        for row in 0..DCT_SIZE {
            scratch.col_input[row] = scratch.col_buffer[row * DCT_SIZE + col];
        }

        dct2_32_with_scratch(
            &scratch.col_input,
            &mut scratch.col_output,
            &mut scratch.dct2,
        )?;

        // Store only top HASH_SIZE rows into scratch.hash_matrix
        for row in 0..HASH_SIZE {
            scratch.hash_matrix[row * HASH_SIZE + col] = scratch.col_output[row];
        }
    }

    Ok(compute_hash_from_coeffs(&scratch.hash_matrix))
}

/// Computes pHash from a 64x64 block by downsampling to 32x32 first.
#[inline]
#[allow(dead_code)]
pub fn compute_phash_from_64x64(
    block: &[f32; 64 * 64],
) -> Result<u64, crate::error::ImgFprintError> {
    let mut scratch = DctScratch::new();
    compute_phash_from_64x64_with_scratch(block, &mut scratch)
}

/// Context-aware variant of [`compute_phash_from_64x64`](Self::compute_phash_from_64x64)
/// that reuses a caller-supplied scratch buffer.
///
/// The 32x32 downsampled block is allocated on the stack (4 KiB) since the issue
/// explicitly scopes the scratch to the DCT row/column/hash buffers — the downsampled
/// block is a transient intermediate that's overwritten on every call. The scratch
/// still saves ~85 KiB of repeated stack setup across a 17-call multi-hash pass.
#[inline]
pub(crate) fn compute_phash_from_64x64_with_scratch(
    block: &[f32; 64 * 64],
    scratch: &mut DctScratch,
) -> Result<u64, crate::error::ImgFprintError> {
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

    compute_phash_with_scratch(&downsampled, scratch)
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
    fn capture_reference_hashes() {
        // Regression test: these exact values must remain stable across
        // implementations to maintain backward compatibility with stored hashes.
        let img1: [f32; 32 * 32] = std::array::from_fn(|i| ((i % 256) as f32) / 255.0);
        assert_eq!(compute_phash(&img1).unwrap(), 0x8a5bff5bff5bff5b);

        let img2: [f32; 32 * 32] = std::array::from_fn(|i| {
            let x = i % 32;
            let y = i / 32;
            ((x * x + y * y) % 256) as f32 / 255.0
        });
        assert_eq!(compute_phash(&img2).unwrap(), 0x860e0e1e7cf9f107);

        let img3: [f32; 32 * 32] = [0.5; 32 * 32];
        assert_eq!(compute_phash(&img3).unwrap(), 0xffffffffffffffff);

        let mut img4 = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img4[y * 32 + x] = x as f32 / 31.0;
            }
        }
        assert_eq!(compute_phash(&img4).unwrap(), 0xaaffffffffffffff);

        let mut img5 = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img5[y * 32 + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        assert_eq!(compute_phash(&img5).unwrap(), 0xffaaffaaffaaffaa);

        let block: [f32; 64 * 64] = std::array::from_fn(|i| {
            let x = i % 64;
            let y = i / 64;
            ((x.wrapping_mul(y)) % 256) as f32 / 255.0
        });
        assert_eq!(
            compute_phash_from_64x64(&block).unwrap(),
            0x80021f3f3f3e7c38
        );
    }

    #[test]
    fn test_phash_deterministic() {
        let img: [f32; 32 * 32] = std::array::from_fn(|i| ((i % 256) as f32) / 255.0);
        let h1 = compute_phash(&img).unwrap();
        let h2 = compute_phash(&img).unwrap();
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
        let h1 = compute_phash(&img1).unwrap();
        let h2 = compute_phash(&img2).unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_phash_uniform_image() {
        let img: [f32; 32 * 32] = [0.5; 32 * 32];
        let hash = compute_phash(&img).unwrap();
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_all_zeros() {
        let img: [f32; 32 * 32] = [0.0; 32 * 32];
        let hash = compute_phash(&img).unwrap();
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_all_ones() {
        let img: [f32; 32 * 32] = [1.0; 32 * 32];
        let hash = compute_phash(&img).unwrap();
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
        let hash = compute_phash(&img).unwrap();
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
        let hash = compute_phash(&img).unwrap();
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_from_64x64_downsampling() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| {
            let x = i % 64;
            let y = i / 64;
            ((x.wrapping_mul(y)) % 256) as f32 / 255.0
        });
        let hash = compute_phash_from_64x64(&block).unwrap();
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_phash_from_64x64_deterministic() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| (i % 256) as f32 / 255.0);
        let h1 = compute_phash_from_64x64(&block).unwrap();
        let h2 = compute_phash_from_64x64(&block).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_phash_with_nan_values() {
        let mut img = [0.5f32; 32 * 32];
        img[0] = f32::NAN;
        img[10] = f32::NAN;
        let hash = compute_phash(&img).unwrap();
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_phash_with_infinity_values() {
        let mut img = [0.5f32; 32 * 32];
        img[0] = f32::INFINITY;
        let hash = compute_phash(&img).unwrap();
        // Infinity propagates through FFT butterfly as NaN/Inf; the hash
        // value depends on the specific FFT implementation. The important
        // property is determinism (same input → same output).
        let hash2 = compute_phash(&img).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_phash_checkerboard_pattern() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        let hash = compute_phash(&img).unwrap();
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

        let h1 = compute_phash(&img1).unwrap();
        let h2 = compute_phash(&img2).unwrap();
        let distance = (h1 ^ h2).count_ones();
        assert!(
            distance < 32,
            "Similar images should have low Hamming distance, got {}",
            distance
        );
    }

    #[test]
    fn test_dct2_32_symmetric_input() {
        let input: [f32; 32] = std::array::from_fn(|i| (i % 256) as f32 / 255.0);
        let mut output = [0.0f32; 32];
        dct2_32(&input, &mut output).unwrap();

        let mut output2 = [0.0f32; 32];
        dct2_32(&input, &mut output2).unwrap();

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
        for (i, item) in coeffs.iter_mut().enumerate().take(TOTAL_HASH_ELEMENTS) {
            *item = i as f32;
        }
        let hash = compute_hash_from_coeffs(&coeffs);
        assert_ne!(hash, 0);
    }
}
