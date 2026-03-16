//! Difference Hash (dHash) implementation.
//!
//! Algorithm: Resize to 9x8, compare adjacent pixels horizontally.
//! Each row produces 8 bits (9 pixels compared pairwise).
//! Total: 8 rows × 8 bits = 64-bit hash.

use crate::imgproc::preprocess::bilinear_resample;

const DHASH_WIDTH: usize = 9;
const DHASH_HEIGHT: usize = 8;
const DHASH_SIZE: usize = DHASH_WIDTH * DHASH_HEIGHT;

/// Computes difference hash from a 32x32 grayscale buffer.
///
/// First resamples to 9x8, then compares each pixel to its right neighbor.
/// Returns 64-bit hash where each bit represents a gradient direction.
///
/// # Arguments
/// * `pixels` - 32x32 grayscale buffer with values in [0.0, 1.0]
pub fn compute_dhash(pixels: &[f32; 32 * 32]) -> u64 {
    let mut small = [0.0f32; DHASH_SIZE];

    bilinear_resample(pixels, 32, 32, &mut small, DHASH_WIDTH, DHASH_HEIGHT);

    compute_hash_from_gradient(&small)
}

/// Computes dHash from a 64x64 block by downsampling to 9x8 first.
#[inline]
pub fn compute_dhash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; 32 * 32];
    
    // First downsample 64x64 to 32x32 using bilinear resampling
    bilinear_resample(block, 64, 64, &mut downsampled, 32, 32);

    compute_dhash(&downsampled)
}

/// Computes hash by comparing adjacent pixels horizontally.
///
/// For each row, compares `pixel[i]` with `pixel[i+1]`.
/// Sets bit to 1 if `pixel[i]` > `pixel[i+1]` (dark to light transition).
/// Bits are ordered from MSB (bit 63) to LSB (bit 0) for consistent ordering.
#[inline(always)]
fn compute_hash_from_gradient(pixels: &[f32; DHASH_SIZE]) -> u64 {
    let mut hash = 0u64;
    // Start from bit 63 (MSB) and count down to bit 0 (LSB)
    // This ensures consistent bit ordering across all hash algorithms
    let mut bit_pos = 63u32;

    for y in 0..DHASH_HEIGHT {
        let row_start = y * DHASH_WIDTH;
        for x in 0..(DHASH_WIDTH - 1) {
            let left = pixels[row_start + x];
            let right = pixels[row_start + x + 1];

            if left > right {
                hash |= 1u64 << bit_pos;
            }
            bit_pos = bit_pos.saturating_sub(1);
        }
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dhash_determinism() {
        let img: [f32; 32 * 32] = std::array::from_fn(|i| ((i % 256) as f32) / 255.0);

        let h1 = compute_dhash(&img);
        let h2 = compute_dhash(&img);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_dhash_different_images() {
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

        let h1 = compute_dhash(&img1);
        let h2 = compute_dhash(&img2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_dhash_from_64x64() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| {
            let x = i % 64;
            let y = i / 64;
            ((x.wrapping_mul(y)) % 256) as f32 / 255.0
        });

        let hash = compute_dhash_from_64x64(&block);
        let _ = hash;
    }

    #[test]
    fn test_dhash_uniform_image() {
        let img: [f32; 32 * 32] = [0.5; 32 * 32];
        let hash = compute_dhash(&img);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_dhash_all_zeros() {
        let img: [f32; 32 * 32] = [0.0; 32 * 32];
        let hash = compute_dhash(&img);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_dhash_all_ones() {
        let img: [f32; 32 * 32] = [1.0; 32 * 32];
        let hash = compute_dhash(&img);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_dhash_gradient_horizontal() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = x as f32 / 31.0;
            }
        }
        let hash = compute_dhash(&img);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_dhash_gradient_vertical() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = y as f32 / 31.0;
            }
        }
        let hash = compute_dhash(&img);
        let _ = hash;
    }

    #[test]
    fn test_dhash_checkerboard_pattern() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        let hash = compute_dhash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_dhash_bit_ordering() {
        let mut img1 = [0.0f32; 32 * 32];
        let mut img2 = [0.0f32; 32 * 32];
        
        for i in 0..img1.len() {
            img1[i] = (i % 128) as f32 / 255.0;
            img2[i] = (i % 128 + 1) as f32 / 255.0;
        }
        
        let h1 = compute_dhash(&img1);
        let h2 = compute_dhash(&img2);
        
        let distance = (h1 ^ h2).count_ones();
        assert!(distance < 32, "Similar images should have low Hamming distance, got {}", distance);
    }

    #[test]
    fn test_dhash_from_64x64_deterministic() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| (i % 256) as f32 / 255.0);
        let h1 = compute_dhash_from_64x64(&block);
        let h2 = compute_dhash_from_64x64(&block);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_dhash_from_64x64_uniform() {
        let block: [f32; 64 * 64] = [0.5; 64 * 64];
        let hash = compute_dhash_from_64x64(&block);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_compute_hash_from_gradient_uniform() {
        let pixels = [0.5f32; DHASH_SIZE];
        let hash = compute_hash_from_gradient(&pixels);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_compute_hash_from_gradient_ascending() {
        let mut pixels = [0.0f32; DHASH_SIZE];
        for i in 0..DHASH_SIZE {
            pixels[i] = i as f32 / DHASH_SIZE as f32;
        }
        let hash = compute_hash_from_gradient(&pixels);
        assert_eq!(hash, 0);
    }

    #[test]
    fn test_compute_hash_from_gradient_descending() {
        let mut pixels = [1.0f32; DHASH_SIZE];
        for i in 0..DHASH_SIZE {
            pixels[i] = 1.0 - (i as f32 / DHASH_SIZE as f32);
        }
        let hash = compute_hash_from_gradient(&pixels);
        assert_ne!(hash, 0);
    }
}
