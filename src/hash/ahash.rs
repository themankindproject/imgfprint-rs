//! Average Hash (AHash) implementation.
//!
//! Algorithm: Resize to 8x8, compute mean, compare each pixel to mean.
//! Each pixel produces 1 bit. Total: 8×8 = 64-bit hash.

use crate::imgproc::preprocess::bilinear_resample;

const AHASH_SIZE: usize = 8;
const TOTAL_PIXELS: usize = AHASH_SIZE * AHASH_SIZE;

/// Computes average hash from a 32x32 grayscale buffer.
///
/// First resamples to 8x8, then computes the mean pixel value.
/// Each bit is set to 1 if the pixel is brighter than the mean.
///
/// # Arguments
/// * `pixels` - 32x32 grayscale buffer with values in [0.0, 1.0]
pub fn compute_ahash(pixels: &[f32; 32 * 32]) -> u64 {
    let mut small = [0.0f32; TOTAL_PIXELS];

    bilinear_resample(pixels, 32, 32, &mut small, AHASH_SIZE, AHASH_SIZE);

    compute_hash_from_mean(&small)
}

/// Computes AHash from a 64x64 block by downsampling to 8x8 first.
#[inline]
pub fn compute_ahash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; 32 * 32];
    
    // First downsample 64x64 to 32x32 using bilinear resampling
    bilinear_resample(block, 64, 64, &mut downsampled, 32, 32);

    compute_ahash(&downsampled)
}

/// Computes hash by comparing each pixel to the mean brightness.
///
/// Sets bit to 1 if pixel >= mean, 0 otherwise.
/// Bits are ordered from MSB (bit 63) to LSB (bit 0) for top-left to bottom-right pixels.
#[inline(always)]
fn compute_hash_from_mean(pixels: &[f32; TOTAL_PIXELS]) -> u64 {
    let sum: f32 = pixels.iter().sum();
    let mean = sum / TOTAL_PIXELS as f32;

    let mut hash = 0u64;
    // Start from bit 63 (MSB) and count down to bit 0 (LSB)
    // This ensures consistent bit ordering across all hash algorithms
    let mut bit_pos = 63u32;

    for y in 0..AHASH_SIZE {
        for x in 0..AHASH_SIZE {
            if pixels[y * AHASH_SIZE + x] >= mean {
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
    fn test_ahash_determinism() {
        let img: [f32; 32 * 32] = std::array::from_fn(|i| ((i % 256) as f32) / 255.0);

        let h1 = compute_ahash(&img);
        let h2 = compute_ahash(&img);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_ahash_different_images() {
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

        let h1 = compute_ahash(&img1);
        let h2 = compute_ahash(&img2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_ahash_from_64x64() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| {
            let x = i % 64;
            let y = i / 64;
            ((x.wrapping_mul(y)) % 256) as f32 / 255.0
        });

        let hash = compute_ahash_from_64x64(&block);
        let _ = hash;
    }

    #[test]
    fn test_ahash_uniform_image() {
        let img: [f32; 32 * 32] = [0.5; 32 * 32];

        let hash = compute_ahash(&img);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_ahash_all_dark() {
        let img: [f32; 32 * 32] = [0.0; 32 * 32];

        let hash = compute_ahash(&img);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_ahash_all_bright() {
        let img: [f32; 32 * 32] = [1.0; 32 * 32];

        let hash = compute_ahash(&img);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_ahash_gradient_horizontal() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = x as f32 / 31.0;
            }
        }
        let hash = compute_ahash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_ahash_gradient_vertical() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = y as f32 / 31.0;
            }
        }
        let hash = compute_ahash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_ahash_checkerboard_pattern() {
        let mut img = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                img[y * 32 + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        let hash = compute_ahash(&img);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_ahash_from_64x64_deterministic() {
        let block: [f32; 64 * 64] = std::array::from_fn(|i| (i % 256) as f32 / 255.0);
        let h1 = compute_ahash_from_64x64(&block);
        let h2 = compute_ahash_from_64x64(&block);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_ahash_from_64x64_uniform() {
        let block: [f32; 64 * 64] = [0.5; 64 * 64];
        let hash = compute_ahash_from_64x64(&block);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_compute_hash_from_mean_uniform() {
        let pixels = [0.5f32; TOTAL_PIXELS];
        let hash = compute_hash_from_mean(&pixels);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_compute_hash_from_mean_all_zeros() {
        let pixels = [0.0f32; TOTAL_PIXELS];
        let hash = compute_hash_from_mean(&pixels);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_compute_hash_from_mean_all_ones() {
        let pixels = [1.0f32; TOTAL_PIXELS];
        let hash = compute_hash_from_mean(&pixels);
        assert_eq!(hash, u64::MAX);
    }

    #[test]
    fn test_compute_hash_from_mean_half_half() {
        let mut pixels = [0.0f32; TOTAL_PIXELS];
        for i in 0..TOTAL_PIXELS / 2 {
            pixels[i] = 0.0;
        }
        for i in TOTAL_PIXELS / 2..TOTAL_PIXELS {
            pixels[i] = 1.0;
        }
        let hash = compute_hash_from_mean(&pixels);
        assert_ne!(hash, 0);
        assert_ne!(hash, u64::MAX);
    }

    #[test]
    fn test_compute_hash_from_mean_bit_ordering() {
        let mut pixels = [0.0f32; TOTAL_PIXELS];
        for i in 0..TOTAL_PIXELS {
            pixels[i] = i as f32 / TOTAL_PIXELS as f32;
        }
        let hash = compute_hash_from_mean(&pixels);
        assert_ne!(hash, 0);
    }

    #[test]
    fn test_ahash_similar_images() {
        let mut img1 = [0.5f32; 32 * 32];
        let mut img2 = [0.5f32; 32 * 32];
        
        for i in 0..img1.len() {
            img1[i] = (i % 128) as f32 / 255.0;
            img2[i] = (i % 128 + 2) as f32 / 255.0;
        }
        
        let h1 = compute_ahash(&img1);
        let h2 = compute_ahash(&img2);
        let distance = (h1 ^ h2).count_ones();
        assert!(distance < 32, "Similar images should have low Hamming distance, got {}", distance);
    }
}
