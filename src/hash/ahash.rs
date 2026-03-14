//! Average Hash (AHash) implementation.
//!
//! Algorithm: Resize to 8x8, compute mean, compare each pixel to mean.
//! Each pixel produces 1 bit. Total: 8×8 = 64-bit hash.

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

    resample_32x32_to_8x8(pixels, &mut small);

    compute_hash_from_mean(&small)
}

/// Computes AHash from a 64x64 block by downsampling to 8x8 first.
#[inline]
pub fn compute_ahash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; 32 * 32];
    const FACTOR: f32 = 1.0 / 4.0;

    for y in 0..32 {
        let src_y = y * 2;
        for x in 0..32 {
            let src_x = x * 2;
            let idx = y * 32 + x;
            let base = src_y * 64 + src_x;

            downsampled[idx] =
                (block[base] + block[base + 1] + block[base + 64] + block[base + 65]) * FACTOR;
        }
    }

    compute_ahash(&downsampled)
}

/// Resamples 32x32 buffer to 8x8 using bilinear interpolation.
#[inline(always)]
fn resample_32x32_to_8x8(src: &[f32; 32 * 32], dst: &mut [f32; TOTAL_PIXELS]) {
    let x_ratio = 32.0 / 8.0;
    let y_ratio = 32.0 / 8.0;

    for y in 0..AHASH_SIZE {
        let src_y = y as f32 * y_ratio;
        let y0 = src_y as usize;
        let y1 = (y0 + 1).min(31);
        let dy = src_y - y0 as f32;

        for x in 0..AHASH_SIZE {
            let src_x = x as f32 * x_ratio;
            let x0 = src_x as usize;
            let x1 = (x0 + 1).min(31);
            let dx = src_x - x0 as f32;

            let i00 = y0 * 32 + x0;
            let i01 = y0 * 32 + x1;
            let i10 = y1 * 32 + x0;
            let i11 = y1 * 32 + x1;

            let v00 = src[i00];
            let v01 = src[i01];
            let v10 = src[i10];
            let v11 = src[i11];

            let v0 = v00 + (v01 - v00) * dx;
            let v1 = v10 + (v11 - v10) * dx;

            dst[y * AHASH_SIZE + x] = v0 + (v1 - v0) * dy;
        }
    }
}

/// Computes hash by comparing each pixel to the mean brightness.
///
/// Sets bit to 1 if pixel >= mean, 0 otherwise.
/// Bits are ordered from MSB (top-left pixel) to LSB (bottom-right).
#[inline(always)]
fn compute_hash_from_mean(pixels: &[f32; TOTAL_PIXELS]) -> u64 {
    let sum: f32 = pixels.iter().sum();
    let mean = sum / TOTAL_PIXELS as f32;

    let mut hash = 0u64;
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
}
