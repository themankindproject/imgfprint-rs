//! Difference Hash (dHash) implementation.
//!
//! Algorithm: Resize to 9x8, compare adjacent pixels horizontally.
//! Each row produces 8 bits (9 pixels compared pairwise).
//! Total: 8 rows × 8 bits = 64-bit hash.

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

    resample_32x32_to_9x8(pixels, &mut small);

    compute_hash_from_gradient(&small)
}

/// Computes dHash from a 64x64 block by downsampling to 9x8 first.
#[inline]
pub fn compute_dhash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
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

    compute_dhash(&downsampled)
}

/// Resamples 32x32 buffer to 9x8 using bilinear interpolation.
#[inline(always)]
fn resample_32x32_to_9x8(src: &[f32; 32 * 32], dst: &mut [f32; DHASH_SIZE]) {
    let x_ratio = 32.0 / 9.0;
    let y_ratio = 32.0 / 8.0;

    for y in 0..DHASH_HEIGHT {
        let src_y = y as f32 * y_ratio;
        let y0 = src_y as usize;
        let y1 = (y0 + 1).min(31);
        let dy = src_y - y0 as f32;

        for x in 0..DHASH_WIDTH {
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

            dst[y * DHASH_WIDTH + x] = v0 + (v1 - v0) * dy;
        }
    }
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
}
