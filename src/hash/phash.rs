//! Perceptual hash (pHash) implementation using 2D DCT.

use rustdct::DctPlanner;
use std::cell::RefCell;

const HASH_SIZE: usize = 8;
const DCT_SIZE: usize = 32;
const TOTAL_HASH_ELEMENTS: usize = HASH_SIZE * HASH_SIZE;

thread_local! {
    static DCT_PLANNER: RefCell<DctPlanner<f32>> = RefCell::new(DctPlanner::new());
}

/// Computes perceptual hash from a 32x32 grayscale buffer.
///
/// Uses 2D DCT to extract frequency components, then applies median thresholding
/// to generate a 64-bit hash. The algorithm is robust to minor visual changes.
///
/// Pixel values should be in range [0.0, 1.0].
pub fn compute_phash(pixels: &[f32; DCT_SIZE * DCT_SIZE]) -> u64 {
    let mut row_dct = [0.0f32; DCT_SIZE];
    let mut col_buffer = [0.0f32; DCT_SIZE * DCT_SIZE];

    DCT_PLANNER.with(|planner| {
        let dct = planner.borrow_mut().plan_dct2(DCT_SIZE);

        for row in 0..DCT_SIZE {
            let start = row * DCT_SIZE;
            row_dct.copy_from_slice(&pixels[start..start + DCT_SIZE]);
            dct.process_dct2(&mut row_dct);

            for col in 0..DCT_SIZE {
                col_buffer[col * DCT_SIZE + row] = row_dct[col];
            }
        }

        let mut hash_matrix = [0.0f32; TOTAL_HASH_ELEMENTS];
        for col in 0..HASH_SIZE {
            let start = col * DCT_SIZE;
            row_dct.copy_from_slice(&col_buffer[start..start + DCT_SIZE]);
            dct.process_dct2(&mut row_dct);

            for row in 0..HASH_SIZE {
                hash_matrix[row * HASH_SIZE + col] = row_dct[row];
            }
        }

        let mut coeffs = [0.0f32; TOTAL_HASH_ELEMENTS];
        coeffs.copy_from_slice(&hash_matrix);

        coeffs.sort_unstable_by(|a, b| match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        });

        let median = coeffs[TOTAL_HASH_ELEMENTS / 2];

        let mut hash: u64 = 0;
        for i in 0..TOTAL_HASH_ELEMENTS {
            if hash_matrix[i] >= median {
                hash |= 1u64 << (63 - i);
            }
        }

        hash
    })
}

/// Computes pHash from a 64x64 block by downsampling to 32x32 first.
///
/// Uses simple 2x2 averaging for downsampling, then applies standard pHash.
pub fn compute_phash_from_64x64(block: &[f32; 64 * 64]) -> u64 {
    let mut downsampled = [0.0f32; DCT_SIZE * DCT_SIZE];

    for y in 0..32 {
        for x in 0..32 {
            let src_y = y * 2;
            let src_x = x * 2;
            let idx = y * 32 + x;

            downsampled[idx] = (block[src_y * 64 + src_x]
                + block[src_y * 64 + src_x + 1]
                + block[(src_y + 1) * 64 + src_x]
                + block[(src_y + 1) * 64 + src_x + 1])
                * 0.25;
        }
    }

    compute_phash(&downsampled)
}
