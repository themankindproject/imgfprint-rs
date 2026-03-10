//! Image preprocessing and normalization.

use image::{imageops::FilterType, DynamicImage, GrayImage};

const NORMALIZED_SIZE: u32 = 256;
const BLOCK_SIZE: u32 = 64;
const PHASH_SIZE: u32 = 32;

/// Normalizes image to 256x256 grayscale.
///
/// Applies EXIF orientation (handled automatically by image crate),
/// resizes using Lanczos3 filtering, and converts to 8-bit grayscale.
pub fn normalize(image: DynamicImage) -> GrayImage {
    let oriented = fix_orientation(image);
    oriented
        .resize_exact(NORMALIZED_SIZE, NORMALIZED_SIZE, FilterType::Lanczos3)
        .to_luma8()
}

/// Extracts center 32x32 region as normalized float buffer.
///
/// Returns pixel values in range [0.0, 1.0].
pub fn extract_global_region(image: &GrayImage) -> [f32; (PHASH_SIZE * PHASH_SIZE) as usize] {
    let start_x = (NORMALIZED_SIZE - PHASH_SIZE) / 2;
    let start_y = (NORMALIZED_SIZE - PHASH_SIZE) / 2;
    let mut buffer = [0.0f32; (PHASH_SIZE * PHASH_SIZE) as usize];

    for y in 0..PHASH_SIZE {
        for x in 0..PHASH_SIZE {
            let px = image.get_pixel(start_x + x, start_y + y);
            buffer[(y * PHASH_SIZE + x) as usize] = px[0] as f32 / 255.0;
        }
    }

    buffer
}

/// Extracts 4x4 grid of 64x64 blocks as float buffers.
///
/// Each block contains pixel values in range [0.0, 1.0].
pub fn extract_blocks(image: &GrayImage) -> [[f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16] {
    let mut blocks = [[0.0f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16];

    for block_y in 0..4 {
        for block_x in 0..4 {
            let block_idx = (block_y * 4 + block_x) as usize;
            let start_x = block_x * BLOCK_SIZE;
            let start_y = block_y * BLOCK_SIZE;

            for y in 0..BLOCK_SIZE {
                for x in 0..BLOCK_SIZE {
                    let px = image.get_pixel(start_x + x, start_y + y);
                    blocks[block_idx][(y * BLOCK_SIZE + x) as usize] = px[0] as f32 / 255.0;
                }
            }
        }
    }

    blocks
}

fn fix_orientation(image: DynamicImage) -> DynamicImage {
    // image crate (0.25+) handles EXIF orientation automatically
    image
}
