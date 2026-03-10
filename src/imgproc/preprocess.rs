//! Image preprocessing and normalization with SIMD-accelerated resize.

use fast_image_resize::images::Image;
use fast_image_resize::{CpuExtensions, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, GrayImage};

const NORMALIZED_SIZE: u32 = 256;
const BLOCK_SIZE: u32 = 64;
const PHASH_SIZE: u32 = 32;

/// Preprocessor with cached resizer and CPU extension detection.
pub struct Preprocessor {
    resizer: Resizer,
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Preprocessor {
    /// Creates a new preprocessor with optimal CPU extensions.
    pub fn new() -> Self {
        let mut resizer = Resizer::new();

        // Auto-detect and enable best CPU extensions
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    resizer.set_cpu_extensions(CpuExtensions::Avx2);
                }
            } else if std::is_x86_feature_detected!("sse4.1") {
                unsafe {
                    resizer.set_cpu_extensions(CpuExtensions::Sse4_1);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                resizer.set_cpu_extensions(CpuExtensions::Neon);
            }
        }

        Self { resizer }
    }

    /// Normalizes image to 256x256 grayscale using SIMD-accelerated resize.
    ///
    /// Uses Lanczos3 filtering for high-quality downsampling, then converts
    /// to grayscale. The SIMD-accelerated resize provides 3-4x speedup
    /// compared to the image crate's implementation.
    pub fn normalize(&mut self, image: &DynamicImage) -> GrayImage {
        let oriented = fix_orientation(image.clone());

        // Get dimensions
        let (src_w, src_h) = oriented.dimensions();

        // Convert to RGB8 for fast_image_resize input
        let rgb_img = oriented.to_rgb8();
        let src_data = rgb_img.into_raw();

        // Create source image view in fast_image_resize format (RGB8)
        let src = Image::from_vec_u8(src_w, src_h, src_data, PixelType::U8x3)
            .expect("valid RGB source image");

        // Create destination buffer as RGB8 (same pixel type as source)
        let mut dst = Image::new(NORMALIZED_SIZE, NORMALIZED_SIZE, PixelType::U8x3);

        // Configure resize with Lanczos3
        let options = ResizeOptions {
            algorithm: ResizeAlg::Convolution(FilterType::Lanczos3),
            ..Default::default()
        };

        // Perform resize (RGB8 -> RGB8)
        self.resizer
            .resize(&src, &mut dst, &options)
            .expect("resize operation failed");

        // Convert to grayscale
        let rgb_bytes = dst.into_vec();
        let mut gray_bytes = Vec::with_capacity((NORMALIZED_SIZE * NORMALIZED_SIZE) as usize);

        // Fast RGB to grayscale conversion using ITU-R BT.601 coefficients
        // L = 0.299*R + 0.587*G + 0.114*B
        for chunk in rgb_bytes.chunks_exact(3) {
            let r = chunk[0] as u32;
            let g = chunk[1] as u32;
            let b = chunk[2] as u32;
            // Using integer arithmetic for speed: (77*R + 150*G + 29*B) / 256
            let luma = ((77 * r + 150 * g + 29 * b) / 256) as u8;
            gray_bytes.push(luma);
        }

        // Convert to GrayImage
        GrayImage::from_raw(NORMALIZED_SIZE, NORMALIZED_SIZE, gray_bytes)
            .expect("valid grayscale image")
    }
}

/// Extracts center 32x32 region as normalized float buffer.
///
/// Returns pixel values in range [0.0, 1.0].
pub fn extract_global_region(image: &GrayImage) -> [f32; (PHASH_SIZE * PHASH_SIZE) as usize] {
    let start_x = (NORMALIZED_SIZE - PHASH_SIZE) / 2;
    let start_y = (NORMALIZED_SIZE - PHASH_SIZE) / 2;
    let mut buffer = [0.0f32; (PHASH_SIZE * PHASH_SIZE) as usize];

    // Optimized: iterate row by row for cache locality
    for y in 0..PHASH_SIZE {
        let row_start = ((start_y + y) * NORMALIZED_SIZE + start_x) as usize;
        let buf_start = (y * PHASH_SIZE) as usize;
        let row_pixels = image.as_raw();

        for x in 0..PHASH_SIZE {
            buffer[buf_start + x as usize] = row_pixels[row_start + x as usize] as f32 / 255.0;
        }
    }

    buffer
}

/// Extracts 4x4 grid of 64x64 blocks as float buffers.
///
/// Each block contains pixel values in range [0.0, 1.0].
pub fn extract_blocks(image: &GrayImage) -> [[f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16] {
    let mut blocks = [[0.0f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16];
    let pixels = image.as_raw();

    // Extract all blocks with cache-friendly access pattern
    for block_y in 0..4 {
        for block_x in 0..4 {
            let block_idx = (block_y * 4 + block_x) as usize;
            let start_x = block_x * BLOCK_SIZE;
            let start_y = block_y * BLOCK_SIZE;

            for y in 0..BLOCK_SIZE {
                let src_row_start = ((start_y + y) * NORMALIZED_SIZE + start_x) as usize;
                let dst_row_start = (y * BLOCK_SIZE) as usize;

                for x in 0..BLOCK_SIZE {
                    blocks[block_idx][dst_row_start + x as usize] =
                        pixels[src_row_start + x as usize] as f32 / 255.0;
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
