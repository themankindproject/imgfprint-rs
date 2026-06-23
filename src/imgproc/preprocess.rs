//! Image preprocessing and normalization with SIMD-accelerated resize.

use crate::error::ImgFprintError;
use fast_image_resize::images::{Image, ImageRef};
pub(crate) use fast_image_resize::CpuExtensions;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
#[cfg(test)]
use image::GrayImage;
use image::{DynamicImage, GenericImageView};
use std::sync::OnceLock;

/// Cached CPU extensions detection to avoid repeated feature checks.
static CPU_EXTENSIONS: OnceLock<CpuExtensions> = OnceLock::new();

/// Gets the optimal CPU extensions for the current platform.
/// This is cached to avoid the overhead of feature detection on every Preprocessor::new() call.
pub(crate) fn get_cpu_extensions() -> CpuExtensions {
    *CPU_EXTENSIONS.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                CpuExtensions::Avx2
            } else if std::is_x86_feature_detected!("sse4.1") {
                CpuExtensions::Sse4_1
            } else {
                CpuExtensions::None
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            #[cfg(target_arch = "aarch64")]
            {
                CpuExtensions::Neon
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                CpuExtensions::None
            }
        }
    })
}

/// ITU-R BT.601 luma coefficients for RGB to grayscale conversion.
/// These coefficients represent the relative luminance of each color channel.
/// See https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
const LUMA_COEFF_R: u32 = 77;
const LUMA_COEFF_G: u32 = 150;
const LUMA_COEFF_B: u32 = 29;
const LUMA_SHIFT: u32 = 8;

const NORMALIZED_SIZE: u32 = 256;
const BLOCK_SIZE: u32 = 64;
const PHASH_SIZE: u32 = 32;

// Compile-time assertions to ensure constants are valid
const _: () = assert!(NORMALIZED_SIZE == 256, "NORMALIZED_SIZE must be 256");
const _: () = assert!(BLOCK_SIZE == 64, "BLOCK_SIZE must be 64");
const _: () = assert!(PHASH_SIZE == 32, "PHASH_SIZE must be 32");
const _: () = assert!(
    BLOCK_SIZE * 4 == NORMALIZED_SIZE,
    "4 blocks must fit in normalized size"
);
const _: () = assert!(
    PHASH_SIZE * 8 == NORMALIZED_SIZE,
    "PHASH_SIZE must divide evenly"
);

/// Common bilinear resampling utility for hash algorithms.
///
/// Resamples a source buffer from (src_w × src_h) to (dst_w × dst_h)
/// using bilinear interpolation. Used by AHash, PHash, and DHash algorithms.
///
/// # Arguments
/// * `src` - Source pixel buffer in row-major order
/// * `src_w` - Source width
/// * `src_h` - Source height
/// * `dst` - Destination buffer (must be dst_w × dst_h elements)
/// * `dst_w` - Destination width
/// * `dst_h` - Destination height
#[inline(always)]
pub fn bilinear_resample(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    dst: &mut [f32],
    dst_w: usize,
    dst_h: usize,
) {
    // Fast path: identity resample (same dimensions) - just copy
    if src_w == dst_w && src_h == dst_h {
        dst.copy_from_slice(&src[..dst_w * dst_h]);
        return;
    }

    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;

    for y in 0..dst_h {
        let src_y = y as f32 * y_ratio;
        let y0 = src_y as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let dy = src_y - y0 as f32;

        for x in 0..dst_w {
            let src_x = x as f32 * x_ratio;
            let x0 = src_x as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let dx = src_x - x0 as f32;

            let i00 = y0 * src_w + x0;
            let i01 = y0 * src_w + x1;
            let i10 = y1 * src_w + x0;
            let i11 = y1 * src_w + x1;

            let v00 = src[i00];
            let v01 = src[i01];
            let v10 = src[i10];
            let v11 = src[i11];

            let v0 = v00 + (v01 - v00) * dx;
            let v1 = v10 + (v11 - v10) * dx;

            dst[y * dst_w + x] = v0 + (v1 - v0) * dy;
        }
    }
}

/// Preprocessor with cached resizer and CPU extension detection.
#[derive(Debug)]
pub struct Preprocessor {
    resizer: Resizer,
    dst_buffer: Vec<u8>,
    gray_buffer: Vec<u8>,
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

        // Use cached CPU extensions detection
        let cpu_extensions = get_cpu_extensions();
        if cpu_extensions != CpuExtensions::None {
            // SAFETY: get_cpu_extensions() only returns AVX2/SSE4.1/NEON after
            // verifying the CPU supports them via is_x86_feature_detected! or
            // target_arch checks. Using unsupported SIMD instructions would be UB.
            unsafe {
                resizer.set_cpu_extensions(cpu_extensions);
            }
        }

        Self {
            resizer,
            dst_buffer: Vec::with_capacity((NORMALIZED_SIZE * NORMALIZED_SIZE * 3) as usize),
            gray_buffer: Vec::with_capacity((NORMALIZED_SIZE * NORMALIZED_SIZE) as usize),
        }
    }

    /// Normalizes image to 256x256 grayscale using SIMD-accelerated resize.
    ///
    /// Uses Lanczos3 filtering for high-quality downsampling, then converts
    /// to grayscale. The SIMD-accelerated resize provides 3-4x speedup
    /// compared to the image crate's implementation.
    ///
    /// Applies EXIF orientation metadata if present.
    ///
    /// # Errors
    ///
    /// Returns `ImgFprintError::ProcessingError` if resize or conversion fails.
    #[cfg(test)]
    pub fn normalize(&mut self, image: &DynamicImage) -> Result<GrayImage, ImgFprintError> {
        let gray = self.normalize_as_slice(image)?;

        GrayImage::from_raw(NORMALIZED_SIZE, NORMALIZED_SIZE, gray.to_vec()).ok_or_else(|| {
            ImgFprintError::ProcessingError("failed to create grayscale image".to_string())
        })
    }

    /// Normalizes image to 256x256 grayscale and returns a borrowed reusable buffer.
    ///
    /// This is the hot-path API used by the fingerprinter. It keeps the
    /// grayscale allocation owned by the preprocessor so repeated fingerprint
    /// calls can reuse the same capacity instead of allocating a new 64 KiB
    /// image buffer every time.
    pub(crate) fn normalize_as_slice(
        &mut self,
        image: &DynamicImage,
    ) -> Result<&[u8], ImgFprintError> {
        let (src_w, src_h) = image.dimensions();

        // Reuse destination buffer to avoid allocation
        self.dst_buffer.clear();
        let target_len = (NORMALIZED_SIZE * NORMALIZED_SIZE * 3) as usize;
        self.dst_buffer.resize(target_len, 0u8);
        let dst_buffer = std::mem::take(&mut self.dst_buffer);

        let mut dst = Image::from_vec_u8(
            NORMALIZED_SIZE,
            NORMALIZED_SIZE,
            dst_buffer,
            PixelType::U8x3,
        )
        .map_err(|e| {
            ImgFprintError::ProcessingError(format!("invalid destination image: {}", e))
        })?;

        let options = ResizeOptions {
            algorithm: ResizeAlg::Convolution(FilterType::Lanczos3),
            ..Default::default()
        };

        // Use borrowed ImageRef for RGB8 to avoid allocation; convert otherwise
        let rgb_owned;
        match image {
            DynamicImage::ImageRgb8(rgb) => {
                let src =
                    ImageRef::new(src_w, src_h, rgb.as_raw(), PixelType::U8x3).map_err(|e| {
                        ImgFprintError::ProcessingError(format!("invalid source image: {}", e))
                    })?;
                self.resizer.resize(&src, &mut dst, &options).map_err(|e| {
                    ImgFprintError::ProcessingError(format!("resize failed: {}", e))
                })?;
            }
            _ => {
                rgb_owned = image.to_rgb8().into_raw();
                let src =
                    Image::from_vec_u8(src_w, src_h, rgb_owned, PixelType::U8x3).map_err(|e| {
                        ImgFprintError::ProcessingError(format!("invalid source image: {}", e))
                    })?;
                self.resizer.resize(&src, &mut dst, &options).map_err(|e| {
                    ImgFprintError::ProcessingError(format!("resize failed: {}", e))
                })?;
            }
        }

        let rgb_bytes = dst.into_vec();

        // Reuse grayscale buffer
        self.gray_buffer.clear();
        let gray_target_len = (NORMALIZED_SIZE * NORMALIZED_SIZE) as usize;
        self.gray_buffer.resize(gray_target_len, 0u8);

        // SIMD-friendly grayscale conversion with better cache locality
        // Process in chunks to improve CPU pipeline efficiency
        rgb_to_grayscale(&rgb_bytes, &mut self.gray_buffer);

        // Reclaim RGB buffer for reuse after grayscale conversion is complete
        self.dst_buffer = rgb_bytes;

        debug_assert_eq!(
            self.gray_buffer.len(),
            (NORMALIZED_SIZE * NORMALIZED_SIZE) as usize
        );

        Ok(&self.gray_buffer)
    }
}

/// RGB to grayscale conversion using ITU-R BT.601 luma coefficients.
///
/// Dispatches to a SIMD implementation when the CPU supports SSE4.1 (x86_64) or
/// NEON (aarch64); otherwise falls back to the scalar 4-pixel ILP-unrolled path.
/// The SIMD and scalar paths produce bit-identical output because the luma sum
/// (77·R + 150·G + 29·B) is at most 65280, which fits exactly in `u16`/`i16`,
/// so the arithmetic is order-preserving in either signed or unsigned 16-bit
/// interpretation.
///
/// # Arguments
/// * `rgb` - RGB bytes (3 bytes per pixel, RGBRGB... format)
/// * `gray` - Output grayscale buffer (1 byte per pixel)
#[inline(always)]
fn rgb_to_grayscale(rgb: &[u8], gray: &mut [u8]) {
    let cpu = get_cpu_extensions();

    #[cfg(target_arch = "x86_64")]
    {
        if cpu != CpuExtensions::None {
            // SAFETY: `get_cpu_extensions()` only returns SSE4.1 / AVX2 after
            // `is_x86_feature_detected!` confirmed the feature at runtime. AVX2
            // implies SSE4.1, so the SSE4.1-targeted intrinsic body is safe to
            // call under either detection.
            unsafe { rgb_to_grayscale_sse41(rgb, gray) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if cpu == CpuExtensions::Neon {
            // SAFETY: gated on `target_arch = "aarch64"` and runtime Neon detection.
            unsafe { rgb_to_grayscale_neon(rgb, gray) };
            return;
        }
    }

    rgb_to_grayscale_scalar(rgb, gray);
}

/// Scalar fallback for [`rgb_to_grayscale`]: 4-pixel ILP-unrolled BT.601 luma.
///
/// Used when `CpuExtensions::None` is detected, or as the SIMD tail for any
/// leftover bytes that don't fill a full SIMD chunk.
#[inline(always)]
fn rgb_to_grayscale_scalar(rgb: &[u8], gray: &mut [u8]) {
    debug_assert_eq!(gray.len(), rgb.len() / 3);

    // Process 4 pixels at a time for better instruction-level parallelism
    let len = rgb.len();
    let chunks = len / 12; // 12 bytes = 4 pixels
    let remainder = len % 12;

    for i in 0..chunks {
        let base = i * 12;
        let gray_base = i * 4;

        let r0 = rgb[base] as u32;
        let g0 = rgb[base + 1] as u32;
        let b0 = rgb[base + 2] as u32;

        let r1 = rgb[base + 3] as u32;
        let g1 = rgb[base + 4] as u32;
        let b1 = rgb[base + 5] as u32;

        let r2 = rgb[base + 6] as u32;
        let g2 = rgb[base + 7] as u32;
        let b2 = rgb[base + 8] as u32;

        let r3 = rgb[base + 9] as u32;
        let g3 = rgb[base + 10] as u32;
        let b3 = rgb[base + 11] as u32;

        gray[gray_base] =
            ((LUMA_COEFF_R * r0 + LUMA_COEFF_G * g0 + LUMA_COEFF_B * b0) >> LUMA_SHIFT) as u8;
        gray[gray_base + 1] =
            ((LUMA_COEFF_R * r1 + LUMA_COEFF_G * g1 + LUMA_COEFF_B * b1) >> LUMA_SHIFT) as u8;
        gray[gray_base + 2] =
            ((LUMA_COEFF_R * r2 + LUMA_COEFF_G * g2 + LUMA_COEFF_B * b2) >> LUMA_SHIFT) as u8;
        gray[gray_base + 3] =
            ((LUMA_COEFF_R * r3 + LUMA_COEFF_G * g3 + LUMA_COEFF_B * b3) >> LUMA_SHIFT) as u8;
    }

    // Handle remaining pixels
    let remainder_start = chunks * 12;
    for i in 0..remainder / 3 {
        let base = remainder_start + i * 3;
        let r = rgb[base] as u32;
        let g = rgb[base + 1] as u32;
        let b = rgb[base + 2] as u32;
        gray[chunks * 4 + i] =
            ((LUMA_COEFF_R * r + LUMA_COEFF_G * g + LUMA_COEFF_B * b) >> LUMA_SHIFT) as u8;
    }
}

/// SSE4.1 (and AVX2; AVX2 ⊇ SSE4.1) implementation of [`rgb_to_grayscale`].
///
/// Process 4 pixels (12 bytes RGB) per iteration. Loads 16 bytes per iteration
/// — the 4 trailing bytes are padding ignored via the shuffle masks. The last
/// iteration that can safely load 16 bytes is guarded by the `n_simd` bound
/// derived from `len >= 16`; any remaining bytes fall through to the scalar
/// tail (`rgb_to_grayscale_scalar`).
///
/// Channel deinterleaving uses `_mm_shuffle_epi8` with three precomputed masks
/// that select the R, G, or B byte of each of 4 pixels into the low 4 bytes of
/// an `__m128i`. Those bytes are zero-extended to 4×`i16`, multiplied by the
/// BT.601 coefficients (which fit in `i16`), summed, then logically shifted
/// right by 8 to extract the high byte — identical to the scalar `(sum >> 8)
/// as u8` operation because `sum` fits in `u16`/`i16` (max 65280).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn rgb_to_grayscale_sse41(rgb: &[u8], gray: &mut [u8]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(gray.len(), rgb.len() / 3);

    // Number of 4-pixel iterations where the 16-byte load is safe (base + 16 <= len).
    // `base = i * 12`, so we require `12 * i + 16 <= len`, i.e. `i <= (len - 16) / 12`.
    // Ensure at least one iteration is possible to avoid underflow.
    let n_simd: usize = if rgb.len() >= 16 {
        (rgb.len() - 16) / 12 + 1
    } else {
        0
    };

    let zero = _mm_setzero_si128();
    let coeff_r = _mm_set1_epi16(LUMA_COEFF_R as i16);
    let coeff_g = _mm_set1_epi16(LUMA_COEFF_G as i16);
    let coeff_b = _mm_set1_epi16(LUMA_COEFF_B as i16);

    // Channel-select shuffle masks: pick bytes at offsets {0,3,6,9} for R,
    // {1,4,7,10} for G, {2,5,8,11} for B, and set the high 12 lanes to zero via
    // the top-bit-set sentinel (0x80 / -1i8).
    let mask_r = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9, 6, 3, 0);
    let mask_g = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 7, 4, 1);
    let mask_b = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 8, 5, 2);

    for i in 0..n_simd {
        let rgb_base = i * 12;
        let gray_base = i * 4;

        // SAFETY: bounded by `n_simd` derivation above; `rgb_base + 16 <= len`.
        let v = _mm_loadu_si128(rgb.as_ptr().add(rgb_base) as *const __m128i);

        // Deinterleave R/G/B into the low 4 bytes of each register.
        let r_bytes = _mm_shuffle_epi8(v, mask_r);
        let g_bytes = _mm_shuffle_epi8(v, mask_g);
        let b_bytes = _mm_shuffle_epi8(v, mask_b);

        // Widen low 4 bytes to 4×i16 via interleave-with-zero. After
        // `_mm_unpacklo_epi8(x, 0)`, bytes layout is [b0, 0, b1, 0, b2, 0, b3, 0, ...]
        // which is the memory layout of 4 i16 values: [b0, b1, b2, b3, ...].
        let r_i16 = _mm_unpacklo_epi8(r_bytes, zero);
        let g_i16 = _mm_unpacklo_epi8(g_bytes, zero);
        let b_i16 = _mm_unpacklo_epi8(b_bytes, zero);

        // i16 multiply: low 16 bits of (b * coeff) — bit-pattern-identical to
        // the same computation in u16 (no signed-vs-unsigned ambiguity since we
        // only read the low 16 bits).
        let r_scaled = _mm_mullo_epi16(r_i16, coeff_r);
        let g_scaled = _mm_mullo_epi16(g_i16, coeff_g);
        let b_scaled = _mm_mullo_epi16(b_i16, coeff_b);

        // Sum: max sum = 65280 which fits in i16 bit pattern (0xFF00).
        let sum = _mm_add_epi16(_mm_add_epi16(r_scaled, g_scaled), b_scaled);

        // Logical shift right by 8 extracts the high byte of each i16 (= sum >> 8).
        // The result's bit pattern (low byte zeroed, high byte = Y) matches the
        // scalar `(sum as u32) >> 8` cast to u8.
        let shifted = _mm_srli_epi16(sum, LUMA_SHIFT as i32);

        // Extract the low byte of each i16 lane (bytes at offsets 0,2,4,6).
        // `_mm_extract_epi8` returns i32; cast as u8 truncates to the Y value.
        gray[gray_base] = _mm_extract_epi8(shifted, 0) as u8;
        gray[gray_base + 1] = _mm_extract_epi8(shifted, 2) as u8;
        gray[gray_base + 2] = _mm_extract_epi8(shifted, 4) as u8;
        gray[gray_base + 3] = _mm_extract_epi8(shifted, 6) as u8;
    }

    // Scalar tail handles: (a) the remaining 1-11 bytes if `len % 12 != 0`,
    // (b) the entire input if `len < 16` (n_simd = 0). The stores above wrote
    // exactly `n_simd * 4` bytes; the scalar tail fills the rest.
    let tail_rgb_start = n_simd * 12;
    let tail_gray_start = n_simd * 4;
    if tail_rgb_start < rgb.len() {
        rgb_to_grayscale_scalar(&rgb[tail_rgb_start..], &mut gray[tail_gray_start..]);
    }
}

/// NEON implementation of [`rgb_to_grayscale`].
///
/// Uses `vld3q_u8` to load 48 bytes (16 RGB pixels) and simultaneously
/// deinterleave them into 3 planar `uint8x16_t` vectors (`R`, `G`, `B`).
/// Each plane is then widened to `uint16x8_t` (split into low/high halves),
/// multiplied by the BT.601 coefficients, summed, and shifted right by 8 bits
/// before being narrowed back to `uint8x16_t` for a single store.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn rgb_to_grayscale_neon(rgb: &[u8], gray: &mut [u8]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(gray.len(), rgb.len() / 3);

    const PIXELS_PER_ITER: usize = 16;
    const BYTES_PER_ITER: usize = PIXELS_PER_ITER * 3; // 48

    let n_simd = rgb.len() / BYTES_PER_ITER;

    let coeff_r = vdupq_n_u16(LUMA_COEFF_R as u16);
    let coeff_g = vdupq_n_u16(LUMA_COEFF_G as u16);
    let coeff_b = vdupq_n_u16(LUMA_COEFF_B as u16);

    for i in 0..n_simd {
        let rgb_base = i * BYTES_PER_ITER;
        let gray_base = i * PIXELS_PER_ITER;

        // Load 48 bytes and deinterleave RGB into 3 planar vectors (16 bytes each).
        let planes = vld3q_u8(rgb.as_ptr().add(rgb_base));

        // Widen each 16-byte plane into two 8×u16 halves (low 8 / high 8 pixels).
        let r_lo = vmovl_u8(vget_low_u8(planes.val[0]));
        let r_hi = vmovl_u8(vget_high_u8(planes.val[0]));
        let g_lo = vmovl_u8(vget_low_u8(planes.val[1]));
        let g_hi = vmovl_u8(vget_high_u8(planes.val[1]));
        let b_lo = vmovl_u8(vget_low_u8(planes.val[2]));
        let b_hi = vmovl_u8(vget_high_u8(planes.val[2]));

        // u16 multiplies: max value 0x9552 ≤ 0xFFFF (no overflow into 17th bit).
        let rs_lo = vmulq_u16(r_lo, coeff_r);
        let rs_hi = vmulq_u16(r_hi, coeff_r);
        let gs_lo = vmulq_u16(g_lo, coeff_g);
        let gs_hi = vmulq_u16(g_hi, coeff_g);
        let bs_lo = vmulq_u16(b_lo, coeff_b);
        let bs_hi = vmulq_u16(b_hi, coeff_b);

        // Sum and shift right by 8 (logical).
        let sum_lo = vaddq_u16(vaddq_u16(rs_lo, gs_lo), bs_lo);
        let sum_hi = vaddq_u16(vaddq_u16(rs_hi, gs_hi), bs_hi);
        let y_lo = vshrq_n_u16(sum_lo, LUMA_SHIFT as i32);
        let y_hi = vshrq_n_u16(sum_hi, LUMA_SHIFT as i32);

        // Narrow each 8×u16 to 8×u8 (taking the low byte = Y) and combine into
        // a single 16-byte vector for a single store.
        let y_lo_u8 = vmovn_u16(y_lo);
        let y_hi_u8 = vmovn_u16(y_hi);
        let y_u8x16 = vcombine_u8(y_lo_u8, y_hi_u8);

        vst1q_u8(gray.as_mut_ptr().add(gray_base), y_u8x16);
    }

    // Scalar tail for remaining 1..47 bytes.
    let tail_rgb_start = n_simd * BYTES_PER_ITER;
    let tail_gray_start = n_simd * PIXELS_PER_ITER;
    if tail_rgb_start < rgb.len() {
        rgb_to_grayscale_scalar(&rgb[tail_rgb_start..], &mut gray[tail_gray_start..]);
    }
}

/// Extracts center 32x32 region as normalized float buffer.
#[inline]
#[cfg(test)]
pub fn extract_global_region(image: &GrayImage) -> [f32; (PHASH_SIZE * PHASH_SIZE) as usize] {
    extract_global_region_from_raw(image.as_raw())
}

/// Extracts center 32x32 region from a normalized 256x256 grayscale byte buffer.
#[inline]
pub(crate) fn extract_global_region_from_raw(
    pixels: &[u8],
) -> [f32; (PHASH_SIZE * PHASH_SIZE) as usize] {
    debug_assert_eq!(pixels.len(), (NORMALIZED_SIZE * NORMALIZED_SIZE) as usize);

    let start_x = (NORMALIZED_SIZE - PHASH_SIZE) / 2;
    let start_y = (NORMALIZED_SIZE - PHASH_SIZE) / 2;
    let mut buffer = [0.0f32; (PHASH_SIZE * PHASH_SIZE) as usize];
    const SCALE: f32 = 1.0 / 255.0;

    // Optimized: process row by row with better cache locality
    for y in 0..PHASH_SIZE as usize {
        let src_row_start = (start_y as usize + y) * NORMALIZED_SIZE as usize + start_x as usize;
        let dst_row_start = y * PHASH_SIZE as usize;

        // Unroll inner loop for better performance
        for x in 0..PHASH_SIZE as usize {
            buffer[dst_row_start + x] = pixels[src_row_start + x] as f32 * SCALE;
        }
    }

    buffer
}

/// Extracts 4x4 grid of 64x64 blocks as float buffers.
///
/// Optimized for cache locality by processing the image in a single pass
/// and distributing pixels to their respective blocks.
#[inline]
#[cfg(test)]
pub fn extract_blocks(image: &GrayImage) -> [[f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16] {
    extract_blocks_from_raw(image.as_raw())
}

/// Extracts 4x4 grid of 64x64 blocks from a normalized 256x256 grayscale byte buffer.
#[inline]
pub(crate) fn extract_blocks_from_raw(
    pixels: &[u8],
) -> [[f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16] {
    debug_assert_eq!(pixels.len(), (NORMALIZED_SIZE * NORMALIZED_SIZE) as usize);

    let mut blocks = [[0.0f32; (BLOCK_SIZE * BLOCK_SIZE) as usize]; 16];
    const SCALE: f32 = 1.0 / 255.0;

    // Optimized: single pass through the image with better cache locality
    // Process each block row
    for block_y in 0..4 {
        let start_y = block_y * BLOCK_SIZE;
        for block_x in 0..4 {
            let block_idx = (block_y * 4 + block_x) as usize;
            let start_x = block_x * BLOCK_SIZE;
            let block = &mut blocks[block_idx];

            // Process each row in the block
            for y in 0..BLOCK_SIZE as usize {
                let src_row = ((start_y + y as u32) * NORMALIZED_SIZE + start_x) as usize;
                let dst_row = y * BLOCK_SIZE as usize;

                // Process pixels in the row
                for x in 0..BLOCK_SIZE as usize {
                    block[dst_row + x] = pixels[src_row + x] as f32 * SCALE;
                }
            }
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_bilinear_resample_same_size() {
        let src: [f32; 4] = [0.0, 0.5, 0.5, 1.0];
        let mut dst = [0.0f32; 4];
        bilinear_resample(&src, 2, 2, &mut dst, 2, 2);

        assert!((dst[0] - 0.0).abs() < 1e-6);
        assert!((dst[1] - 0.5).abs() < 1e-6);
        assert!((dst[2] - 0.5).abs() < 1e-6);
        assert!((dst[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_resample_downsample() {
        let src: [f32; 16] = [
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut dst = [0.0f32; 4];
        bilinear_resample(&src, 4, 4, &mut dst, 2, 2);

        for &v in &dst {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_bilinear_resample_uniform() {
        let src: [f32; 64] = [0.5; 64];
        let mut dst = [0.0f32; 16];
        bilinear_resample(&src, 8, 8, &mut dst, 4, 4);

        for &v in &dst {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bilinear_resample_gradient() {
        let mut src = [0.0f32; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                src[y * 32 + x] = x as f32 / 31.0;
            }
        }

        let mut dst = [0.0f32; 16 * 16];
        bilinear_resample(&src, 32, 32, &mut dst, 16, 16);

        assert!((dst[0] - 0.0).abs() < 0.1);
        assert!((dst[15] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_bilinear_resample_zeros() {
        let src: [f32; 16] = [0.0; 16];
        let mut dst = [0.0f32; 4];
        bilinear_resample(&src, 4, 4, &mut dst, 2, 2);

        for &v in &dst {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_rgb_to_grayscale_uniform() {
        let rgb = vec![128u8; 36];
        let mut gray = vec![0u8; 12];
        rgb_to_grayscale(&rgb, &mut gray);

        for &g in &gray {
            assert!(g > 0 && g < 255);
        }
    }

    #[test]
    fn test_rgb_to_grayscale_red() {
        let mut rgb = vec![0u8; 36];
        for i in 0..12 {
            rgb[i * 3] = 255;
        }
        let mut gray = vec![0u8; 12];
        rgb_to_grayscale(&rgb, &mut gray);

        for &g in &gray {
            assert!(g > 0 && g < 255);
        }
    }

    #[test]
    fn test_rgb_to_grayscale_remainder() {
        let rgb = vec![100u8; 39];
        let mut gray = vec![0u8; 13];
        rgb_to_grayscale(&rgb, &mut gray);

        assert_eq!(gray.len(), 13);
        for &g in &gray {
            assert!(g > 0);
        }
    }

    #[test]
    fn test_rgb_to_grayscale_black() {
        let rgb = vec![0u8; 12];
        let mut gray = vec![0u8; 4];
        rgb_to_grayscale(&rgb, &mut gray);

        for &g in &gray {
            assert_eq!(g, 0);
        }
    }

    #[test]
    fn test_rgb_to_grayscale_white() {
        let rgb = vec![255u8; 12];
        let mut gray = vec![0u8; 4];
        rgb_to_grayscale(&rgb, &mut gray);

        for &g in &gray {
            assert_eq!(g, 255);
        }
    }

    /// Parity test: SIMD path must produce bit-identical output to scalar for all
    /// input sizes that exercise the SIMD bulk, the no-SIMD early-out (`len < 16`),
    /// and the SIMD tail (`len % 12 != 0`).
    #[test]
    fn test_rgb_to_grayscale_simd_matches_scalar() {
        // Sizes (in bytes of RGB) probe the SIMD/scalar boundary and tail.
        // All are multiples of 3 (the library's public invariant for this fn).
        //   3 — 1 px, below the 16-byte load floor → pure scalar path
        //  24 — 8 px, 1 SIMD iter, no tail
        //  27 — 9 px, 1 SIMD iter, 4-byte tail
        //  36 — 12 px, 2 SIMD iters, 4-px tail
        //  48 — 16 px, 3 SIMD iters, 4-px tail  (n_simd stops at floor((48-16)/12)+1=3)
        //  63 — 21 px, irregular tail
        //  96 — 32 px, exercised by 256×256 normalization pass
        //  300 — 100 px, an arbitrary larger oddity
        let sizes: [usize; 8] = [3, 24, 27, 36, 48, 63, 96, 300];

        // Deterministic PRNG: LinCon (LCG) with fixed seed so each pixel's RGB
        // triplet is uncorrelated with its neighbors.
        let build_rgb = |n: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(n);
            let mut s: u32 = 0x6d4b_7f15;
            for _ in 0..n {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push((s >> 24) as u8);
            }
            v
        };

        for &n in &sizes {
            assert!(n % 3 == 0, "test invariant: n must be a multiple of 3");
            let rgb = build_rgb(n);
            let npix = n / 3;

            let mut gray_scalar = vec![0u8; npix];
            rgb_to_grayscale_scalar(&rgb, &mut gray_scalar);

            let mut gray_dispatch = vec![0u8; npix];
            rgb_to_grayscale(&rgb, &mut gray_dispatch);

            assert_eq!(
                gray_scalar,
                gray_dispatch,
                "SIMD/scalar mismatch at rgb len {} ({} px): scalar={:02x?}, dispatch={:02x?}",
                n,
                npix,
                &gray_scalar[..gray_scalar.len().min(16)],
                &gray_dispatch[..gray_dispatch.len().min(16)],
            );
        }
    }

    /// Bit-identical parity test for the standard 256×256 image used everywhere
    /// else in the library. This guards against regressions introduced when the
    /// SIMD path's tail logic is touched.
    #[test]
    fn test_rgb_to_grayscale_simd_matches_scalar_256x256() {
        let n = (256 * 256) * 3;
        let rgb: Vec<u8> = (0..n as u32).map(|i| (i & 0xFF) as u8).collect();

        let mut gray_scalar = vec![0u8; 256 * 256];
        rgb_to_grayscale_scalar(&rgb, &mut gray_scalar);

        let mut gray_dispatch = vec![0u8; 256 * 256];
        rgb_to_grayscale(&rgb, &mut gray_dispatch);

        assert_eq!(gray_scalar, gray_dispatch);
    }

    #[test]
    fn test_extract_global_region_uniform() {
        let img = GrayImage::from_pixel(256, 256, Luma([128u8]));
        let region = extract_global_region(&img);

        assert_eq!(region.len(), 32 * 32);
        for &v in &region {
            assert!((v - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_extract_global_region_gradient() {
        let mut img = GrayImage::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                img.put_pixel(x, y, Luma([((x + y) / 2) as u8]));
            }
        }
        let region = extract_global_region(&img);

        assert_eq!(region.len(), 32 * 32);
        assert!(region[0] < region[region.len() - 1]);
    }

    #[test]
    fn test_extract_blocks_uniform() {
        let img = GrayImage::from_pixel(256, 256, Luma([128u8]));
        let blocks = extract_blocks(&img);

        assert_eq!(blocks.len(), 16);
        for block in &blocks {
            assert_eq!(block.len(), 64 * 64);
            for &v in block {
                assert!((v - 0.5).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_extract_blocks_positions() {
        let img = GrayImage::from_pixel(256, 256, Luma([128u8]));
        let blocks = extract_blocks(&img);

        assert_eq!(blocks.len(), 16);

        for (i, block) in blocks.iter().enumerate() {
            assert_eq!(block.len(), 64 * 64);
            let block_x = i % 4;
            let block_y = i / 4;
            assert!(block_x < 4 && block_y < 4);
        }
    }

    #[test]
    fn test_extract_blocks_different_regions() {
        let mut img = GrayImage::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                let value = if x < 128 && y < 128 { 255u8 } else { 0u8 };
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let blocks = extract_blocks(&img);

        assert!((blocks[0].iter().sum::<f32>() / (64.0 * 64.0)) > 0.4);
        assert!((blocks[15].iter().sum::<f32>() / (64.0 * 64.0)) < 0.1);
    }

    #[test]
    fn test_preprocessor_new() {
        let preprocessor = Preprocessor::new();
        assert!(preprocessor.dst_buffer.is_empty());
        assert!(preprocessor.gray_buffer.is_empty());
    }

    #[test]
    fn test_preprocessor_normalize_uniform() {
        let mut preprocessor = Preprocessor::new();
        let img = GrayImage::from_pixel(256, 256, Luma([128u8]));
        let dynamic = DynamicImage::ImageLuma8(img);

        let result = preprocessor.normalize(&dynamic);
        assert!(result.is_ok());

        let gray = result.unwrap();
        assert_eq!(gray.width(), 256);
        assert_eq!(gray.height(), 256);
    }

    #[test]
    fn test_preprocessor_normalize_small_image() {
        let mut preprocessor = Preprocessor::new();
        let img = GrayImage::from_pixel(64, 64, Luma([128u8]));
        let dynamic = DynamicImage::ImageLuma8(img);

        let result = preprocessor.normalize(&dynamic);
        assert!(result.is_ok());

        let gray = result.unwrap();
        assert_eq!(gray.width(), 256);
        assert_eq!(gray.height(), 256);
    }

    #[test]
    fn test_preprocessor_reuse() {
        let mut preprocessor = Preprocessor::new();
        let img1 = GrayImage::from_pixel(100, 100, Luma([50u8]));
        let img2 = GrayImage::from_pixel(200, 200, Luma([200u8]));

        let dynamic1 = DynamicImage::ImageLuma8(img1);
        let dynamic2 = DynamicImage::ImageLuma8(img2);

        let result1 = preprocessor.normalize(&dynamic1);
        let result2 = preprocessor.normalize(&dynamic2);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[test]
    fn test_constants_compile_time_assertions() {
        assert_eq!(NORMALIZED_SIZE, 256);
        assert_eq!(BLOCK_SIZE, 64);
        assert_eq!(PHASH_SIZE, 32);
    }

    #[test]
    fn test_luma_coefficients() {
        assert_eq!(LUMA_COEFF_R, 77);
        assert_eq!(LUMA_COEFF_G, 150);
        assert_eq!(LUMA_COEFF_B, 29);
        assert_eq!(LUMA_SHIFT, 8);
    }
}
