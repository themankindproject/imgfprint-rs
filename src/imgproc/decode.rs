//! Image decoding with dimension validation and EXIF orientation support.

use crate::error::ImgFprintError;
use image::{DynamicImage, GenericImageView};
use std::io::Cursor;

/// Default maximum image edge length, in pixels. Beyond this, decode is rejected.
pub const DEFAULT_MAX_DIMENSION: u32 = 8192;
/// Default minimum image edge length, in pixels. Below this, decode is rejected.
pub const DEFAULT_MIN_DIMENSION: u32 = 32;
/// Default maximum input size, in bytes (50 MiB).
///
/// Caps memory exposure from maliciously large inputs while still admitting
/// typical high-resolution photos.
pub const DEFAULT_MAX_INPUT_BYTES: usize = 50 * 1024 * 1024;

/// Decode-time guards that an integrator (UCFP, server pipelines) can tune.
///
/// All defaults reproduce the historic hardcoded limits. Tighten them on
/// untrusted input paths; widen `max_input_bytes` and `max_dimension` only
/// for trusted batch jobs where OOM is acceptable risk.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessConfig {
    /// Reject inputs whose serialized byte size exceeds this. Enforced
    /// before any decode allocation and again at the file-read site.
    pub max_input_bytes: usize,
    /// Reject images where either edge exceeds this in pixels.
    pub max_dimension: u32,
    /// Reject images where either edge is below this in pixels.
    /// The fingerprinter pipeline needs at least 32 pixels per edge.
    pub min_dimension: u32,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            max_input_bytes: DEFAULT_MAX_INPUT_BYTES,
            max_dimension: DEFAULT_MAX_DIMENSION,
            min_dimension: DEFAULT_MIN_DIMENSION,
        }
    }
}

/// Reads EXIF orientation from JPEG image bytes.
/// Returns orientation value (1-8) or 1 if no EXIF data found.
///
/// Parses the JPEG APP1 (EXIF) marker directly without an external library.
/// Only looks for the Orientation tag (0x0112) in IFD0.
fn read_exif_orientation(image_bytes: &[u8]) -> u32 {
    // JPEG must start with SOI (0xFFD8)
    if image_bytes.len() < 4 || image_bytes[0] != 0xFF || image_bytes[1] != 0xD8 {
        return 1;
    }

    // Scan JPEG markers for APP1 (0xFFE1) containing EXIF
    let mut pos = 2;
    while pos + 4 <= image_bytes.len() {
        if image_bytes[pos] != 0xFF {
            return 1; // Invalid marker
        }
        let marker = image_bytes[pos + 1];

        // Skip padding 0xFF bytes
        if marker == 0xFF {
            pos += 1;
            continue;
        }

        // SOS (Start of Scan) — stop searching
        if marker == 0xDA {
            return 1;
        }

        // Marker segment length (big-endian, includes length bytes themselves)
        if pos + 4 > image_bytes.len() {
            return 1;
        }
        let seg_len = ((image_bytes[pos + 2] as usize) << 8) | (image_bytes[pos + 3] as usize);
        if seg_len < 2 {
            return 1;
        }

        // APP1 marker with EXIF header?
        if marker == 0xE1 {
            let seg_start = pos + 4; // after marker + length
            let seg_end = pos + 2 + seg_len;
            if seg_end > image_bytes.len() {
                return 1;
            }
            let seg_data = &image_bytes[seg_start..seg_end];

            // Check for "Exif\0\0" header (6 bytes)
            if seg_data.len() >= 6 && &seg_data[0..6] == b"Exif\0\0" {
                if let Some(orient) = parse_tiff_orientation(&seg_data[6..]) {
                    return orient;
                }
            }
        }

        // Advance past this marker segment
        pos += 2 + seg_len;
    }

    1
}

/// Parses TIFF/IFD0 data to extract the Orientation tag (0x0112).
fn parse_tiff_orientation(tiff: &[u8]) -> Option<u32> {
    if tiff.len() < 8 {
        return None;
    }

    // Byte order: "II" (little-endian) or "MM" (big-endian)
    let le = match (tiff[0], tiff[1]) {
        (b'I', b'I') => true,
        (b'M', b'M') => false,
        _ => return None,
    };

    // Validate TIFF magic number (42)
    let magic = read_u16(tiff, 2, le);
    if magic != 42 {
        return None;
    }

    // Offset to IFD0
    let ifd_offset = read_u32(tiff, 4, le) as usize;
    if ifd_offset + 2 > tiff.len() {
        return None;
    }

    // Number of IFD entries
    let entry_count = read_u16(tiff, ifd_offset, le) as usize;
    let entries_start = ifd_offset + 2;

    // Each IFD entry is 12 bytes: tag(2) + type(2) + count(4) + value/offset(4)
    for i in 0..entry_count {
        let entry_pos = entries_start + i * 12;
        if entry_pos + 12 > tiff.len() {
            return None;
        }

        let tag = read_u16(tiff, entry_pos, le);
        if tag == 0x0112 {
            // Orientation tag found
            // Type should be SHORT (3), count should be 1
            let value = read_u16(tiff, entry_pos + 8, le) as u32;
            if (1..=8).contains(&value) {
                return Some(value);
            }
            return None;
        }
    }

    None
}

#[inline]
fn read_u16(data: &[u8], offset: usize, le: bool) -> u16 {
    if le {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    }
}

#[inline]
fn read_u32(data: &[u8], offset: usize, le: bool) -> u32 {
    if le {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    } else {
        u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
}

/// Applies EXIF orientation transformation to an image.
fn apply_orientation_transform(image: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.rotate90().fliph(),
        6 => image.rotate90(),
        7 => image.rotate270().fliph(),
        8 => image.rotate270(),
        _ => image, // 1 or invalid - no transformation
    }
}

/// Decodes image bytes, validates dimensions, and applies EXIF orientation.
///
/// Uses the default [`PreprocessConfig`]; equivalent to
/// [`decode_image_with_config`] called with `&PreprocessConfig::default()`.
///
/// - Validates input size before processing to prevent OOM attacks.
/// - Maximum allowed dimension is 8192x8192 pixels (checked both before and after EXIF rotation).
/// - Minimum required dimension is 32x32 pixels for fingerprinting.
pub fn decode_image(image_bytes: &[u8]) -> Result<DynamicImage, ImgFprintError> {
    decode_image_with_config(image_bytes, &PreprocessConfig::default())
}

/// Rejects an inconsistent config where `min_dimension > max_dimension`.
///
/// Shared by [`validate_dimensions`] and the byte-decode path so the sanity
/// check lives in exactly one place.
fn check_config_sanity(config: &PreprocessConfig) -> Result<(), ImgFprintError> {
    if config.min_dimension > config.max_dimension {
        return Err(ImgFprintError::invalid_image(format!(
            "invalid config: min_dimension ({}) > max_dimension ({})",
            config.min_dimension, config.max_dimension
        )));
    }
    Ok(())
}

/// Validates decoded image dimensions against a [`PreprocessConfig`].
///
/// Shared by the byte-decode path and the already-decoded
/// [`FingerprinterContext::fingerprint_image`](crate::FingerprinterContext::fingerprint_image)
/// path so both entry points enforce identical min/max dimension guards.
///
/// # Errors
///
/// - [`ImgFprintError::InvalidImage`] if either edge exceeds `max_dimension`
///   or the config itself is inconsistent (`min_dimension > max_dimension`).
/// - [`ImgFprintError::ImageTooSmall`] if either edge is below `min_dimension`.
pub(crate) fn validate_dimensions(
    width: u32,
    height: u32,
    config: &PreprocessConfig,
) -> Result<(), ImgFprintError> {
    check_config_sanity(config)?;
    if width > config.max_dimension || height > config.max_dimension {
        return Err(ImgFprintError::invalid_image(format!(
            "dimensions {}x{} exceed limit {}x{}",
            width, height, config.max_dimension, config.max_dimension
        )));
    }
    if width < config.min_dimension || height < config.min_dimension {
        return Err(ImgFprintError::image_too_small(format!(
            "dimensions {}x{} are below minimum {}x{}",
            width, height, config.min_dimension, config.min_dimension
        )));
    }
    Ok(())
}

/// Decodes image bytes with a tunable [`PreprocessConfig`].
pub fn decode_image_with_config(
    image_bytes: &[u8],
    config: &PreprocessConfig,
) -> Result<DynamicImage, ImgFprintError> {
    if image_bytes.is_empty() {
        return Err(ImgFprintError::invalid_image("empty input"));
    }

    check_config_sanity(config)?;

    if image_bytes.len() > config.max_input_bytes {
        return Err(ImgFprintError::invalid_image(format!(
            "input too large: {} bytes exceeds limit of {} bytes",
            image_bytes.len(),
            config.max_input_bytes
        )));
    }

    // Early dimension check without full decode to reject oversized images cheaply
    if let Ok(reader) = image::ImageReader::new(Cursor::new(image_bytes)).with_guessed_format() {
        if let Ok((w, h)) = reader.into_dimensions() {
            validate_dimensions(w, h, config)?;
        }
    }

    // Use a reader with explicit decode limits to prevent decompression bombs.
    // A malicious PNG/GIF within the 50 MiB input cap could otherwise decompress
    // into gigabytes of RAM during decode.
    let image = {
        let mut reader = image::ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .map_err(|e| ImgFprintError::decode_error(format!("format detection failed: {}", e)))?;

        let mut limits = image::Limits::default();
        // Cap decoded pixel buffer: max_dimension² × 4 bytes (RGBA worst case).
        // For the default 8192×8192 config this allows up to 256 MiB decode buffer,
        // which is the absolute maximum a single legitimate image can require.
        limits.max_alloc = Some(config.max_dimension as u64 * config.max_dimension as u64 * 4);
        limits.max_image_width = Some(config.max_dimension);
        limits.max_image_height = Some(config.max_dimension);
        reader.limits(limits);

        reader.decode()
    }
    .map_err(|e| match e {
        image::ImageError::Unsupported(format) => {
            ImgFprintError::UnsupportedFormat(format!("{:?}", format))
        }
        image::ImageError::Decoding(err) => ImgFprintError::decode_error(err.to_string()),
        image::ImageError::IoError(io_err) => {
            ImgFprintError::decode_error(format!("I/O error: {}", io_err))
        }
        image::ImageError::Parameter(param_err) => {
            ImgFprintError::invalid_image(format!("parameter error: {}", param_err))
        }
        image::ImageError::Limits(limits_err) => {
            ImgFprintError::invalid_image(format!("limits exceeded: {}", limits_err))
        }
        other => ImgFprintError::ProcessingError(format!("image processing error: {}", other)),
    })?;

    let (width, height) = image.dimensions();
    validate_dimensions(width, height, config)?;

    let orientation = read_exif_orientation(image_bytes);
    let oriented_image = apply_orientation_transform(image, orientation);

    let (final_w, final_h) = oriented_image.dimensions();
    validate_dimensions(final_w, final_h, config).map_err(|e| match e {
        // Rebuild the post-orientation messages verbatim so callers see the
        // rotated size; the inner message already names the pre-rotation dims.
        ImgFprintError::InvalidImage(_) => ImgFprintError::invalid_image(format!(
            "post-orientation dimensions {}x{} exceed limit {}x{}",
            final_w, final_h, config.max_dimension, config.max_dimension
        )),
        ImgFprintError::ImageTooSmall(_) => ImgFprintError::image_too_small(format!(
            "post-orientation dimensions {}x{} are below minimum {}x{}",
            final_w, final_h, config.min_dimension, config.min_dimension
        )),
        other => other,
    })?;

    Ok(oriented_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_png_image(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    fn create_jpeg_image(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
        buf
    }

    #[test]
    fn test_decode_empty_input() {
        let result = decode_image(&[]);
        assert!(matches!(result, Err(ImgFprintError::InvalidImage(_))));
    }

    #[test]
    fn test_decode_invalid_data() {
        let result = decode_image(b"not an image");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_corrupted_png() {
        let mut corrupted = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        corrupted.extend_from_slice(&[0u8; 100]);
        let result = decode_image(&corrupted);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_valid_png() {
        let img_bytes = create_png_image(100, 100);
        let result = decode_image(&img_bytes);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_decode_valid_jpeg() {
        let img_bytes = create_jpeg_image(100, 100);
        let result = decode_image(&img_bytes);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 100);
    }

    #[test]
    fn test_decode_valid_gif() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Gif)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_valid_webp() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::WebP,
        )
        .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_valid_bmp() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Bmp)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_minimum_dimensions() {
        let img_bytes = create_png_image(32, 32);
        let result = decode_image(&img_bytes);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_too_small_width() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(31, 100, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(matches!(result, Err(ImgFprintError::ImageTooSmall(_))));
    }

    #[test]
    fn test_decode_too_small_height() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 31, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(matches!(result, Err(ImgFprintError::ImageTooSmall(_))));
    }

    #[test]
    fn test_decode_maximum_dimensions() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(8192, 8192, |_, _| Rgb([128u8, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_exceeds_maximum_width() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(8193, 100, |_, _| Rgb([128u8, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(matches!(result, Err(ImgFprintError::InvalidImage(_))));
    }

    #[test]
    fn test_decode_exceeds_maximum_height() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 8193, |_, _| Rgb([128u8, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(matches!(result, Err(ImgFprintError::InvalidImage(_))));
    }

    #[test]
    fn test_decode_non_square_image() {
        let img_bytes = create_png_image(1920, 1080);
        let result = decode_image(&img_bytes);
        assert!(result.is_ok());
        let img = result.unwrap();
        assert_eq!(img.width(), 1920);
        assert_eq!(img.height(), 1080);
    }

    #[test]
    fn test_decode_grayscale_image() {
        let img: ImageBuffer<image::Luma<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 100, |x, y| image::Luma([((x + y) % 256) as u8]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_rgba_image() {
        let img: ImageBuffer<image::Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(100, 100, |x, y| {
            image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_constants() {
        assert_eq!(DEFAULT_MAX_DIMENSION, 8192);
        assert_eq!(DEFAULT_MIN_DIMENSION, 32);
        assert_eq!(DEFAULT_MAX_INPUT_BYTES, 50 * 1024 * 1024);
    }

    #[test]
    fn test_preprocess_config_default() {
        let cfg = PreprocessConfig::default();
        assert_eq!(cfg.max_dimension, DEFAULT_MAX_DIMENSION);
        assert_eq!(cfg.min_dimension, DEFAULT_MIN_DIMENSION);
        assert_eq!(cfg.max_input_bytes, DEFAULT_MAX_INPUT_BYTES);
    }

    #[test]
    fn test_decode_with_tightened_max_input_bytes() {
        let img_bytes = create_png_image(100, 100);
        let tight = PreprocessConfig {
            max_input_bytes: 10,
            ..PreprocessConfig::default()
        };
        let result = decode_image_with_config(&img_bytes, &tight);
        assert!(matches!(result, Err(ImgFprintError::InvalidImage(_))));
    }

    #[test]
    fn test_decode_with_loosened_min_dimension() {
        // 31x31 normally fails. Drop the floor and it should pass.
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(31, 31, |_, _| Rgb([128u8, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let loose = PreprocessConfig {
            min_dimension: 16,
            ..PreprocessConfig::default()
        };
        assert!(decode_image_with_config(&buf, &loose).is_ok());

        // And still rejected by default.
        assert!(matches!(
            decode_image(&buf),
            Err(ImgFprintError::ImageTooSmall(_))
        ));
    }
}
