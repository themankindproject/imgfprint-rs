//! Image decoding with dimension validation and EXIF orientation support.

use crate::error::ImgFprintError;
use exif::{In, Reader, Tag};
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

/// Reads EXIF orientation from image bytes.
/// Returns orientation value (1-8) or 1 if no EXIF data found.
fn read_exif_orientation(image_bytes: &[u8]) -> u32 {
    let mut cursor = Cursor::new(image_bytes);
    let exif_reader = Reader::new();

    match exif_reader.read_from_container(&mut cursor) {
        Ok(exif) => {
            if let Some(orientation_field) = exif.get_field(Tag::Orientation, In::PRIMARY) {
                if let Some(orientation) = orientation_field.value.get_uint(0) {
                    if (1..=8).contains(&orientation) {
                        return orientation;
                    }
                }
            }
        }
        Err(_) => {
            // No EXIF data or error reading it - default to no transformation
        }
    }

    1 // Default: no transformation
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

/// Decodes image bytes with a tunable [`PreprocessConfig`].
pub fn decode_image_with_config(
    image_bytes: &[u8],
    config: &PreprocessConfig,
) -> Result<DynamicImage, ImgFprintError> {
    if image_bytes.is_empty() {
        return Err(ImgFprintError::invalid_image("empty input"));
    }

    if image_bytes.len() > config.max_input_bytes {
        return Err(ImgFprintError::invalid_image(format!(
            "input too large: {} bytes exceeds limit of {} bytes",
            image_bytes.len(),
            config.max_input_bytes
        )));
    }

    let reader = image::ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| ImgFprintError::decode_error(format!("format detection failed: {}", e)))?;

    if let Ok((width, height)) = reader.into_dimensions() {
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
    }

    let image = image::load_from_memory(image_bytes).map_err(|e| match e {
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

    let orientation = read_exif_orientation(image_bytes);
    let oriented_image = apply_orientation_transform(image, orientation);

    let (final_w, final_h) = oriented_image.dimensions();
    if final_w > config.max_dimension || final_h > config.max_dimension {
        return Err(ImgFprintError::invalid_image(format!(
            "post-orientation dimensions {}x{} exceed limit {}x{}",
            final_w, final_h, config.max_dimension, config.max_dimension
        )));
    }
    if final_w < config.min_dimension || final_h < config.min_dimension {
        return Err(ImgFprintError::image_too_small(format!(
            "post-orientation dimensions {}x{} are below minimum {}x{}",
            final_w, final_h, config.min_dimension, config.min_dimension
        )));
    }

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
