//! Image decoding with dimension validation.

use crate::error::ImgFprintError;
use image::DynamicImage;
use std::io::Cursor;

const MAX_DIMENSION: u32 = 8192;
const MIN_DIMENSION: u32 = 32;

/// Maximum input size: 50MB - prevents memory exhaustion from maliciously large inputs.
/// This limit prevents OOM attacks while still allowing typical high-res images.
const MAX_INPUT_BYTES: usize = 50 * 1024 * 1024;

/// Decodes image bytes and validates dimensions.
///
/// - Checks dimensions before full decode to prevent OOM attacks.
/// - Maximum allowed dimension is 8192x8192 pixels.
/// - Minimum required dimension is 32x32 pixels for fingerprinting.
pub fn decode_image(image_bytes: &[u8]) -> Result<DynamicImage, ImgFprintError> {
    if image_bytes.is_empty() {
        return Err(ImgFprintError::invalid_image("empty input"));
    }

    if image_bytes.len() > MAX_INPUT_BYTES {
        return Err(ImgFprintError::invalid_image(format!(
            "input too large: {} bytes exceeds limit of {} bytes",
            image_bytes.len(),
            MAX_INPUT_BYTES
        )));
    }

    let reader = image::ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| ImgFprintError::decode_error(format!("format detection failed: {}", e)))?;

    // Always check dimensions - fail closed if we can't determine them
    match reader.into_dimensions() {
        Ok((width, height)) => {
            if width > MAX_DIMENSION || height > MAX_DIMENSION {
                return Err(ImgFprintError::invalid_image(format!(
                    "dimensions {}x{} exceed limit {}x{}",
                    width, height, MAX_DIMENSION, MAX_DIMENSION
                )));
            }
            if width < MIN_DIMENSION || height < MIN_DIMENSION {
                return Err(ImgFprintError::image_too_small(format!(
                    "dimensions {}x{} are below minimum {}x{}",
                    width, height, MIN_DIMENSION, MIN_DIMENSION
                )));
            }
        }
        Err(e) => {
            // Fail closed - if we can't verify dimensions, don't proceed
            return Err(ImgFprintError::decode_error(format!(
                "failed to read image dimensions: {}",
                e
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
        // Preserve error details for all other cases
        other => ImgFprintError::ProcessingError(format!("image processing error: {}", other)),
    })?;
    
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_png_image(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(width, height, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    fn create_jpeg_image(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(width, height, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)
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
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 100, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Gif)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_valid_webp() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 100, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::WebP)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_valid_bmp() {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 100, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
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
        let img: ImageBuffer<image::Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_fn(100, 100, |x, y| image::Rgba([(x % 256) as u8, (y % 256) as u8, 128, 255]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        let result = decode_image(&buf);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decode_constants() {
        assert_eq!(MAX_DIMENSION, 8192);
        assert_eq!(MIN_DIMENSION, 32);
        assert_eq!(MAX_INPUT_BYTES, 50 * 1024 * 1024);
    }
}
