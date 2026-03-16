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

    image::load_from_memory(image_bytes).map_err(|e| match e {
        image::ImageError::Unsupported(format) => {
            ImgFprintError::UnsupportedFormat(format!("{:?}", format))
        }
        image::ImageError::Decoding(err) => ImgFprintError::decode_error(err.to_string()),
        _ => ImgFprintError::processing_error(e.to_string()),
    })
}
