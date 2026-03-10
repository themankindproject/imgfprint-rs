//! Image decoding with dimension validation.

use crate::error::ImgFprintError;
use image::DynamicImage;
use std::io::Cursor;

const MAX_DIMENSION: u32 = 8192;
const MIN_DIMENSION: u32 = 32;

/// Decodes image bytes and validates dimensions.
///
/// Checks dimensions before full decode to prevent OOM attacks.
/// Maximum allowed dimension is 8192x8192 pixels.
/// Minimum required dimension is 32x32 pixels for fingerprinting.
pub fn decode_image(image_bytes: &[u8]) -> Result<DynamicImage, ImgFprintError> {
    if image_bytes.is_empty() {
        return Err(ImgFprintError::InvalidImage("empty input".to_string()));
    }

    let reader = image::ImageReader::new(Cursor::new(image_bytes))
        .with_guessed_format()
        .map_err(|e| ImgFprintError::DecodeError(format!("format detection failed: {}", e)))?;

    // Always check dimensions - fail closed if we can't determine them
    match reader.into_dimensions() {
        Ok((width, height)) => {
            if width > MAX_DIMENSION || height > MAX_DIMENSION {
                return Err(ImgFprintError::InvalidImage(format!(
                    "dimensions {}x{} exceed limit {}x{}",
                    width, height, MAX_DIMENSION, MAX_DIMENSION
                )));
            }
            if width < MIN_DIMENSION || height < MIN_DIMENSION {
                return Err(ImgFprintError::ImageTooSmall(format!(
                    "dimensions {}x{} are below minimum {}x{}",
                    width, height, MIN_DIMENSION, MIN_DIMENSION
                )));
            }
        }
        Err(e) => {
            // Fail closed - if we can't verify dimensions, don't proceed
            return Err(ImgFprintError::DecodeError(format!(
                "failed to read image dimensions: {}",
                e
            )));
        }
    }

    image::load_from_memory(image_bytes).map_err(|e| match e {
        image::ImageError::Unsupported(format) => {
            ImgFprintError::UnsupportedFormat(format!("{:?}", format))
        }
        image::ImageError::Decoding(err) => ImgFprintError::DecodeError(format!("{}", err)),
        _ => ImgFprintError::ProcessingError(format!("{}", e)),
    })
}
