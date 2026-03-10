//! Error types for image fingerprinting operations.

use thiserror::Error;

/// Errors that can occur during image fingerprinting.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ImgFprintError {
    /// Image decoding failed (corrupted data or unsupported format).
    #[error("decode failed: {0}")]
    DecodeError(String),

    /// Image data is invalid or dimensions exceed limits (8192x8192 max).
    #[error("invalid image: {0}")]
    InvalidImage(String),

    /// Image format is not supported (supported: PNG, JPEG, GIF, WebP, BMP).
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Internal processing error during fingerprint computation.
    #[error("processing error: {0}")]
    ProcessingError(String),
}
