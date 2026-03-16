//! Error types for image fingerprinting operations.

use thiserror::Error;

/// Errors that can occur during image fingerprinting.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
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

    /// Image dimensions are too small for fingerprinting.
    #[error("image dimensions too small: {0}")]
    ImageTooSmall(String),

    /// Embedding dimensions do not match between two embeddings.
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch { expected: usize, actual: usize },

    /// Provider error occurred during embedding generation.
    #[error("embedding provider error: {0}")]
    ProviderError(String),

    /// Invalid embedding data (empty vector, NaN values, etc.).
    #[error("invalid embedding: {0}")]
    InvalidEmbedding(String),
}

impl ImgFprintError {
    #[cold]
    #[inline(never)]
    pub fn decode_error(msg: impl Into<String>) -> Self {
        Self::DecodeError(msg.into())
    }

    #[cold]
    #[inline(never)]
    pub fn invalid_image(msg: impl Into<String>) -> Self {
        Self::InvalidImage(msg.into())
    }

    #[cold]
    #[inline(never)]
    pub fn processing_error(msg: impl Into<String>) -> Self {
        Self::ProcessingError(msg.into())
    }

    #[cold]
    #[inline(never)]
    pub fn image_too_small(msg: impl Into<String>) -> Self {
        Self::ImageTooSmall(msg.into())
    }
}
