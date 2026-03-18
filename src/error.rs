//! Error types for image fingerprinting operations.

use thiserror::Error;

/// Errors that can occur during image fingerprinting.
///
/// This error type implements both `Send` and `Sync` for safe use across
/// threads and async contexts.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ImgFprintError {
    /// Image decoding failed (corrupted data or unsupported format).
    ///
    /// ## Errors
    /// This error occurs when:
    /// - The image data is corrupted or truncated
    /// - The image format cannot be detected from the magic bytes
    /// - The underlying image decoder fails (e.g., invalid compression)
    #[error("decode failed: {0}")]
    DecodeError(String),

    /// Image data is invalid or dimensions exceed limits (8192x8192 max).
    ///
    /// ## Errors
    /// This error occurs when:
    /// - Image dimensions exceed 8192x8192 pixels
    /// - Image dimensions are reported as zero or negative
    /// - The image header contains invalid values
    /// - Color type conversion fails
    #[error("invalid image: {0}")]
    InvalidImage(String),

    /// Image format is not supported (supported: PNG, JPEG, GIF, WebP, BMP).
    ///
    /// ## Errors
    /// This error occurs when:
    /// - The file extension or magic bytes don't match a supported format
    /// - A specific variant of a format is not supported (e.g., JPEG 2000)
    /// - The image uses an unsupported color space
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    /// Internal processing error during fingerprint computation.
    ///
    /// ## Errors
    /// This error occurs when:
    /// - DCT computation fails
    /// - Memory allocation fails during processing
    /// - An invariant in the algorithm is violated
    #[error("processing error: {0}")]
    ProcessingError(String),

    /// Image dimensions are too small for fingerprinting.
    ///
    /// ## Errors
    /// This error occurs when:
    /// - Either dimension is less than 8 pixels
    /// - The image cannot be resized to the minimum 8x8 required for hashing
    #[error("image dimensions too small: {0}")]
    ImageTooSmall(String),

    /// Embedding dimensions do not match between two embeddings.
    ///
    /// ## Errors
    /// This error occurs when:
    /// - Comparing embeddings from different models with different output dimensions
    /// - Calling [`semantic_similarity`](crate::embed::semantic_similarity) with mismatched vectors
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch {
        /// Expected dimension from the first embedding
        expected: usize,
        /// Actual dimension from the second embedding
        actual: usize,
    },

    /// Provider error occurred during embedding generation.
    ///
    /// ## Errors
    /// This error occurs when:
    /// - ONNX model file cannot be loaded
    /// - Model inference fails at runtime
    /// - The model produces an unexpected output shape
    #[error("embedding provider error: {0}")]
    ProviderError(String),

    /// Invalid embedding data (empty vector, NaN values, etc.).
    ///
    /// ## Errors
    /// This error occurs when:
    /// - The embedding vector is empty
    /// - The vector contains NaN or infinity values
    /// - Creating an [`Embedding`](crate::embed::Embedding) with invalid data
    /// - Calling [`semantic_similarity`](crate::embed::semantic_similarity) with invalid embeddings
    #[error("invalid embedding: {0}")]
    InvalidEmbedding(String),
}

impl ImgFprintError {
    /// Creates a decode error with the given message.
    #[cold]
    #[inline(never)]
    pub fn decode_error(msg: impl Into<String>) -> Self {
        Self::DecodeError(msg.into())
    }

    /// Creates an invalid image error with the given message.
    #[cold]
    #[inline(never)]
    pub fn invalid_image(msg: impl Into<String>) -> Self {
        Self::InvalidImage(msg.into())
    }

    /// Creates a processing error with the given message.
    #[cold]
    #[inline(never)]
    pub fn processing_error(msg: impl Into<String>) -> Self {
        Self::ProcessingError(msg.into())
    }

    /// Creates an image too small error with the given message.
    #[cold]
    #[inline(never)]
    pub fn image_too_small(msg: impl Into<String>) -> Self {
        Self::ImageTooSmall(msg.into())
    }
}

// Compile-time assertions to ensure ImgFprintError is Send + Sync
const _: () = {
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}
    assert_send::<ImgFprintError>();
    assert_sync::<ImgFprintError>();
};
