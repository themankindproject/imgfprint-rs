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

    /// Image data is invalid or exceeds the configured limits.
    ///
    /// ## Errors
    /// This error occurs when:
    /// - The input is empty or larger than `PreprocessConfig::max_input_bytes`
    /// - An edge exceeds `PreprocessConfig::max_dimension` (default 8192 px)
    /// - The decoder's allocation limit is exceeded (decompression bomb guard)
    /// - The `PreprocessConfig` itself is inconsistent (`min_dimension > max_dimension`)
    #[error("invalid image: {0}")]
    InvalidImage(String),

    /// Image format is not supported or could not be detected.
    ///
    /// PNG, JPEG, GIF, WebP, and BMP are always supported; TIFF, ICO, PNM,
    /// and QOI are behind the crate features of the same names.
    ///
    /// TGA is intentionally unsupported: it carries no magic bytes, so the
    /// byte-sniffing decode path cannot identify it.
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
    /// This error occurs when either edge is below
    /// `PreprocessConfig::min_dimension` (default 32 px).
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

    /// I/O error reading the image source (file open/read failures).
    ///
    /// ## Errors
    /// This error occurs when:
    /// - The path passed to [`fingerprint_path`](crate::ImageFingerprinter::fingerprint_path) does not exist or is unreadable
    /// - The file is larger than `PreprocessConfig::max_input_bytes`
    ///   (default 50 MiB), detected from metadata or while reading
    /// - The underlying read fails partway through
    #[error("io error: {0}")]
    IoError(String),

    /// A caller-supplied configuration is invalid.
    ///
    /// ## Errors
    /// This error occurs when:
    /// - A [`MultiHashConfig`](crate::MultiHashConfig) contains NaN,
    ///   infinite, or negative weights (see `MultiHashConfig::validate`)
    /// - A block distance threshold is outside the valid 0–64 range
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Encoded fingerprint bytes could not be decoded.
    ///
    /// ## Errors
    /// Returned by `ImageFingerprint::from_bytes` and
    /// `MultiHashFingerprint::from_bytes` when the input has the wrong
    /// length, magic bytes, or kind, or was written under a different
    /// [`FORMAT_VERSION`](crate::FORMAT_VERSION) (recompute it in that case).
    #[error("invalid fingerprint encoding: {0}")]
    InvalidFingerprint(String),
}

impl From<std::io::Error> for ImgFprintError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err.to_string())
    }
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

    /// Creates an invalid config error with the given message.
    #[cold]
    #[inline(never)]
    pub fn invalid_config(msg: impl Into<String>) -> Self {
        Self::InvalidConfig(msg.into())
    }
}

// Compile-time assertions to ensure ImgFprintError is Send + Sync
const _: () = {
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}
    assert_send::<ImgFprintError>();
    assert_sync::<ImgFprintError>();
};
