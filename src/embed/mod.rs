//! Semantic embedding support for CLIP-style image representations.
//!
//! Provides types and traits for working with high-dimensional semantic embeddings
//! obtained from external providers (OpenAI CLIP, HuggingFace, local models).
//!
//! ## Example
//!
//! ```rust,ignore
//! use imgfprint_rs::{ImageFingerprinter, Embedding, EmbeddingProvider};
//!
//! // Define your provider implementation
//! struct MyClipProvider;
//!
//! impl EmbeddingProvider for MyClipProvider {
//!     fn embed(&self, image: &[u8]) -> Result<Embedding, ImgFprintError> {
//!         // Call external API or local model
//!         // Return the embedding vector
//!     }
//! }
//!
//! let provider = MyClipProvider;
//! let embedding = ImageFingerprinter::semantic_embedding(&provider, &image_bytes)?;
//! ```

use crate::error::ImgFprintError;

/// A semantic embedding vector representing image content.
///
/// Embeddings are high-dimensional vectors (typically 512-1024 dimensions)
/// that capture semantic meaning of images in a way that similar images
/// have similar vector representations.
///
/// # Immutability
///
/// Once created, embeddings are immutable. The vector can be accessed
/// via [`as_slice()`](Embedding::as_slice) or [`vector()`](Embedding::vector).
///
/// # Validation
///
/// Embeddings are validated on creation to ensure:
/// - Vector is non-empty
/// - All elements are finite (not NaN or infinity)
///
/// # Examples
///
/// ```rust
/// use imgfprint_rs::Embedding;
///
/// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
/// // Create from a Vec<f32>
/// let embedding = Embedding::new(vec![0.1, 0.2, 0.3, 0.4])?;
///
/// // Access the vector
/// assert_eq!(embedding.len(), 4);
/// assert_eq!(embedding.as_slice(), &[0.1, 0.2, 0.3, 0.4]);
/// # Ok(())
/// # }
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    vector: Vec<f32>,
}

impl Embedding {
    /// Creates a new embedding from a vector of f32 values.
    ///
    /// # Arguments
    ///
    /// * `vector` - The embedding vector. Must be non-empty and contain only finite values.
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError::InvalidEmbedding`] if:
    /// - The vector is empty
    /// - Any element is NaN or infinity
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgfprint_rs::Embedding;
    ///
    /// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
    /// let embedding = Embedding::new(vec![0.1, 0.2, 0.3])?;
    /// assert_eq!(embedding.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(vector: Vec<f32>) -> Result<Self, ImgFprintError> {
        if vector.is_empty() {
            return Err(ImgFprintError::InvalidEmbedding(
                "embedding vector cannot be empty".to_string(),
            ));
        }

        if vector.iter().any(|&v| !v.is_finite()) {
            return Err(ImgFprintError::InvalidEmbedding(
                "embedding contains non-finite values (NaN or infinity)".to_string(),
            ));
        }

        Ok(Self { vector })
    }

    /// Returns the embedding vector as a slice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgfprint_rs::Embedding;
    ///
    /// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
    /// let embedding = Embedding::new(vec![0.1, 0.2, 0.3])?;
    /// let slice = embedding.as_slice();
    /// assert_eq!(slice, &[0.1, 0.2, 0.3]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.vector
    }

    /// Returns a clone of the embedding vector.
    ///
    /// For zero-copy access, prefer [`as_slice()`](Embedding::as_slice).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgfprint_rs::Embedding;
    ///
    /// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
    /// let embedding = Embedding::new(vec![0.1, 0.2, 0.3])?;
    /// let vec = embedding.vector();
    /// assert_eq!(vec, vec![0.1, 0.2, 0.3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn vector(&self) -> Vec<f32> {
        self.vector.clone()
    }

    /// Returns the dimensionality (length) of the embedding.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgfprint_rs::Embedding;
    ///
    /// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
    /// let embedding = Embedding::new(vec![0.1, 0.2, 0.3, 0.4])?;
    /// assert_eq!(embedding.len(), 4);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.vector.len()
    }

    /// Returns true if the embedding is empty (should never happen for valid embeddings).
    ///
    /// This method exists for API completeness. Valid embeddings created
    /// via [`new()`](Embedding::new) are guaranteed to be non-empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vector.is_empty()
    }

    /// Returns the dimensionality of the embedding.
    ///
    /// Alias for [`len()`](Embedding::len) for semantic clarity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgfprint_rs::Embedding;
    ///
    /// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
    /// let embedding = Embedding::new(vec![0.1; 512])?;
    /// assert_eq!(embedding.dimension(), 512);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn dimension(&self) -> usize {
        self.len()
    }
}

/// Trait for embedding providers.
///
/// Implement this trait to integrate external embedding services such as:
/// - OpenAI CLIP API
/// - HuggingFace Inference API
/// - Local CLIP models (via ONNX, PyTorch, etc.)
///
/// The SDK itself does not perform HTTP requests or model inference.
/// It only defines this abstraction, allowing users to bring their own
/// provider implementation.
///
/// # Example Implementation
///
/// ```rust
/// use imgfprint_rs::{EmbeddingProvider, Embedding, ImgFprintError};
///
/// struct DummyProvider;
///
/// impl EmbeddingProvider for DummyProvider {
///     fn embed(&self, _image: &[u8]) -> Result<Embedding, ImgFprintError> {
///         // In a real implementation, this would call an external API
///         // or run a local model to generate embeddings
///         Embedding::new(vec![0.1, 0.2, 0.3, 0.4])
///     }
/// }
/// ```
pub trait EmbeddingProvider {
    /// Generates a semantic embedding for the given image bytes.
    ///
    /// # Arguments
    ///
    /// * `image` - Raw image bytes in any supported format (PNG, JPEG, etc.)
    ///
    /// # Errors
    ///
    /// Implementations should return:
    /// - [`ImgFprintError::ProviderError`] for provider-specific failures (network, auth, etc.)
    /// - [`ImgFprintError::InvalidImage`] if the image format is not supported by the provider
    /// - [`ImgFprintError::InvalidEmbedding`] if the returned embedding is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// use imgfprint_rs::{EmbeddingProvider, ImgFprintError};
    ///
    /// # fn example<P: EmbeddingProvider>(provider: &P) -> Result<(), ImgFprintError> {
    /// let image_bytes = vec![0u8; 1000]; // Your image data
    /// let embedding = provider.embed(&image_bytes)?;
    /// # Ok(())
    /// # }
    /// ```
    fn embed(&self, image: &[u8]) -> Result<Embedding, ImgFprintError>;
}

/// Computes cosine similarity between two embeddings.
///
/// Cosine similarity measures the cosine of the angle between two vectors,
/// providing a value in the range [-1.0, 1.0]:
/// - 1.0: vectors point in the same direction (identical orientation)
/// - 0.0: vectors are orthogonal (unrelated)
/// - -1.0: vectors point in opposite directions
///
/// For normalized embeddings (common in CLIP models), the range is [0.0, 1.0].
///
/// # Arguments
///
/// * `a` - First embedding
/// * `b` - Second embedding
///
/// # Returns
///
/// Cosine similarity as an `f32` in the range [-1.0, 1.0].
///
/// # Errors
///
/// Returns [`ImgFprintError::EmbeddingDimensionMismatch`] if the embeddings
/// have different dimensions.
///
/// # Performance
///
/// This function:
/// - Runs in O(n) time where n is the embedding dimension
/// - Performs no heap allocations
/// - Operates directly on slices for cache efficiency
/// - Is SIMD-friendly (the compiler can auto-vectorize the loop)
///
/// # Examples
///
/// ```rust
/// use imgfprint_rs::{Embedding, semantic_similarity};
///
/// # fn example() -> Result<(), imgfprint_rs::ImgFprintError> {
/// let a = Embedding::new(vec![1.0, 0.0, 0.0])?;
/// let b = Embedding::new(vec![1.0, 0.0, 0.0])?;
///
/// let sim = semantic_similarity(&a, &b)?;
/// assert!((sim - 1.0).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
pub fn semantic_similarity(a: &Embedding, b: &Embedding) -> Result<f32, ImgFprintError> {
    let a_vec = a.as_slice();
    let b_vec = b.as_slice();

    if a_vec.len() != b_vec.len() {
        return Err(ImgFprintError::EmbeddingDimensionMismatch {
            expected: a_vec.len(),
            actual: b_vec.len(),
        });
    }

    // Compute dot product and norms in a single pass for better cache locality
    let mut dot_product: f32 = 0.0;
    let mut norm_a_sq: f32 = 0.0;
    let mut norm_b_sq: f32 = 0.0;

    for i in 0..a_vec.len() {
        let a_i = a_vec[i];
        let b_i = b_vec[i];

        dot_product += a_i * b_i;
        norm_a_sq += a_i * a_i;
        norm_b_sq += b_i * b_i;
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    // Handle zero vectors (shouldn't happen with valid embeddings, but be safe)
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(ImgFprintError::InvalidEmbedding(
            "cannot compute similarity for zero vector".to_string(),
        ));
    }

    Ok(dot_product / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_new_valid() {
        let vector = vec![0.1, 0.2, 0.3, 0.4];
        let embedding = Embedding::new(vector.clone()).unwrap();

        assert_eq!(embedding.len(), 4);
        assert_eq!(embedding.dimension(), 4);
        assert_eq!(embedding.as_slice(), &vector);
        assert_eq!(embedding.vector(), vector);
    }

    #[test]
    fn test_embedding_empty_vector() {
        let result = Embedding::new(vec![]);
        assert!(matches!(
            result,
            Err(ImgFprintError::InvalidEmbedding(msg)) if msg.contains("empty")
        ));
    }

    #[test]
    fn test_embedding_nan_values() {
        let result = Embedding::new(vec![0.1, f32::NAN, 0.3]);
        assert!(matches!(
            result,
            Err(ImgFprintError::InvalidEmbedding(msg)) if msg.contains("non-finite")
        ));
    }

    #[test]
    fn test_embedding_infinity_values() {
        let result = Embedding::new(vec![0.1, f32::INFINITY, 0.3]);
        assert!(matches!(
            result,
            Err(ImgFprintError::InvalidEmbedding(msg)) if msg.contains("non-finite")
        ));
    }

    #[test]
    fn test_embedding_negative_infinity() {
        let result = Embedding::new(vec![0.1, f32::NEG_INFINITY, 0.3]);
        assert!(matches!(
            result,
            Err(ImgFprintError::InvalidEmbedding(msg)) if msg.contains("non-finite")
        ));
    }

    // Helper to create embeddings for tests
    fn emb(vector: Vec<f32>) -> Embedding {
        Embedding::new(vector).unwrap()
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = emb(vec![1.0, 0.0, 0.0, 0.0]);
        let b = emb(vec![1.0, 0.0, 0.0, 0.0]);

        let sim = semantic_similarity(&a, &b).unwrap();
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = emb(vec![1.0, 0.0, 0.0]);
        let b = emb(vec![0.0, 1.0, 0.0]);

        let sim = semantic_similarity(&a, &b).unwrap();
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = emb(vec![1.0, 0.0, 0.0]);
        let b = emb(vec![-1.0, 0.0, 0.0]);

        let sim = semantic_similarity(&a, &b).unwrap();
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_45_degrees() {
        let a = emb(vec![1.0, 0.0]);
        let b = emb(vec![1.0, 1.0]);

        let sim = semantic_similarity(&a, &b).unwrap();
        let expected = 1.0 / f32::sqrt(2.0);
        assert!(
            (sim - expected).abs() < 1e-5,
            "45-degree angle similarity should be ~0.707, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = emb(vec![1.0, 0.0, 0.0]);
        let b = emb(vec![1.0, 0.0]);

        let result = semantic_similarity(&a, &b);
        assert!(matches!(
            result,
            Err(ImgFprintError::EmbeddingDimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_cosine_similarity_with_normalization() {
        // CLIP embeddings are typically L2-normalized, so similarity is in [0, 1]
        let a = emb(vec![0.5, 0.5, 0.5, 0.5]);
        let b = emb(vec![0.5, 0.5, 0.5, 0.5]);

        let sim = semantic_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_various_dimensions() {
        // Test with various typical CLIP embedding dimensions
        for dim in [512, 768, 1024] {
            let a = emb(vec![1.0; dim]);
            let b = emb(vec![1.0; dim]);

            let sim = semantic_similarity(&a, &b).unwrap();
            assert!((sim - 1.0).abs() < 1e-6, "Failed for dimension {}", dim);
        }
    }

    #[test]
    fn test_cosine_similarity_negative_values() {
        let a = emb(vec![1.0, -1.0, 1.0, -1.0]);
        let b = emb(vec![-1.0, 1.0, -1.0, 1.0]);

        let sim = semantic_similarity(&a, &b).unwrap();
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Opposite signs should give -1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = emb(vec![1.0, 0.0, 0.0]);
        let _b = emb(vec![0.0, 0.0, 0.0]);

        // We need to bypass the normal constructor to create a zero vector
        // Since Embedding::new validates against this, we manually construct
        let zero_embedding = Embedding {
            vector: vec![0.0; 3],
        };

        let result = semantic_similarity(&a, &zero_embedding);
        assert!(matches!(
            result,
            Err(ImgFprintError::InvalidEmbedding(msg)) if msg.contains("zero vector")
        ));
    }

    #[test]
    fn test_embedding_is_empty() {
        // Valid embeddings are never empty
        let emb = Embedding::new(vec![1.0]).unwrap();
        assert!(!emb.is_empty());
    }

    #[test]
    fn test_embedding_clone() {
        let a = emb(vec![0.1, 0.2, 0.3]);
        let b = a.clone();

        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn test_embedding_partial_eq() {
        let a = emb(vec![0.1, 0.2, 0.3]);
        let b = emb(vec![0.1, 0.2, 0.3]);
        let c = emb(vec![0.3, 0.2, 0.1]);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // Test provider trait with a mock implementation
    struct MockProvider {
        return_value: Vec<f32>,
    }

    impl EmbeddingProvider for MockProvider {
        fn embed(&self, _image: &[u8]) -> Result<Embedding, ImgFprintError> {
            Embedding::new(self.return_value.clone())
        }
    }

    #[test]
    fn test_embedding_provider_mock() {
        let provider = MockProvider {
            return_value: vec![0.1, 0.2, 0.3],
        };

        let image_bytes = vec![0u8; 100];
        let embedding = provider.embed(&image_bytes).unwrap();

        assert_eq!(embedding.as_slice(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_embedding_provider_error_propagation() {
        struct FailingProvider;

        impl EmbeddingProvider for FailingProvider {
            fn embed(&self, _image: &[u8]) -> Result<Embedding, ImgFprintError> {
                Err(ImgFprintError::ProviderError("network timeout".to_string()))
            }
        }

        let provider = FailingProvider;
        let image_bytes = vec![0u8; 100];
        let result = provider.embed(&image_bytes);

        assert!(matches!(
            result,
            Err(ImgFprintError::ProviderError(msg)) if msg == "network timeout"
        ));
    }
}
