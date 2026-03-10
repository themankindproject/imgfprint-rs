use crate::core::fingerprint::ImageFingerprint;
use crate::core::similarity;
use crate::error::ImgFprintError;
use crate::hash::blocks::compute_block_hashes;
use crate::hash::phash::compute_phash;
use crate::imgproc::decode::decode_image;
use crate::imgproc::preprocess::{extract_blocks, extract_global_region, Preprocessor};
use sha2::{Digest, Sha256};
use std::cell::RefCell;

/// Context for high-performance fingerprinting with buffer reuse.
///
/// Maintains a reusable preprocessor, hasher, and internal buffers
/// to minimize allocations in high-throughput scenarios.
///
/// # Example
/// ```
/// use imgfprint_rs::FingerprinterContext;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut ctx = FingerprinterContext::new();
/// let fp1 = ctx.fingerprint(&std::fs::read("img1.jpg")?)?;
/// let fp2 = ctx.fingerprint(&std::fs::read("img2.jpg")?)?;
/// # Ok(())
/// # }
/// ```
pub struct FingerprinterContext {
    preprocessor: Preprocessor,
    sha_hasher: Sha256,
}

impl Default for FingerprinterContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprinterContext {
    /// Creates a new fingerprinter context with cached resources.
    pub fn new() -> Self {
        Self {
            preprocessor: Preprocessor::new(),
            sha_hasher: Sha256::new(),
        }
    }

    /// Computes a perceptual fingerprint for the given image bytes.
    ///
    /// Uses internal buffer reuse for optimal performance.
    ///
    /// # Errors
    ///
    /// Returns `ImgFprintError` if:
    /// - The image data is empty
    /// - The format is unsupported
    /// - Decoding fails
    /// - Dimensions are outside valid range (32x32 to 8192x8192)
    /// - Processing fails (resize, conversion errors)
    pub fn fingerprint(&mut self, image_bytes: &[u8]) -> Result<ImageFingerprint, ImgFprintError> {
        // Compute exact hash with reusable hasher
        self.sha_hasher.reset();
        self.sha_hasher.update(image_bytes);
        let exact_hash: [u8; 32] = self.sha_hasher.finalize_reset().into();

        // Decode image
        let image = decode_image(image_bytes)?;

        // Normalize with cached preprocessor (includes SIMD-accelerated resize)
        let normalized = self.preprocessor.normalize(&image)?;

        // Extract global region and compute pHash
        let global_region = extract_global_region(&normalized);
        let global_phash = compute_phash(&global_region);

        // Extract blocks and compute block hashes
        let blocks = extract_blocks(&normalized);
        let block_hashes = compute_block_hashes(&blocks);

        Ok(ImageFingerprint::new(
            exact_hash,
            global_phash,
            block_hashes,
        ))
    }
}

/// Static methods for computing and comparing image fingerprints.
///
/// This struct provides the primary API for the library. Methods are static
/// for convenience, but internally use thread-local state for performance
/// (cached resizers, DCT plans, and hashers).
///
/// For high-throughput scenarios, use [`FingerprinterContext`] instead
/// to enable buffer reuse and avoid repeated allocations.
pub struct ImageFingerprinter;

impl ImageFingerprinter {
    /// Computes a perceptual fingerprint for the given image bytes.
    ///
    /// The fingerprint includes:
    /// - SHA256 hash of the original bytes (exact matching)
    /// - Global perceptual hash of the center 32x32 region
    /// - 16 block-level hashes from a 4x4 grid (crop resistance)
    ///
    /// Uses a thread-local context for buffer reuse when called
    /// multiple times from the same thread.
    ///
    /// # Errors
    ///
    /// Returns `ImgFprintError` if:
    /// - The image data is empty
    /// - The format is unsupported
    /// - Decoding fails
    /// - Dimensions exceed 8192x8192 (OOM protection)
    pub fn fingerprint(image_bytes: &[u8]) -> Result<ImageFingerprint, ImgFprintError> {
        // Use thread-local context for buffer reuse
        thread_local! {
            static CTX: RefCell<FingerprinterContext> = RefCell::new(FingerprinterContext::new());
        }

        CTX.with(|ctx| ctx.borrow_mut().fingerprint(image_bytes))
    }

    /// Compares two fingerprints and returns a similarity score.
    ///
    /// The score is a weighted combination (40% global, 60% blocks) of
    /// Hamming distances between the perceptual hashes.
    ///
    /// # Returns
    ///
    /// A `Similarity` struct containing:
    /// - `score`: 0.0 (completely different) to 1.0 (identical)
    /// - `exact_match`: true if SHA256 hashes match
    /// - `perceptual_distance`: Hamming distance between global hashes (0-64)
    pub fn compare(a: &ImageFingerprint, b: &ImageFingerprint) -> similarity::Similarity {
        similarity::compute_similarity(a, b)
    }

    /// Generates a semantic embedding for the given image using an external provider.
    ///
    /// This method delegates to an [`EmbeddingProvider`](crate::embed::EmbeddingProvider) implementation to generate
    /// CLIP-style embeddings that capture semantic content of the image. The SDK
    /// does not implement any specific provider; users must bring their own
    /// implementation (e.g., OpenAI CLIP API, HuggingFace, local models).
    ///
    /// # Type Parameters
    ///
    /// * `P` - An implementation of [`EmbeddingProvider`](crate::embed::EmbeddingProvider)
    ///
    /// # Arguments
    ///
    /// * `provider` - The embedding provider implementation
    /// * `image` - Raw image bytes in any supported format (PNG, JPEG, WebP, etc.)
    ///
    /// # Returns
    ///
    /// An [`Embedding`](crate::Embedding) containing the semantic vector representation.
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError`] if:
    /// - The provider fails to generate an embedding
    /// - The returned embedding is invalid (empty, contains NaN, etc.)
    ///
    /// # Example
    ///
    /// ```rust
    /// use imgfprint_rs::{ImageFingerprinter, EmbeddingProvider, Embedding, ImgFprintError};
    ///
    /// // Example provider implementation
    /// struct MyProvider;
    ///
    /// impl EmbeddingProvider for MyProvider {
    ///     fn embed(&self, _image: &[u8]) -> Result<Embedding, ImgFprintError> {
    ///         // In practice, call your embedding API here
    ///         Embedding::new(vec![0.1, 0.2, 0.3, 0.4])
    ///     }
    /// }
    ///
    /// # fn example() -> Result<(), ImgFprintError> {
    /// let provider = MyProvider;
    /// let image_bytes = vec![0u8; 1000]; // Your image data
    /// let embedding = ImageFingerprinter::semantic_embedding(&provider, &image_bytes)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn semantic_embedding<P: crate::embed::EmbeddingProvider>(
        provider: &P,
        image: &[u8],
    ) -> Result<crate::embed::Embedding, ImgFprintError> {
        provider.embed(image)
    }

    /// Compares two semantic embeddings using cosine similarity.
    ///
    /// This is a convenience wrapper around [`semantic_similarity()`][crate::semantic_similarity]
    /// that provides a consistent API with the rest of the `ImageFingerprinter` methods.
    ///
    /// Cosine similarity measures the angle between two embedding vectors, returning
    /// a value in the range [-1.0, 1.0]:
    /// - 1.0: embeddings point in the same direction (semantically similar)
    /// - 0.0: embeddings are orthogonal (semantically unrelated)
    /// - -1.0: embeddings point in opposite directions (semantically opposite)
    ///
    /// For typical CLIP embeddings (L2-normalized), the range is [0.0, 1.0].
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
    /// - O(n) time complexity where n is the embedding dimension
    /// - Zero heap allocations
    /// - SIMD-friendly loop structure
    ///
    /// # Example
    ///
    /// ```rust
    /// use imgfprint_rs::{ImageFingerprinter, EmbeddingProvider, Embedding, ImgFprintError};
    ///
    /// struct MyProvider;
    ///
    /// impl EmbeddingProvider for MyProvider {
    ///     fn embed(&self, _image: &[u8]) -> Result<Embedding, ImgFprintError> {
    ///         Embedding::new(vec![0.1, 0.2, 0.3, 0.4])
    ///     }
    /// }
    ///
    /// # fn example() -> Result<(), ImgFprintError> {
    /// let provider = MyProvider;
    /// let img1 = vec![0u8; 1000];
    /// let img2 = vec![0u8; 1000];
    ///
    /// let emb1 = ImageFingerprinter::semantic_embedding(&provider, &img1)?;
    /// let emb2 = ImageFingerprinter::semantic_embedding(&provider, &img2)?;
    ///
    /// let similarity = ImageFingerprinter::semantic_similarity(&emb1, &emb2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn semantic_similarity(
        a: &crate::embed::Embedding,
        b: &crate::embed::Embedding,
    ) -> Result<f32, ImgFprintError> {
        crate::embed::semantic_similarity(a, b)
    }

    /// Computes fingerprints for multiple images in batch.
    ///
    /// Processes each image independently and returns results in the same order.
    /// When the `parallel` feature is enabled, images are processed in parallel.
    ///
    /// # Arguments
    /// * `images` - Slice of (id, image_bytes) tuples
    ///
    /// # Returns
    /// Vec of (id, Result<ImageFingerprint, ImgFprintError>) pairs
    ///
    /// # Example
    /// ```
    /// # use imgfprint_rs::ImageFingerprinter;
    /// let images = vec![
    ///     ("img1".to_string(), vec![0u8; 100]),
    ///     ("img2".to_string(), vec![0u8; 100]),
    /// ];
    /// let results = ImageFingerprinter::fingerprint_batch(&images);
    /// ```
    pub fn fingerprint_batch<S>(
        images: &[(S, Vec<u8>)],
    ) -> Vec<(S, Result<ImageFingerprint, ImgFprintError>)>
    where
        S: Send + Sync + Clone + 'static,
    {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // In parallel mode, create a new context per thread to avoid RefCell conflicts
            images
                .par_iter()
                .map(|(id, bytes)| {
                    let mut ctx = FingerprinterContext::new();
                    (id.clone(), ctx.fingerprint(bytes))
                })
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            images
                .iter()
                .map(|(id, bytes)| (id.clone(), Self::fingerprint(bytes)))
                .collect()
        }
    }
}
