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
    /// - Dimensions exceed 8192x8192 (OOM protection)
    pub fn fingerprint(&mut self, image_bytes: &[u8]) -> Result<ImageFingerprint, ImgFprintError> {
        // Compute exact hash with reusable hasher
        self.sha_hasher.reset();
        self.sha_hasher.update(image_bytes);
        let exact_hash: [u8; 32] = self.sha_hasher.finalize_reset().into();

        // Decode image
        let image = decode_image(image_bytes)?;

        // Normalize with cached preprocessor (includes SIMD-accelerated resize)
        let normalized = self.preprocessor.normalize(&image);

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
/// This struct provides the primary API for the library. All methods are static
/// as the fingerprinting process is stateless.
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
