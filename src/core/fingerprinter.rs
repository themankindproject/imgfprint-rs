use crate::{
    core::fingerprint::ImageFingerprint,
    core::similarity,
    error::ImgFprintError,
    hash::blocks::compute_block_hashes,
    hash::phash::compute_phash,
    imgproc::decode::decode_image,
    imgproc::preprocess::{extract_blocks, extract_global_region, normalize},
};
use sha2::{Digest, Sha256};

/// Static methods for computing and comparing image fingerprints.
///
/// This struct provides the primary API for the library. All methods are static
/// as the fingerprinting process is stateless.
pub struct ImageFingerprinter;

impl ImageFingerprinter {
    /// Computes a perceptual fingerprint for the given image bytes.
    ///
    /// The fingerprint includes:
    /// - SHA256 hash of the original bytes (exact matching)
    /// - Global perceptual hash of the center 32x32 region
    /// - 16 block-level hashes from a 4x4 grid (crop resistance)
    ///
    /// # Errors
    ///
    /// Returns `ImgFprintError` if:
    /// - The image data is empty
    /// - The format is unsupported
    /// - Decoding fails
    /// - Dimensions exceed 8192x8192 (OOM protection)
    pub fn fingerprint(image_bytes: &[u8]) -> Result<ImageFingerprint, ImgFprintError> {
        let exact_hash = compute_exact_hash(image_bytes);
        let image = decode_image(image_bytes)?;
        let normalized = normalize(image);

        let global_region = extract_global_region(&normalized);
        let global_phash = compute_phash(&global_region);

        let blocks = extract_blocks(&normalized);
        let block_hashes = compute_block_hashes(&blocks);

        Ok(ImageFingerprint::new(
            exact_hash,
            global_phash,
            block_hashes,
        ))
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
    pub fn fingerprint_batch<S: Clone>(
        images: &[(S, Vec<u8>)],
    ) -> Vec<(S, Result<ImageFingerprint, ImgFprintError>)>
    where
        S: Send + Sync + Clone + 'static,
    {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            images
                .par_iter()
                .map(|(id, bytes)| (id.clone(), Self::fingerprint(bytes)))
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

#[inline]
fn compute_exact_hash(image_bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(image_bytes);
    hasher.finalize().into()
}
