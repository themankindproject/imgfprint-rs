use crate::core::fingerprint::{ImageFingerprint, MultiHashFingerprint};
use crate::core::similarity;
use crate::error::ImgFprintError;
use crate::hash::ahash::{compute_ahash, compute_ahash_from_64x64};
use crate::hash::algorithms::HashAlgorithm;
use crate::hash::dhash::{compute_dhash, compute_dhash_from_64x64};
use crate::hash::phash::{compute_phash, compute_phash_from_64x64};
use crate::imgproc::decode::decode_image;
use crate::imgproc::preprocess::{extract_blocks, extract_global_region, Preprocessor};
use blake3::Hasher;
use std::cell::RefCell;

#[cfg(feature = "tracing")]
use tracing::{debug, instrument};

/// Context for high-performance fingerprinting with buffer reuse.
///
/// Maintains a reusable preprocessor, hasher, and internal buffers
/// to minimize allocations in high-throughput scenarios.
#[derive(Debug)]
pub struct FingerprinterContext {
    preprocessor: Preprocessor,
    exact_hasher: Hasher,
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
            exact_hasher: Hasher::new(),
        }
    }

    /// Computes all perceptual hashes in parallel.
    ///
    /// Calculates both PHash and DHash simultaneously for improved accuracy.
    /// Returns a MultiHashFingerprint containing all hash layers.
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes), fields(size = image_bytes.len())))]
    pub fn fingerprint(
        &mut self,
        image_bytes: &[u8],
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();
        let result = self.compute_all_hashes(image_bytes);
        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            "fingerprint completed"
        );
        result
    }

    /// Computes a single perceptual hash using the specified algorithm.
    ///
    /// More efficient than computing all hashes when only one algorithm is needed.
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes), fields(size = image_bytes.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_with(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();
        let result = self.compute_single_hash(image_bytes, algorithm);
        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            "fingerprint_with completed"
        );
        result
    }

    fn compute_all_hashes(
        &mut self,
        image_bytes: &[u8],
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        self.exact_hasher.reset();
        self.exact_hasher.update(image_bytes);
        let exact_hash: [u8; 32] = *self.exact_hasher.finalize().as_bytes();

        let image = decode_image(image_bytes)?;
        let normalized = self.preprocessor.normalize(&image)?;

        let global_region = extract_global_region(&normalized);
        let blocks = extract_blocks(&normalized);

        #[cfg(feature = "parallel")]
        let (ahash_fp, phash_fp, dhash_fp) = {
            let (ahash_result, (phash_result, dhash_result)) = rayon::join(
                || Self::compute_ahash_data(&global_region, &blocks),
                || {
                    rayon::join(
                        || Self::compute_phash_data(&global_region, &blocks),
                        || Self::compute_dhash_data(&global_region, &blocks),
                    )
                },
            );

            let (global_ahash, block_ahashes) = ahash_result;
            let (global_hash, block_hashes) = phash_result;
            let (global_dhash, block_dhashes) = dhash_result;

            (
                ImageFingerprint::new(exact_hash, global_ahash, block_ahashes),
                ImageFingerprint::new(exact_hash, global_hash, block_hashes),
                ImageFingerprint::new(exact_hash, global_dhash, block_dhashes),
            )
        };

        #[cfg(not(feature = "parallel"))]
        let (ahash_fp, phash_fp, dhash_fp) = {
            let (global_ahash, block_ahashes) = Self::compute_ahash_data(&global_region, &blocks);
            let (global_hash, block_hashes) = Self::compute_phash_data(&global_region, &blocks);
            let (global_dhash, block_dhashes) = Self::compute_dhash_data(&global_region, &blocks);

            (
                ImageFingerprint::new(exact_hash, global_ahash, block_ahashes),
                ImageFingerprint::new(exact_hash, global_hash, block_hashes),
                ImageFingerprint::new(exact_hash, global_dhash, block_dhashes),
            )
        };

        Ok(MultiHashFingerprint::new(
            exact_hash, ahash_fp, phash_fp, dhash_fp,
        ))
    }

    fn compute_single_hash(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        self.exact_hasher.reset();
        self.exact_hasher.update(image_bytes);
        let exact_hash: [u8; 32] = *self.exact_hasher.finalize().as_bytes();

        let image = decode_image(image_bytes)?;
        let normalized = self.preprocessor.normalize(&image)?;

        let global_region = extract_global_region(&normalized);
        let blocks = extract_blocks(&normalized);

        let (global_hash, block_hashes) = match algorithm {
            HashAlgorithm::AHash => Self::compute_ahash_data(&global_region, &blocks),
            HashAlgorithm::PHash => Self::compute_phash_data(&global_region, &blocks),
            HashAlgorithm::DHash => Self::compute_dhash_data(&global_region, &blocks),
        };

        Ok(ImageFingerprint::new(exact_hash, global_hash, block_hashes))
    }

    fn compute_phash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
    ) -> (u64, [u64; 16]) {
        let global_hash = compute_phash(global_region);

        #[cfg(feature = "parallel")]
        let block_hashes = {
            use rayon::prelude::*;
            let mut hashes = [0u64; 16];
            hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
                *hash = compute_phash_from_64x64(&blocks[i]);
            });
            hashes
        };

        #[cfg(not(feature = "parallel"))]
        let block_hashes = {
            let mut hashes = [0u64; 16];
            for (i, block) in blocks.iter().enumerate() {
                hashes[i] = compute_phash_from_64x64(block);
            }
            hashes
        };

        (global_hash, block_hashes)
    }

    fn compute_ahash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
    ) -> (u64, [u64; 16]) {
        let global_hash = compute_ahash(global_region);

        #[cfg(feature = "parallel")]
        let block_hashes = {
            use rayon::prelude::*;
            let mut hashes = [0u64; 16];
            hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
                *hash = compute_ahash_from_64x64(&blocks[i]);
            });
            hashes
        };

        #[cfg(not(feature = "parallel"))]
        let block_hashes = {
            let mut hashes = [0u64; 16];
            for (i, block) in blocks.iter().enumerate() {
                hashes[i] = compute_ahash_from_64x64(block);
            }
            hashes
        };

        (global_hash, block_hashes)
    }

    fn compute_dhash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
    ) -> (u64, [u64; 16]) {
        let global_dhash = compute_dhash(global_region);

        #[cfg(feature = "parallel")]
        let block_hashes = {
            use rayon::prelude::*;
            let mut hashes = [0u64; 16];
            hashes.par_iter_mut().enumerate().for_each(|(i, hash)| {
                *hash = compute_dhash_from_64x64(&blocks[i]);
            });
            hashes
        };

        #[cfg(not(feature = "parallel"))]
        let block_hashes = {
            let mut hashes = [0u64; 16];
            for (i, block) in blocks.iter().enumerate() {
                hashes[i] = compute_dhash_from_64x64(block);
            }
            hashes
        };

        (global_dhash, block_hashes)
    }

    /// Computes fingerprints for multiple images in chunks to limit memory usage.
    ///
    /// Processes images in chunks of `chunk_size` and invokes the callback
    /// for each result. This prevents unbounded memory consumption when
    /// processing large batches.
    #[cfg_attr(feature = "tracing", instrument(skip(self, images, callback), fields(chunk_size, image_count = images.len())))]
    pub fn fingerprint_batch_chunked<S, F>(
        &mut self,
        images: &[(S, Vec<u8>)],
        chunk_size: usize,
        mut callback: F,
    ) where
        S: Send + Sync + Clone + 'static,
        F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>),
    {
        let chunk_size = chunk_size.max(1);

        #[cfg(feature = "tracing")]
        tracing::debug!(
            chunk_size,
            image_count = images.len(),
            "starting chunked batch processing"
        );

        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        #[cfg(feature = "tracing")]
        let mut processed = 0usize;
        #[cfg(feature = "tracing")]
        let mut failed = 0usize;

        for chunk in images.chunks(chunk_size) {
            for (id, bytes) in chunk {
                let result = self.fingerprint(bytes);
                #[cfg(feature = "tracing")]
                {
                    if result.is_err() {
                        failed += 1;
                    }
                    processed += 1;
                }
                callback(id.clone(), result);
            }
        }

        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            processed, failed, "batch processing completed"
        );
    }
}

/// Static methods for computing and comparing image fingerprints.
///
/// Provides both single-algorithm and multi-algorithm fingerprinting.
/// Multi-algorithm mode (default) computes PHash and DHash in parallel
/// for improved accuracy through weighted combination.
pub struct ImageFingerprinter;

impl ImageFingerprinter {
    /// Computes all perceptual hashes in parallel.
    ///
    /// Calculates both PHash and DHash simultaneously and returns a
    /// MultiHashFingerprint containing both hash layers. This provides
    /// superior accuracy compared to single-algorithm fingerprinting.
    ///
    /// # Errors
    ///
    /// Returns `ImgFprintError` if any algorithm fails.
    #[cfg_attr(feature = "tracing", instrument(skip(image_bytes), fields(size = image_bytes.len())))]
    pub fn fingerprint(image_bytes: &[u8]) -> Result<MultiHashFingerprint, ImgFprintError> {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        thread_local! {
            static CTX: RefCell<FingerprinterContext> = RefCell::new(FingerprinterContext::new());
        }

        let result = CTX.with(|ctx| ctx.borrow_mut().fingerprint(image_bytes));

        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            "ImageFingerprinter::fingerprint completed"
        );

        result
    }

    /// Computes a single perceptual hash using the specified algorithm.
    ///
    /// Use this when you need a specific algorithm or want to minimize
    /// computation for high-throughput scenarios.
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image data
    /// * `algorithm` - Hash algorithm to use (PHash or DHash)
    #[cfg_attr(feature = "tracing", instrument(skip(image_bytes), fields(size = image_bytes.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_with(
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        thread_local! {
            static CTX: RefCell<FingerprinterContext> = RefCell::new(FingerprinterContext::new());
        }

        let result = CTX.with(|ctx| ctx.borrow_mut().fingerprint_with(image_bytes, algorithm));

        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            "ImageFingerprinter::fingerprint_with completed"
        );

        result
    }

    /// Compares two fingerprints and returns a similarity score.
    ///
    /// For MultiHashFingerprint, use the compare() method directly.
    /// For ImageFingerprint, this computes similarity using the global hash.
    #[must_use]
    pub fn compare(a: &ImageFingerprint, b: &ImageFingerprint) -> similarity::Similarity {
        similarity::compute_similarity(a, b)
    }

    /// Generates a semantic embedding for the given image using an external provider.
    pub fn semantic_embedding<P: crate::embed::EmbeddingProvider>(
        provider: &P,
        image: &[u8],
    ) -> Result<crate::embed::Embedding, ImgFprintError> {
        provider.embed(image)
    }

    /// Compares two semantic embeddings using cosine similarity.
    pub fn semantic_similarity(
        a: &crate::embed::Embedding,
        b: &crate::embed::Embedding,
    ) -> Result<f32, ImgFprintError> {
        crate::embed::semantic_similarity(a, b)
    }

    /// Computes fingerprints for multiple images in batch.
    ///
    /// Processes each image independently and returns results in the same order.
    #[cfg_attr(feature = "tracing", instrument(skip(images), fields(image_count = images.len())))]
    pub fn fingerprint_batch<S>(
        images: &[(S, Vec<u8>)],
    ) -> Vec<(S, Result<MultiHashFingerprint, ImgFprintError>)>
    where
        S: Send + Sync + Clone + 'static,
    {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let results = images
                .par_iter()
                .map(|(id, bytes)| {
                    let mut ctx = FingerprinterContext::new();
                    (id.clone(), ctx.fingerprint(bytes))
                })
                .collect();

            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                "parallel batch completed"
            );

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            let results = images
                .iter()
                .map(|(id, bytes)| (id.clone(), Self::fingerprint(bytes)))
                .collect();

            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                "sequential batch completed"
            );

            results
        }
    }

    /// Computes fingerprints with specific algorithm for multiple images.
    #[cfg_attr(feature = "tracing", instrument(skip(images), fields(image_count = images.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_batch_with<S>(
        images: &[(S, Vec<u8>)],
        algorithm: HashAlgorithm,
    ) -> Vec<(S, Result<ImageFingerprint, ImgFprintError>)>
    where
        S: Send + Sync + Clone + 'static,
    {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let results = images
                .par_iter()
                .map(|(id, bytes)| {
                    let mut ctx = FingerprinterContext::new();
                    (id.clone(), ctx.fingerprint_with(bytes, algorithm))
                })
                .collect();

            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                "parallel batch_with completed"
            );

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            let results = images
                .iter()
                .map(|(id, bytes)| (id.clone(), Self::fingerprint_with(bytes, algorithm)))
                .collect();

            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                "sequential batch_with completed"
            );

            results
        }
    }

    /// Computes fingerprints for multiple images in chunks to limit memory usage.
    ///
    /// Processes images in chunks of `chunk_size` and invokes the callback
    /// for each result. This prevents unbounded memory consumption when
    /// processing large batches.
    ///
    /// # Arguments
    /// * `images` - Slice of (id, image_bytes) pairs
    /// * `chunk_size` - Number of images to process per chunk
    /// * `callback` - Function called with each result as (id, Result<...>)
    #[cfg_attr(feature = "tracing", instrument(skip(images, callback), fields(chunk_size, image_count = images.len())))]
    pub fn fingerprint_batch_chunked<S, F>(
        images: &[(S, Vec<u8>)],
        chunk_size: usize,
        mut callback: F,
    ) where
        S: Send + Sync + Clone + 'static,
        F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>),
    {
        let chunk_size = chunk_size.max(1);

        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        #[cfg(feature = "tracing")]
        tracing::debug!(
            chunk_size,
            image_count = images.len(),
            "starting static batch_chunked"
        );

        #[cfg(feature = "tracing")]
        let mut processed = 0usize;
        #[cfg(feature = "tracing")]
        let mut failed = 0usize;

        for chunk in images.chunks(chunk_size) {
            for (id, bytes) in chunk {
                let result = Self::fingerprint(bytes);
                #[cfg(feature = "tracing")]
                {
                    if result.is_err() {
                        failed += 1;
                    }
                    processed += 1;
                }
                callback(id.clone(), result);
            }
        }

        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            processed, failed, "static batch_chunked completed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_image(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(width, height, |x, y| Rgb([(x % 256) as u8, (y % 256) as u8, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn test_fingerprinter_context_new() {
        let ctx = FingerprinterContext::new();
        let _ = ctx;
    }

    #[test]
    fn test_fingerprinter_context_default() {
        let ctx = FingerprinterContext::default();
        let _ = ctx;
    }

    #[test]
    fn test_fingerprinter_context_single_image() {
        let mut ctx = FingerprinterContext::new();
        let img = create_test_image(100, 100);
        let result = ctx.fingerprint(&img);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fingerprinter_context_determinism() {
        let mut ctx = FingerprinterContext::new();
        let img = create_test_image(100, 100);
        
        let fp1 = ctx.fingerprint(&img).unwrap();
        let fp2 = ctx.fingerprint(&img).unwrap();
        
        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
    }

    #[test]
    fn test_fingerprinter_context_fingerprint_with() {
        let mut ctx = FingerprinterContext::new();
        let img = create_test_image(100, 100);
        
        let result = ctx.fingerprint_with(&img, HashAlgorithm::PHash);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fingerprinter_batch_empty() {
        let images: Vec<(usize, Vec<u8>)> = vec![];
        let results = ImageFingerprinter::fingerprint_batch(&images);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fingerprinter_batch_single_image() {
        let img = create_test_image(100, 100);
        let images = vec![(0, img)];
        let results = ImageFingerprinter::fingerprint_batch(&images);
        
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_ok());
    }

    #[test]
    fn test_fingerprinter_batch_multiple_images() {
        let images: Vec<(usize, Vec<u8>)> = (0..5usize)
            .map(|i| (i, create_test_image(100 + i as u32 * 10, 100 + i as u32 * 10)))
            .collect();
        
        let results = ImageFingerprinter::fingerprint_batch(&images);
        
        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.0, i);
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_determinism() {
        let img = create_test_image(100, 100);
        let images = vec![(0, img.clone()), (1, img.clone())];
        
        let results1 = ImageFingerprinter::fingerprint_batch(&images);
        let results2 = ImageFingerprinter::fingerprint_batch(&images);
        
        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            let fp1 = r1.1.as_ref().unwrap();
            let fp2 = r2.1.as_ref().unwrap();
            assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        }
    }

    #[test]
    fn test_fingerprinter_batch_with_empty() {
        let images: Vec<(usize, Vec<u8>)> = vec![];
        let results = ImageFingerprinter::fingerprint_batch_with(&images, HashAlgorithm::PHash);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fingerprinter_batch_with_phash() {
        let images: Vec<(usize, Vec<u8>)> = (0..3usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        
        let results = ImageFingerprinter::fingerprint_batch_with(&images, HashAlgorithm::PHash);
        
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_with_dhash() {
        let images: Vec<(usize, Vec<u8>)> = (0..3usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        
        let results = ImageFingerprinter::fingerprint_batch_with(&images, HashAlgorithm::DHash);
        
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_with_ahash() {
        let images: Vec<(usize, Vec<u8>)> = (0..3usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        
        let results = ImageFingerprinter::fingerprint_batch_with(&images, HashAlgorithm::AHash);
        
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_chunked_empty() {
        let images: Vec<(usize, Vec<u8>)> = vec![];
        let mut results = Vec::new();
        
        ImageFingerprinter::fingerprint_batch_chunked(&images, 2, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_fingerprinter_batch_chunked_single() {
        let img = create_test_image(100, 100);
        let images = vec![(0, img)];
        let mut results = Vec::new();
        
        ImageFingerprinter::fingerprint_batch_chunked(&images, 2, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_ok());
    }

    #[test]
    fn test_fingerprinter_batch_chunked_multiple() {
        let images: Vec<(usize, Vec<u8>)> = (0..10usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        let mut results = Vec::new();
        
        ImageFingerprinter::fingerprint_batch_chunked(&images, 3, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.0, i);
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_chunked_chunk_size_one() {
        let images: Vec<(usize, Vec<u8>)> = (0..5usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        let mut results = Vec::new();
        
        ImageFingerprinter::fingerprint_batch_chunked(&images, 1, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_fingerprinter_batch_chunked_chunk_size_zero() {
        let images: Vec<(usize, Vec<u8>)> = (0..5usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        let mut results = Vec::new();
        
        ImageFingerprinter::fingerprint_batch_chunked(&images, 0, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_fingerprinter_batch_chunked_large_chunk_size() {
        let images: Vec<(usize, Vec<u8>)> = (0..5usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        let mut results = Vec::new();
        
        ImageFingerprinter::fingerprint_batch_chunked(&images, 100, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_fingerprinter_context_batch_chunked() {
        let mut ctx = FingerprinterContext::new();
        let images: Vec<(usize, Vec<u8>)> = (0..5usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        let mut results = Vec::new();
        
        ctx.fingerprint_batch_chunked(&images, 2, |id, result| {
            results.push((id, result));
        });
        
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_with_mixed_sizes() {
        let sizes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)];
        let images: Vec<(usize, Vec<u8>)> = sizes
            .iter()
            .enumerate()
            .map(|(i, &(w, h))| (i, create_test_image(w, h)))
            .collect();
        
        let results = ImageFingerprinter::fingerprint_batch(&images);
        
        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.1.is_ok());
        }
    }

    #[test]
    fn test_fingerprinter_batch_error_handling() {
        let mut images: Vec<(usize, Vec<u8>)> = (0..3usize)
            .map(|i| (i, create_test_image(100, 100)))
            .collect();
        images.push((3, vec![]));
        
        let results = ImageFingerprinter::fingerprint_batch(&images);
        
        assert_eq!(results.len(), 4);
        assert!(results[0].1.is_ok());
        assert!(results[1].1.is_ok());
        assert!(results[2].1.is_ok());
        assert!(results[3].1.is_err());
    }

    #[test]
    fn test_fingerprinter_static_methods() {
        let img = create_test_image(100, 100);
        
        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();
        
        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
    }

    #[test]
    fn test_fingerprinter_compare_static() {
        let img1 = create_test_image(100, 100);
        let img2 = create_test_image(100, 100);
        
        let fp1 = ImageFingerprinter::fingerprint_with(&img1, HashAlgorithm::PHash).unwrap();
        let fp2 = ImageFingerprinter::fingerprint_with(&img2, HashAlgorithm::PHash).unwrap();
        
        let sim = ImageFingerprinter::compare(&fp1, &fp2);
        assert!(sim.score >= 0.0 && sim.score <= 1.0);
    }
}
