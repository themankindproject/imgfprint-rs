use crate::core::fingerprint::{ImageFingerprint, MultiHashFingerprint};
use crate::core::similarity;
use crate::error::ImgFprintError;
use crate::hash::ahash::{compute_ahash, compute_ahash_from_64x64};
use crate::hash::algorithms::HashAlgorithm;
use crate::hash::dhash::{compute_dhash, compute_dhash_from_64x64};
use crate::hash::phash::{
    compute_phash_from_64x64_with_scratch, compute_phash_with_scratch, DctScratch,
};
use crate::imgproc::decode::{decode_image_with_config, PreprocessConfig};
use crate::imgproc::preprocess::{
    extract_blocks_from_raw, extract_global_region_from_raw, get_cpu_extensions, Preprocessor,
};
use blake3::Hasher;
use std::cell::RefCell;
use std::path::Path;
#[cfg(feature = "tracing")]
use std::time::Instant;

// Reads a file from disk into memory, rejecting inputs larger than the
// configured cap before any read happens. Keeps oversized files from being
// pulled into RAM just to be rejected by the decode pass.
fn read_image_file(path: &Path, config: &PreprocessConfig) -> Result<Vec<u8>, ImgFprintError> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > config.max_input_bytes as u64 {
        return Err(ImgFprintError::IoError(format!(
            "file size {} bytes exceeds maximum {} bytes",
            metadata.len(),
            config.max_input_bytes
        )));
    }
    std::fs::read(path).map_err(Into::into)
}

// Module-level shared thread-local context to avoid duplication
thread_local! {
    static SHARED_CTX: RefCell<FingerprinterContext> = RefCell::new(FingerprinterContext::new());
}

#[cfg(feature = "tracing")]
use tracing::{debug, instrument};

macro_rules! trace_stage {
    ($stage:literal, $body:block) => {{
        #[cfg(feature = "tracing")]
        let stage_start = Instant::now();
        let stage_result = $body;
        #[cfg(feature = "tracing")]
        debug!(
            stage = $stage,
            duration_us = stage_start.elapsed().as_micros(),
            "fingerprint stage completed"
        );
        stage_result
    }};
}

macro_rules! trace_result_stage {
    ($stage:literal, $body:block) => {{
        #[cfg(feature = "tracing")]
        let stage_start = Instant::now();
        let stage_result = $body;
        match stage_result {
            Ok(value) => {
                #[cfg(feature = "tracing")]
                debug!(
                    stage = $stage,
                    duration_us = stage_start.elapsed().as_micros(),
                    "fingerprint stage completed"
                );
                value
            }
            Err(error) => {
                #[cfg(feature = "tracing")]
                debug!(
                    stage = $stage,
                    duration_us = stage_start.elapsed().as_micros(),
                    error = ?error,
                    "fingerprint stage failed"
                );
                return Err(error);
            }
        }
    }};
}

#[cfg(feature = "tracing")]
fn count_results<S, T>(results: &[(S, Result<T, ImgFprintError>)]) -> (usize, usize) {
    let failed = results.iter().filter(|(_, result)| result.is_err()).count();
    (results.len() - failed, failed)
}

/// Context for high-performance fingerprinting with buffer reuse.
///
/// Maintains a reusable preprocessor, hasher, and internal buffers
/// to minimize allocations in high-throughput scenarios.
#[derive(Debug)]
pub struct FingerprinterContext {
    preprocessor: Preprocessor,
    exact_hasher: Hasher,
    #[allow(dead_code)]
    has_simd: bool,
    dct_scratch: DctScratch,
}

impl Default for FingerprinterContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprinterContext {
    /// Creates a new fingerprinter context with cached resources.
    #[must_use]
    pub fn new() -> Self {
        let has_simd = get_cpu_extensions() != crate::imgproc::preprocess::CpuExtensions::None;
        Self {
            preprocessor: Preprocessor::new(),
            exact_hasher: Hasher::new(),
            has_simd,
            dct_scratch: DctScratch::new(),
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
        self.fingerprint_with_preprocess(image_bytes, &PreprocessConfig::default())
    }

    /// Computes all perceptual hashes with a tunable [`PreprocessConfig`].
    ///
    /// Use this to tighten or widen the decode-time guards
    /// (`max_input_bytes`, `max_dimension`, `min_dimension`) per call.
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes, preprocess), fields(size = image_bytes.len())))]
    pub fn fingerprint_with_preprocess(
        &mut self,
        image_bytes: &[u8],
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();
        let result = self.compute_all_hashes(image_bytes, preprocess);
        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            "fingerprint completed"
        );
        result
    }

    /// Reads an image from disk and computes its multi-algorithm fingerprint.
    ///
    /// Convenience wrapper around [`fingerprint`](Self::fingerprint) that handles
    /// the file read. Files larger than 50 MB are rejected before any read happens.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::IoError`] if the file cannot be opened, read, or exceeds 50 MB.
    /// - All errors documented on [`fingerprint`](Self::fingerprint).
    pub fn fingerprint_path<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        self.fingerprint_path_with_preprocess(path, &PreprocessConfig::default())
    }

    /// Reads an image from disk and computes its multi-algorithm fingerprint
    /// with a tunable [`PreprocessConfig`]. The same config gates both the
    /// pre-read file-size check and the decode-time guards.
    pub fn fingerprint_path_with_preprocess<P: AsRef<Path>>(
        &mut self,
        path: P,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let bytes = read_image_file(path.as_ref(), preprocess)?;
        self.fingerprint_with_preprocess(&bytes, preprocess)
    }

    /// Reads an image from disk and computes a single-algorithm fingerprint.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::IoError`] if the file cannot be opened, read, or exceeds 50 MB.
    /// - All errors documented on [`fingerprint_with`](Self::fingerprint_with).
    pub fn fingerprint_path_with<P: AsRef<Path>>(
        &mut self,
        path: P,
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let bytes = read_image_file(path.as_ref(), &PreprocessConfig::default())?;
        self.fingerprint_with(&bytes, algorithm)
    }

    /// Computes a multi-algorithm fingerprint from an already-decoded [`DynamicImage`].
    ///
    /// Skips the decode step entirely — useful when you already hold a
    /// `DynamicImage` (e.g., from a video frame or in-memory composition).
    /// The BLAKE3 exact hash is computed over the image's raw RGB8 pixel data.
    pub fn fingerprint_image(
        &mut self,
        image: &image::DynamicImage,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let rgb = image.to_rgb8();
        let raw = rgb.as_raw();

        let exact_hash: [u8; 32] = {
            self.exact_hasher.reset();
            self.exact_hasher.update(raw);
            *self.exact_hasher.finalize().as_bytes()
        };

        let normalized = self.preprocessor.normalize_as_slice(image)?;

        let global_region = extract_global_region_from_raw(normalized);
        let blocks = extract_blocks_from_raw(normalized);

        let (ahash_fp, phash_fp, dhash_fp) = {
            #[cfg(feature = "parallel")]
            {
                let (ahash_result, (phash_result, dhash_result)) = rayon::join(
                    || Self::compute_ahash_data(&global_region, &blocks),
                    || {
                        rayon::join(
                            || {
                                Self::compute_phash_data(
                                    &global_region,
                                    &blocks,
                                    &mut self.dct_scratch,
                                )
                            },
                            || Self::compute_dhash_data(&global_region, &blocks),
                        )
                    },
                );
                let (ahash_global, ahash_blocks) = ahash_result;
                let (phash_global, phash_blocks) = phash_result;
                let (dhash_global, dhash_blocks) = dhash_result;
                (
                    ImageFingerprint::new(exact_hash, ahash_global, ahash_blocks),
                    ImageFingerprint::new(exact_hash, phash_global, phash_blocks),
                    ImageFingerprint::new(exact_hash, dhash_global, dhash_blocks),
                )
            }
            #[cfg(not(feature = "parallel"))]
            {
                let (ahash_global, ahash_blocks) =
                    Self::compute_ahash_data(&global_region, &blocks);
                let (phash_global, phash_blocks) =
                    Self::compute_phash_data(&global_region, &blocks, &mut self.dct_scratch);
                let (dhash_global, dhash_blocks) =
                    Self::compute_dhash_data(&global_region, &blocks);
                (
                    ImageFingerprint::new(exact_hash, ahash_global, ahash_blocks),
                    ImageFingerprint::new(exact_hash, phash_global, phash_blocks),
                    ImageFingerprint::new(exact_hash, dhash_global, dhash_blocks),
                )
            }
        };

        Ok(MultiHashFingerprint::new(
            exact_hash, ahash_fp, phash_fp, dhash_fp,
        ))
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
        self.fingerprint_with_algorithm_and_preprocess(
            image_bytes,
            algorithm,
            &PreprocessConfig::default(),
        )
    }

    /// Computes a single perceptual hash with a tunable [`PreprocessConfig`].
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes, preprocess), fields(size = image_bytes.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_with_algorithm_and_preprocess(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
        preprocess: &PreprocessConfig,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();
        let result = self.compute_single_hash(image_bytes, algorithm, preprocess);
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
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let exact_hash: [u8; 32] = trace_stage!("exact_hash", {
            self.exact_hasher.reset();
            self.exact_hasher.update(image_bytes);
            *self.exact_hasher.finalize().as_bytes()
        });

        let image = trace_result_stage!("decode", {
            decode_image_with_config(image_bytes, preprocess)
        });
        let normalized = trace_result_stage!("normalize", {
            self.preprocessor.normalize_as_slice(&image)
        });

        let global_region = trace_stage!("extract_global_region", {
            extract_global_region_from_raw(normalized)
        });
        let blocks = trace_stage!("extract_blocks", { extract_blocks_from_raw(normalized) });

        let (ahash_fp, phash_fp, dhash_fp) = trace_stage!("multi_hash", {
            #[cfg(feature = "parallel")]
            {
                let (ahash_result, (phash_result, dhash_result)) = rayon::join(
                    || Self::compute_ahash_data(&global_region, &blocks),
                    || {
                        rayon::join(
                            || {
                                Self::compute_phash_data(
                                    &global_region,
                                    &blocks,
                                    &mut self.dct_scratch,
                                )
                            },
                            || Self::compute_dhash_data(&global_region, &blocks),
                        )
                    },
                );

                let (ahash_global, ahash_blocks) = ahash_result;
                let (phash_global, phash_blocks) = phash_result;
                let (dhash_global, dhash_blocks) = dhash_result;

                (
                    ImageFingerprint::new(exact_hash, ahash_global, ahash_blocks),
                    ImageFingerprint::new(exact_hash, phash_global, phash_blocks),
                    ImageFingerprint::new(exact_hash, dhash_global, dhash_blocks),
                )
            }

            #[cfg(not(feature = "parallel"))]
            {
                let (ahash_global, ahash_blocks) =
                    Self::compute_ahash_data(&global_region, &blocks);
                let (phash_global, phash_blocks) =
                    Self::compute_phash_data(&global_region, &blocks, &mut self.dct_scratch);
                let (dhash_global, dhash_blocks) =
                    Self::compute_dhash_data(&global_region, &blocks);

                (
                    ImageFingerprint::new(exact_hash, ahash_global, ahash_blocks),
                    ImageFingerprint::new(exact_hash, phash_global, phash_blocks),
                    ImageFingerprint::new(exact_hash, dhash_global, dhash_blocks),
                )
            }
        });

        Ok(MultiHashFingerprint::new(
            exact_hash, ahash_fp, phash_fp, dhash_fp,
        ))
    }

    fn compute_single_hash(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
        preprocess: &PreprocessConfig,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let exact_hash: [u8; 32] = trace_stage!("exact_hash", {
            self.exact_hasher.reset();
            self.exact_hasher.update(image_bytes);
            *self.exact_hasher.finalize().as_bytes()
        });

        let image = trace_result_stage!("decode", {
            decode_image_with_config(image_bytes, preprocess)
        });
        let normalized = trace_result_stage!("normalize", {
            self.preprocessor.normalize_as_slice(&image)
        });

        let global_region = trace_stage!("extract_global_region", {
            extract_global_region_from_raw(normalized)
        });
        let blocks = trace_stage!("extract_blocks", { extract_blocks_from_raw(normalized) });

        let (global_hash, block_hashes) = trace_stage!("single_hash", {
            match algorithm {
                HashAlgorithm::AHash => Self::compute_ahash_data(&global_region, &blocks),
                HashAlgorithm::PHash => {
                    Self::compute_phash_data(&global_region, &blocks, &mut self.dct_scratch)
                }
                HashAlgorithm::DHash => Self::compute_dhash_data(&global_region, &blocks),
            }
        });

        Ok(ImageFingerprint::new(exact_hash, global_hash, block_hashes))
    }

    fn compute_phash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
        scratch: &mut DctScratch,
    ) -> (u64, [u64; 16]) {
        let global_hash = compute_phash_with_scratch(global_region, scratch).unwrap_or(0);

        let block_hashes = {
            let mut hashes = [0u64; 16];
            for (i, block) in blocks.iter().enumerate() {
                hashes[i] = compute_phash_from_64x64_with_scratch(block, scratch).unwrap_or(0);
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

        let mut hashes = [0u64; 16];
        for (i, block) in blocks.iter().enumerate() {
            hashes[i] = compute_ahash_from_64x64(block);
        }

        (global_hash, hashes)
    }

    fn compute_dhash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
    ) -> (u64, [u64; 16]) {
        let global_dhash = compute_dhash(global_region);

        let mut hashes = [0u64; 16];
        for (i, block) in blocks.iter().enumerate() {
            hashes[i] = compute_dhash_from_64x64(block);
        }

        (global_dhash, hashes)
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
    pub fn fingerprint(image_bytes: &[u8]) -> Result<MultiHashFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| ctx.borrow_mut().fingerprint(image_bytes))
    }

    /// Computes all perceptual hashes with a tunable [`PreprocessConfig`].
    pub fn fingerprint_with_preprocess(
        image_bytes: &[u8],
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| {
            ctx.borrow_mut()
                .fingerprint_with_preprocess(image_bytes, preprocess)
        })
    }

    /// Reads an image from disk and computes its multi-algorithm fingerprint.
    ///
    /// Convenience wrapper around [`fingerprint`](Self::fingerprint) that handles
    /// the file read. Files larger than 50 MB are rejected before any read happens.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::IoError`] if the file cannot be opened, read, or exceeds 50 MB.
    /// - All errors documented on [`fingerprint`](Self::fingerprint).
    pub fn fingerprint_path<P: AsRef<Path>>(
        path: P,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        Self::fingerprint_path_with_preprocess(path, &PreprocessConfig::default())
    }

    /// Reads an image from disk and computes its multi-algorithm fingerprint
    /// with a tunable [`PreprocessConfig`].
    pub fn fingerprint_path_with_preprocess<P: AsRef<Path>>(
        path: P,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let bytes = read_image_file(path.as_ref(), preprocess)?;
        Self::fingerprint_with_preprocess(&bytes, preprocess)
    }

    /// Reads an image from disk and computes a single-algorithm fingerprint.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::IoError`] if the file cannot be opened, read, or exceeds 50 MB.
    /// - All errors documented on [`fingerprint_with`](Self::fingerprint_with).
    pub fn fingerprint_path_with<P: AsRef<Path>>(
        path: P,
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let bytes = read_image_file(path.as_ref(), &PreprocessConfig::default())?;
        Self::fingerprint_with(&bytes, algorithm)
    }

    /// Computes a multi-algorithm fingerprint from an already-decoded [`DynamicImage`].
    ///
    /// Skips the decode step — useful when you already hold a `DynamicImage`
    /// (e.g., from a video frame or in-memory composition).
    pub fn fingerprint_image(
        image: &image::DynamicImage,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| ctx.borrow_mut().fingerprint_image(image))
    }

    /// Computes a single perceptual hash using the specified algorithm.
    ///
    /// Use this when you need a specific algorithm or want to minimize
    /// computation for high-throughput scenarios.
    ///
    /// # Arguments
    /// * `image_bytes` - Raw image data
    /// * `algorithm` - Hash algorithm to use (PHash or DHash)
    pub fn fingerprint_with(
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| ctx.borrow_mut().fingerprint_with(image_bytes, algorithm))
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
    /// When the `parallel` feature is enabled, uses per-thread context caching
    /// to minimize allocations across parallel workers.
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

            let results: Vec<(S, Result<MultiHashFingerprint, ImgFprintError>)> = images
                .par_iter()
                .map_init(
                    || std::cell::RefCell::new(FingerprinterContext::new()),
                    |ctx, (id, bytes)| (id.clone(), ctx.borrow_mut().fingerprint(bytes)),
                )
                .collect();

            #[cfg(feature = "tracing")]
            let (succeeded, failed) = count_results(&results);
            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                succeeded, failed, "parallel batch completed"
            );

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            let results: Vec<(S, Result<MultiHashFingerprint, ImgFprintError>)> = images
                .iter()
                .map(|(id, bytes)| (id.clone(), Self::fingerprint(bytes)))
                .collect();

            #[cfg(feature = "tracing")]
            let (succeeded, failed) = count_results(&results);
            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                succeeded, failed, "sequential batch completed"
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

            let results: Vec<(S, Result<ImageFingerprint, ImgFprintError>)> = images
                .par_iter()
                .map_init(
                    || std::cell::RefCell::new(FingerprinterContext::new()),
                    |ctx, (id, bytes)| {
                        (
                            id.clone(),
                            ctx.borrow_mut().fingerprint_with(bytes, algorithm),
                        )
                    },
                )
                .collect();

            #[cfg(feature = "tracing")]
            let (succeeded, failed) = count_results(&results);
            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                succeeded, failed, "parallel batch_with completed"
            );

            results
        }

        #[cfg(not(feature = "parallel"))]
        {
            let results: Vec<(S, Result<ImageFingerprint, ImgFprintError>)> = images
                .iter()
                .map(|(id, bytes)| (id.clone(), Self::fingerprint_with(bytes, algorithm)))
                .collect();

            #[cfg(feature = "tracing")]
            let (succeeded, failed) = count_results(&results);
            #[cfg(feature = "tracing")]
            debug!(
                duration_ms = start.elapsed().as_millis(),
                succeeded, failed, "sequential batch_with completed"
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
    pub fn fingerprint_batch_chunked<S, F>(images: &[(S, Vec<u8>)], chunk_size: usize, callback: F)
    where
        S: Send + Sync + Clone + 'static,
        F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>),
    {
        let mut ctx = FingerprinterContext::new();
        ctx.fingerprint_batch_chunked(images, chunk_size, callback);
    }

    /// Processes an iterator of file paths, yielding fingerprint results lazily.
    ///
    /// Unlike [`fingerprint_batch`](Self::fingerprint_batch), this does not require
    /// loading all images into memory at once. Each path is read and fingerprinted
    /// on demand.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use imgfprint::ImageFingerprinter;
    /// use std::path::PathBuf;
    ///
    /// let paths = vec![PathBuf::from("a.jpg"), PathBuf::from("b.png")];
    /// for (path, result) in ImageFingerprinter::fingerprint_stream(paths.into_iter()) {
    ///     match result {
    ///         Ok(fp) => println!("{}: {}", path.display(), fp),
    ///         Err(e) => eprintln!("{}: {}", path.display(), e),
    ///     }
    /// }
    /// ```
    pub fn fingerprint_stream<I, P>(
        paths: I,
    ) -> impl Iterator<Item = (P, Result<MultiHashFingerprint, ImgFprintError>)>
    where
        I: Iterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut ctx = FingerprinterContext::new();
        paths.map(move |p| {
            let result = ctx.fingerprint_path(p.as_ref());
            (p, result)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_image(width: u32, height: u32) -> Vec<u8> {
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
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
            .map(|i| {
                (
                    i,
                    create_test_image(100 + i as u32 * 10, 100 + i as u32 * 10),
                )
            })
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

    #[test]
    fn test_fingerprint_path_static() {
        let img = create_test_image(64, 64);
        let dir = std::env::temp_dir();
        let path = dir.join("imgfprint_test_path_static.png");
        std::fs::write(&path, &img).unwrap();

        let from_path = ImageFingerprinter::fingerprint_path(&path).unwrap();
        let from_bytes = ImageFingerprinter::fingerprint(&img).unwrap();
        assert_eq!(from_path, from_bytes);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_fingerprint_path_with_static() {
        let img = create_test_image(64, 64);
        let dir = std::env::temp_dir();
        let path = dir.join("imgfprint_test_path_with_static.png");
        std::fs::write(&path, &img).unwrap();

        let from_path =
            ImageFingerprinter::fingerprint_path_with(&path, HashAlgorithm::PHash).unwrap();
        let from_bytes = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::PHash).unwrap();
        assert_eq!(from_path, from_bytes);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_fingerprint_path_context() {
        let img = create_test_image(64, 64);
        let dir = std::env::temp_dir();
        let path = dir.join("imgfprint_test_path_ctx.png");
        std::fs::write(&path, &img).unwrap();

        let mut ctx = FingerprinterContext::new();
        let fp1 = ctx.fingerprint_path(&path).unwrap();
        let fp2 = ctx.fingerprint(&img).unwrap();
        assert_eq!(fp1, fp2);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_fingerprint_path_missing_file() {
        let path = std::env::temp_dir().join("imgfprint_does_not_exist_xyzzy.png");
        let err = ImageFingerprinter::fingerprint_path(&path).unwrap_err();
        assert!(matches!(err, ImgFprintError::IoError(_)), "got: {:?}", err);
    }

    #[test]
    fn test_fingerprint_path_oversized_file() {
        use crate::imgproc::decode::DEFAULT_MAX_INPUT_BYTES;

        let dir = std::env::temp_dir();
        let path = dir.join("imgfprint_test_oversized.bin");
        let f = std::fs::File::create(&path).unwrap();
        f.set_len((DEFAULT_MAX_INPUT_BYTES as u64) + 1).unwrap();
        drop(f);

        let err = ImageFingerprinter::fingerprint_path(&path).unwrap_err();
        assert!(matches!(err, ImgFprintError::IoError(_)), "got: {:?}", err);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_preprocess_config_path_size_guard() {
        let dir = std::env::temp_dir();
        let path = dir.join("imgfprint_test_preprocess_path_guard.bin");
        let f = std::fs::File::create(&path).unwrap();
        // Just over 1 KiB.
        f.set_len(1025).unwrap();
        drop(f);

        let tight = PreprocessConfig {
            max_input_bytes: 1024,
            ..PreprocessConfig::default()
        };
        let err = ImageFingerprinter::fingerprint_path_with_preprocess(&path, &tight).unwrap_err();
        assert!(matches!(err, ImgFprintError::IoError(_)), "got: {:?}", err);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_fingerprint_in_hashset() {
        // The whole point of deriving Hash: fingerprints must be HashSet-able.
        use std::collections::HashSet;

        let img1 = create_test_image(64, 64);
        let img2 = create_test_image(80, 80);

        let fp1 = ImageFingerprinter::fingerprint(&img1).unwrap();
        let fp1_again = ImageFingerprinter::fingerprint(&img1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img2).unwrap();

        let mut set = HashSet::new();
        set.insert(fp1);
        set.insert(fp1_again);
        set.insert(fp2);
        assert_eq!(set.len(), 2);

        let mut single_set = HashSet::new();
        let single = ImageFingerprinter::fingerprint_with(&img1, HashAlgorithm::DHash).unwrap();
        single_set.insert(single);
        single_set.insert(single);
        assert_eq!(single_set.len(), 1);
    }

    #[test]
    fn test_fingerprint_stream_basic() {
        let img = create_test_image(64, 64);
        let dir = std::env::temp_dir();
        let p1 = dir.join("imgfprint_stream_test1.png");
        let p2 = dir.join("imgfprint_stream_test2.png");
        std::fs::write(&p1, &img).unwrap();
        std::fs::write(&p2, &img).unwrap();

        let paths = vec![p1.clone(), p2.clone()];
        let results: Vec<_> = ImageFingerprinter::fingerprint_stream(paths.into_iter()).collect();

        assert_eq!(results.len(), 2);
        assert!(results[0].1.is_ok());
        assert!(results[1].1.is_ok());
        // Same image → same fingerprint
        assert_eq!(
            results[0].1.as_ref().unwrap().exact_hash(),
            results[1].1.as_ref().unwrap().exact_hash()
        );

        let _ = std::fs::remove_file(&p1);
        let _ = std::fs::remove_file(&p2);
    }

    #[test]
    fn test_fingerprint_stream_missing_file() {
        let paths = vec![std::path::PathBuf::from("/nonexistent_imgfprint_xyz.png")];
        let results: Vec<_> = ImageFingerprinter::fingerprint_stream(paths.into_iter()).collect();
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_err());
    }

    #[test]
    fn test_fingerprint_image_basic() {
        let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let dynamic = image::DynamicImage::ImageRgb8(img);

        let mut ctx = FingerprinterContext::new();
        let fp = ctx.fingerprint_image(&dynamic).unwrap();
        assert!(!fp.exact_hash().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_fingerprint_image_deterministic() {
        let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let dynamic = image::DynamicImage::ImageRgb8(img);

        let fp1 = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();
        let fp2 = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();
        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        assert_eq!(fp1.phash().global_hash(), fp2.phash().global_hash());
    }

    #[test]
    fn test_fingerprint_image_matches_bytes_perceptually() {
        let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let dynamic = image::DynamicImage::ImageRgb8(img.clone());

        // Encode to PNG bytes
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let fp_image = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();
        let fp_bytes = ImageFingerprinter::fingerprint(&buf).unwrap();

        // Perceptual hashes should match (same pixel data)
        assert_eq!(
            fp_image.phash().global_hash(),
            fp_bytes.phash().global_hash()
        );
        assert_eq!(
            fp_image.dhash().global_hash(),
            fp_bytes.dhash().global_hash()
        );
        // Exact hashes differ (one hashes raw RGB, other hashes PNG bytes)
        assert_ne!(fp_image.exact_hash(), fp_bytes.exact_hash());
    }
}
