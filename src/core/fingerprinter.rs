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
    extract_blocks_into_buffer, extract_global_region_into_buffer, Preprocessor,
};
use blake3::Hasher;
use image::GenericImageView;
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

/// Context for high-performance fingerprinting with buffer reuse.
///
/// Maintains a reusable preprocessor, hasher, and internal buffers
/// to minimize allocations in high-throughput scenarios.
///
/// The `blocks_buffer` and `global_region_buffer` are heap-allocated once
/// and reused across calls, eliminating a 256 KiB + 4 KiB stack allocation
/// per fingerprint call. This prevents stack overflow under rayon workers
/// with limited stack sizes.
#[derive(Debug)]
pub struct FingerprinterContext {
    preprocessor: Preprocessor,
    exact_hasher: Hasher,
    dct_scratch: DctScratch,
    /// Heap-allocated 4x4 grid of 64x64 float blocks (256 KiB).
    /// Reused across fingerprint calls to avoid per-call stack allocation.
    blocks_buffer: Box<[[f32; 64 * 64]; 16]>,
    /// Heap-allocated center 32x32 float region (4 KiB).
    /// Reused across fingerprint calls to avoid per-call stack allocation.
    global_region_buffer: Box<[f32; 32 * 32]>,
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
        Self {
            preprocessor: Preprocessor::new(),
            exact_hasher: Hasher::new(),
            dct_scratch: DctScratch::new(),
            blocks_buffer: Box::new([[0.0f32; 64 * 64]; 16]),
            global_region_buffer: Box::new([0.0f32; 32 * 32]),
        }
    }

    /// BLAKE3-digests `bytes` with the reusable hasher (reset first).
    fn update_exact(&mut self, bytes: &[u8]) -> [u8; 32] {
        self.exact_hasher.reset();
        self.exact_hasher.update(bytes);
        *self.exact_hasher.finalize().as_bytes()
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
    ///
    /// # Exact hash semantics
    ///
    /// Unlike [`fingerprint`](Self::fingerprint) (which hashes the raw
    /// compressed file bytes), this method computes the BLAKE3 `exact_hash`
    /// from the **decoded RGB8 pixel buffer**. Consequently:
    ///
    /// - Two `DynamicImage` values with identical pixel data will always
    ///   produce the **same** exact hash, regardless of how they were
    ///   originally encoded on disk.
    /// - The exact hash from `fingerprint_image` will **differ** from the
    ///   exact hash produced by `fingerprint` on the encoded file bytes of
    ///   the same image.
    ///
    /// Use `fingerprint` for byte-level deduplication of files, and
    /// `fingerprint_image` for pixel-level deduplication of decoded images.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::ImageTooSmall`] if either edge is below the
    ///   minimum dimension (default 32 px) — same guard as the byte paths.
    /// - [`ImgFprintError::InvalidImage`] if either edge exceeds the maximum
    ///   dimension (default 8192 px).
    /// - [`ImgFprintError::ProcessingError`] if normalization fails.
    pub fn fingerprint_image(
        &mut self,
        image: &image::DynamicImage,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        self.fingerprint_image_with_preprocess(image, &PreprocessConfig::default())
    }

    /// Computes a multi-algorithm fingerprint from an already-decoded
    /// [`DynamicImage`] with a tunable [`PreprocessConfig`].
    ///
    /// Same semantics as [`fingerprint_image`](Self::fingerprint_image); the
    /// config's `min_dimension` / `max_dimension` guards are applied to the
    /// decoded image dimensions before any hashing happens.
    ///
    /// # Errors
    ///
    /// See [`fingerprint_image`](Self::fingerprint_image).
    pub fn fingerprint_image_with_preprocess(
        &mut self,
        image: &image::DynamicImage,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        // Enforce the same dimension guards as the byte-decode paths so a
        // pre-decoded image can't bypass min/max validation.
        let (width, height) = image.dimensions();
        crate::imgproc::decode::validate_dimensions(width, height, preprocess)?;

        // Compute exact hash from RGB8 pixels, avoiding clone when already RGB8.
        // Luma8/RGBA8 use fast raw (triplicate/stride-copy) conversions that
        // produce byte-identical input to `to_rgb8()` (verified: Luma maps
        // Y -> (Y,Y,Y); RGBA drops alpha) without its per-pixel dispatch cost.
        // The converted bytes feed the hasher directly; no intermediate image.
        let rgb_owned;
        let raw: &[u8] = match image {
            image::DynamicImage::ImageRgb8(rgb) => rgb.as_raw(),
            image::DynamicImage::ImageLuma8(gray) => {
                let src = gray.as_raw();
                let mut buf = Vec::with_capacity(src.len() * 3);
                for &y in src {
                    buf.extend_from_slice(&[y, y, y]);
                }
                rgb_owned = buf;
                &rgb_owned
            }
            image::DynamicImage::ImageRgba8(rgba) => {
                let src = rgba.as_raw();
                let mut buf = Vec::with_capacity(src.len() / 4 * 3);
                let (chunks, _) = src.as_chunks::<4>();
                for px in chunks {
                    buf.extend_from_slice(&px[..3]);
                }
                rgb_owned = buf;
                &rgb_owned
            }
            _ => {
                rgb_owned = image.to_rgb8().into_raw();
                &rgb_owned
            }
        };

        let exact_hash: [u8; 32] = self.update_exact(raw);

        let normalized = self.preprocessor.normalize_as_slice(image)?;

        extract_global_region_into_buffer(normalized, &mut self.global_region_buffer);
        extract_blocks_into_buffer(normalized, &mut self.blocks_buffer);

        let (ahash_fp, phash_fp, dhash_fp) = self.compute_all_layers(exact_hash)?;

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
        let result = self.compute_single_hash(image_bytes, algorithm, preprocess, false);
        #[cfg(feature = "tracing")]
        debug!(
            duration_ms = start.elapsed().as_millis(),
            "fingerprint_with completed"
        );
        result
    }

    /// Computes a single perceptual hash using a faster bilinear resize.
    ///
    /// # Consistency warning
    ///
    /// This method resizes with **bilinear** filtering, while
    /// [`fingerprint`](Self::fingerprint) and
    /// [`fingerprint_with`](Self::fingerprint_with) use **Lanczos3**.
    /// Fingerprints produced here are therefore *not* bit-identical to those
    /// produced by the standard methods for the same image and algorithm —
    /// comparing fingerprints across modes can shift similarity scores by
    /// several Hamming bits. Only use this method when **every** fingerprint
    /// in your index and every query fingerprint is produced by this method.
    ///
    /// The speedup comes from the resize stage (~2x faster resize); AHash and
    /// DHash tolerate the simpler interpolation.
    /// [`HashAlgorithm::PHash`] falls back to the Lanczos3 path because its
    /// DCT requires high-quality downsampling.
    ///
    /// # Errors
    ///
    /// Same errors as [`fingerprint_with`](Self::fingerprint_with).
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes), fields(size = image_bytes.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_with_fast(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        self.compute_single_hash(image_bytes, algorithm, &PreprocessConfig::default(), true)
    }

    /// Internal: computes all three perceptual hashes plus the BLAKE3 exact hash.
    ///
    /// The `exact_hash` field in the returned [`MultiHashFingerprint`] is the
    /// BLAKE3 digest of the **raw compressed file bytes** (`image_bytes`). This
    /// means two files with identical pixel content but different encodings
    /// (e.g., the same photo saved as PNG vs JPEG, or two JPEG files with
    /// different compression settings) will produce **different** exact hashes.
    ///
    /// For an exact hash computed from decoded pixel data instead, see
    /// [`fingerprint_image`](Self::fingerprint_image).
    fn compute_all_hashes(
        &mut self,
        image_bytes: &[u8],
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let exact_hash: [u8; 32] = trace_stage!("exact_hash", { self.update_exact(image_bytes) });

        let image = trace_result_stage!("decode", {
            decode_image_with_config(image_bytes, preprocess)
        });
        let normalized = trace_result_stage!("normalize", {
            self.preprocessor.normalize_as_slice(&image)
        });

        trace_stage!("extract_global_region", {
            extract_global_region_into_buffer(normalized, &mut self.global_region_buffer)
        });
        trace_stage!("extract_blocks", {
            extract_blocks_into_buffer(normalized, &mut self.blocks_buffer)
        });

        let (ahash_fp, phash_fp, dhash_fp) =
            trace_stage!("multi_hash", { self.compute_all_layers(exact_hash)? });

        Ok(MultiHashFingerprint::new(
            exact_hash, ahash_fp, phash_fp, dhash_fp,
        ))
    }

    /// Computes all three perceptual layers from the extracted buffers.
    ///
    /// Shared by [`compute_all_hashes`](Self::compute_all_hashes) and the
    /// already-decoded [`fingerprint_image_with_preprocess`](Self::fingerprint_image_with_preprocess)
    /// path so the parallel/sequential fan-out lives in exactly one place.
    /// Output is bit-identical under both `parallel` configurations.
    fn compute_all_layers(
        &mut self,
        exact_hash: [u8; 32],
    ) -> Result<(ImageFingerprint, ImageFingerprint, ImageFingerprint), ImgFprintError> {
        #[cfg(feature = "parallel")]
        {
            let (ahash_result, (phash_result, dhash_result)) = rayon::join(
                || Self::compute_ahash_data(&self.global_region_buffer, &self.blocks_buffer),
                || {
                    rayon::join(
                        || {
                            Self::compute_phash_data(
                                &self.global_region_buffer,
                                &self.blocks_buffer,
                                &mut self.dct_scratch,
                            )
                        },
                        || {
                            Self::compute_dhash_data(
                                &self.global_region_buffer,
                                &self.blocks_buffer,
                            )
                        },
                    )
                },
            );

            let (ahash_global, ahash_blocks) = ahash_result;
            let (phash_global, phash_blocks) = phash_result?;
            let (dhash_global, dhash_blocks) = dhash_result;

            Ok((
                ImageFingerprint::new(exact_hash, ahash_global, ahash_blocks),
                ImageFingerprint::new(exact_hash, phash_global, phash_blocks),
                ImageFingerprint::new(exact_hash, dhash_global, dhash_blocks),
            ))
        }

        #[cfg(not(feature = "parallel"))]
        {
            let (ahash_global, ahash_blocks) =
                Self::compute_ahash_data(&self.global_region_buffer, &self.blocks_buffer);
            let (phash_global, phash_blocks) = Self::compute_phash_data(
                &self.global_region_buffer,
                &self.blocks_buffer,
                &mut self.dct_scratch,
            )?;
            let (dhash_global, dhash_blocks) =
                Self::compute_dhash_data(&self.global_region_buffer, &self.blocks_buffer);

            Ok((
                ImageFingerprint::new(exact_hash, ahash_global, ahash_blocks),
                ImageFingerprint::new(exact_hash, phash_global, phash_blocks),
                ImageFingerprint::new(exact_hash, dhash_global, dhash_blocks),
            ))
        }
    }

    fn compute_single_hash(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
        preprocess: &PreprocessConfig,
        fast_resize: bool,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let exact_hash: [u8; 32] = trace_stage!("exact_hash", { self.update_exact(image_bytes) });

        let image = trace_result_stage!("decode", {
            decode_image_with_config(image_bytes, preprocess)
        });

        // Default path uses Lanczos3 for every algorithm so that
        // `fingerprint_with(bytes, alg)` is bit-identical to the `alg` layer
        // of `fingerprint(bytes)`. The bilinear fast path is opt-in via
        // `fingerprint_with_fast` (see its docs for the consistency caveat).
        // PHash always uses Lanczos3 — its DCT needs high-quality downsampling.
        let normalized = match algorithm {
            HashAlgorithm::AHash | HashAlgorithm::DHash if fast_resize => {
                trace_result_stage!("normalize_fast", {
                    self.preprocessor.normalize_as_slice_fast(&image)
                })
            }
            _ => trace_result_stage!("normalize", {
                self.preprocessor.normalize_as_slice(&image)
            }),
        };

        trace_stage!("extract_global_region", {
            extract_global_region_into_buffer(normalized, &mut self.global_region_buffer)
        });
        trace_stage!("extract_blocks", {
            extract_blocks_into_buffer(normalized, &mut self.blocks_buffer)
        });

        let (global_hash, block_hashes) = trace_stage!("single_hash", {
            match algorithm {
                HashAlgorithm::AHash => {
                    Self::compute_ahash_data(&self.global_region_buffer, &self.blocks_buffer)
                }
                HashAlgorithm::PHash => Self::compute_phash_data(
                    &self.global_region_buffer,
                    &self.blocks_buffer,
                    &mut self.dct_scratch,
                )?,
                HashAlgorithm::DHash => {
                    Self::compute_dhash_data(&self.global_region_buffer, &self.blocks_buffer)
                }
            }
        });

        Ok(ImageFingerprint::new(exact_hash, global_hash, block_hashes))
    }

    fn compute_phash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
        scratch: &mut DctScratch,
    ) -> Result<(u64, [u64; 16]), ImgFprintError> {
        let global_hash = compute_phash_with_scratch(global_region, scratch)?;

        // The 16 block hashes are independent, so with the `parallel` feature
        // they are computed across rayon workers (one DctScratch per worker).
        // Output is bit-identical to the sequential loop.
        #[cfg(feature = "parallel")]
        let block_hashes = {
            use rayon::prelude::*;
            let hashes: Vec<u64> = blocks
                .par_iter()
                .map_init(DctScratch::new, |block_scratch, block| {
                    compute_phash_from_64x64_with_scratch(block, block_scratch)
                })
                .collect::<Result<Vec<u64>, _>>()?;
            let mut arr = [0u64; 16];
            arr.copy_from_slice(&hashes);
            arr
        };

        #[cfg(not(feature = "parallel"))]
        let block_hashes = {
            let mut hashes = [0u64; 16];
            for (i, block) in blocks.iter().enumerate() {
                hashes[i] = compute_phash_from_64x64_with_scratch(block, scratch)?;
            }
            hashes
        };

        Ok((global_hash, block_hashes))
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
    ///
    /// With the `parallel` feature enabled, each chunk is fingerprinted in
    /// parallel (per-worker contexts, same strategy as
    /// [`fingerprint_batch`](crate::ImageFingerprinter::fingerprint_batch));
    /// the callback is still invoked sequentially in input order, so
    /// observable behavior matches the sequential implementation.
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

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            for chunk in images.chunks(chunk_size) {
                // Fingerprint the chunk in parallel with per-worker contexts,
                // then drain results in input order so the callback sequence
                // is deterministic.
                let results: Vec<(S, Result<MultiHashFingerprint, ImgFprintError>)> = chunk
                    .par_iter()
                    .map_init(FingerprinterContext::new, |ctx, (id, bytes)| {
                        (id.clone(), ctx.fingerprint(bytes))
                    })
                    .collect();

                for (id, result) in results {
                    #[cfg(feature = "tracing")]
                    {
                        if result.is_err() {
                            failed += 1;
                        }
                        processed += 1;
                    }
                    callback(id, result);
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
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
    /// # Exact hash semantics
    ///
    /// The `exact_hash` field is the BLAKE3 digest of `image_bytes` — the raw
    /// compressed file bytes as passed in. Two files encoding the same pixels
    /// differently (e.g., two distinct PNG encodings) will yield **different**
    /// exact hashes. For pixel-level exact matching, use
    /// [`fingerprint_image`](Self::fingerprint_image) instead.
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
    ///
    /// # Exact hash semantics
    ///
    /// The `exact_hash` is computed from the **decoded RGB8 pixel buffer**,
    /// not from file bytes. Two images with identical pixels will produce the
    /// same exact hash regardless of their on-disk encoding. See
    /// [`fingerprint`](Self::fingerprint) for file-byte-based exact hashing.
    pub fn fingerprint_image(
        image: &image::DynamicImage,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| ctx.borrow_mut().fingerprint_image(image))
    }

    /// Computes a multi-algorithm fingerprint from an already-decoded
    /// [`DynamicImage`] with a tunable [`PreprocessConfig`].
    pub fn fingerprint_image_with_preprocess(
        image: &image::DynamicImage,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| {
            ctx.borrow_mut()
                .fingerprint_image_with_preprocess(image, preprocess)
        })
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

    /// Computes a single perceptual hash using a faster bilinear resize.
    ///
    /// See [`FingerprinterContext::fingerprint_with_fast`] for the
    /// cross-mode consistency warning — fingerprints from this method are
    /// not bit-identical to those from [`fingerprint_with`](Self::fingerprint_with).
    pub fn fingerprint_with_fast(
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        SHARED_CTX.with(|ctx| {
            ctx.borrow_mut()
                .fingerprint_with_fast(image_bytes, algorithm)
        })
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

    /// Runs `f` over every image, preserving input order, in parallel when the
    /// `parallel` feature is on (per-worker contexts) and sequentially otherwise.
    fn run_batch<S, T, F>(
        images: &[(S, Vec<u8>)],
        #[cfg_attr(not(feature = "tracing"), allow(unused_variables))] stage: &'static str,
        f: F,
    ) -> Vec<(S, T)>
    where
        S: Send + Sync + Clone + 'static,
        T: Send,
        F: Fn(&mut FingerprinterContext, &[u8]) -> T + Send + Sync,
    {
        #[cfg(feature = "tracing")]
        let start = std::time::Instant::now();

        #[cfg(feature = "parallel")]
        let results: Vec<(S, T)> = {
            use rayon::prelude::*;

            images
                .par_iter()
                .map_init(FingerprinterContext::new, |ctx, (id, bytes)| {
                    (id.clone(), f(ctx, bytes))
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let results: Vec<(S, T)> = {
            let mut ctx = FingerprinterContext::new();
            images
                .iter()
                .map(|(id, bytes)| (id.clone(), f(&mut ctx, bytes)))
                .collect()
        };

        #[cfg(feature = "tracing")]
        {
            #[cfg(feature = "parallel")]
            let mode = "parallel";
            #[cfg(not(feature = "parallel"))]
            let mode = "sequential";
            debug!(
                duration_ms = start.elapsed().as_millis(),
                count = results.len(),
                "{mode} {stage} completed"
            );
        }

        results
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
        Self::run_batch(images, "batch", |ctx, bytes| ctx.fingerprint(bytes))
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
        Self::run_batch(images, "batch_with", |ctx, bytes| {
            ctx.fingerprint_with(bytes, algorithm)
        })
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
