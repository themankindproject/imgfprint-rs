use crate::core::fingerprint::{ImageFingerprint, MultiHashFingerprint};
use crate::core::similarity;
use crate::error::ImgFprintError;
use crate::hash::ahash::{compute_ahash, compute_ahash_from_64x64};
use crate::hash::algorithms::HashAlgorithm;
use crate::hash::dhash::{compute_dhash, compute_dhash_from_64x64};
use crate::hash::phash::{
    compute_phash_from_64x64_with_scratch, compute_phash_with_scratch, DctScratch,
};
use crate::imgproc::decode::{decode_image_with_config, validate_dimensions, PreprocessConfig};
use crate::imgproc::preprocess::{
    extract_blocks_into_buffer, extract_global_region_into_buffer, Preprocessor,
};
use blake3::Hasher;
use image::{DynamicImage, GenericImageView};
use std::cell::RefCell;
use std::path::Path;
#[cfg(feature = "tracing")]
use std::time::Instant;

/// Reads a file into memory, never buffering more than
/// `config.max_input_bytes + 1` bytes.
///
/// The size reported by metadata is only a fast-path rejection: special files
/// (`/dev/zero`, FIFOs, procfs entries) report a length of 0 while yielding
/// unbounded data, so the read itself is capped with [`Read::take`].
///
/// [`Read::take`]: std::io::Read::take
fn read_image_file(path: &Path, config: &PreprocessConfig) -> Result<Vec<u8>, ImgFprintError> {
    use std::io::Read;

    let max = config.max_input_bytes as u64;
    let too_large = |size: u64| {
        ImgFprintError::IoError(format!(
            "file size {size} bytes exceeds maximum {max} bytes"
        ))
    };

    let file = std::fs::File::open(path)?;
    let reported = file.metadata()?.len();
    if reported > max {
        return Err(too_large(reported));
    }

    // `reported` is only a hint (0 for special files); cap the pre-allocation
    // so a lying size can never trigger a large up-front reservation.
    #[allow(clippy::cast_possible_truncation)] // reported <= max_input_bytes: usize
    let mut bytes = Vec::with_capacity(reported as usize);
    file.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(too_large(bytes.len() as u64));
    }
    Ok(bytes)
}

thread_local! {
    /// Per-thread context backing the zero-config [`ImageFingerprinter`] API
    /// and the batch workers.
    static SHARED_CTX: RefCell<FingerprinterContext> = RefCell::new(FingerprinterContext::new());
}

/// Runs `f` against this thread's shared, default-config context.
///
/// Falls back to a fresh context if the shared one is already borrowed. That
/// can only happen through re-entrancy (e.g. an executor running another
/// fingerprint task on this thread while one is in progress); the fallback
/// trades one allocation for never panicking.
fn with_shared_ctx<T>(f: impl FnOnce(&mut FingerprinterContext) -> T) -> T {
    SHARED_CTX.with(|cell| match cell.try_borrow_mut() {
        Ok(mut ctx) => f(&mut ctx),
        Err(_) => f(&mut FingerprinterContext::new()),
    })
}

/// Heap-allocates the zeroed 16-block buffer without first building the
/// 256 KiB array on the stack (which `Box::new([..])` may do in debug builds).
fn zeroed_blocks() -> Box<[[f32; 64 * 64]; 16]> {
    vec![[0.0f32; 64 * 64]; 16]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("vec! built exactly 16 blocks"))
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

/// Reusable fingerprinting context: the decode guards ([`PreprocessConfig`])
/// plus every scratch buffer the pipeline needs, so repeated calls allocate
/// nothing beyond decoding the input.
///
/// Keep one per thread and reuse it for every image that thread processes.
/// The zero-config [`ImageFingerprinter`] functions do exactly that behind a
/// thread-local, so a context is only needed for custom guards or to own the
/// buffers explicitly.
///
/// # Example
///
/// ```rust
/// use imgfprint::{FingerprinterContext, PreprocessConfig};
///
/// // Tighter guards for untrusted uploads; applied to every call below.
/// let mut ctx = FingerprinterContext::with_config(PreprocessConfig {
///     max_input_bytes: 5 * 1024 * 1024,
///     max_dimension: 4096,
///     ..PreprocessConfig::default()
/// });
///
/// for bytes in [b"not an image".to_vec()] {
///     match ctx.fingerprint(&bytes) {
///         Ok(fp) => println!("{fp}"),
///         Err(e) => println!("rejected: {e}"),
///     }
/// }
/// ```
#[derive(Debug)]
pub struct FingerprinterContext {
    config: PreprocessConfig,
    preprocessor: Preprocessor,
    exact_hasher: Hasher,
    dct_scratch: DctScratch,
    /// 4x4 grid of 64x64 float blocks (256 KiB), heap-allocated once.
    blocks_buffer: Box<[[f32; 64 * 64]; 16]>,
    /// Center 32x32 float region (4 KiB), heap-allocated once.
    global_region_buffer: Box<[f32; 32 * 32]>,
}

impl Default for FingerprinterContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprinterContext {
    /// Creates a context with the default [`PreprocessConfig`].
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(PreprocessConfig::default())
    }

    /// Creates a context whose decode guards come from `config`.
    ///
    /// The config applies to every call on this context: the read cap of the
    /// path methods, the byte-size and dimension guards during decode, and
    /// the dimension guards of [`fingerprint_image`](Self::fingerprint_image).
    #[must_use]
    pub fn with_config(config: PreprocessConfig) -> Self {
        Self {
            config,
            preprocessor: Preprocessor::new(),
            exact_hasher: Hasher::new(),
            dct_scratch: DctScratch::new(),
            blocks_buffer: zeroed_blocks(),
            global_region_buffer: Box::new([0.0f32; 32 * 32]),
        }
    }

    /// Returns the decode guards this context applies.
    #[must_use]
    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }

    /// Replaces the decode guards for subsequent calls, keeping the warmed
    /// buffers.
    pub fn set_config(&mut self, config: PreprocessConfig) {
        self.config = config;
    }

    /// Fingerprints encoded image bytes with all three algorithms
    /// (AHash, PHash, DHash).
    ///
    /// The `exact_hash` is the BLAKE3 digest of `image_bytes` exactly as
    /// given, so two encodings of the same pixels have different exact
    /// hashes but (near-)identical perceptual hashes. EXIF orientation is
    /// applied before hashing.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::InvalidImage`]: empty input, input larger than
    ///   `max_input_bytes`, or an edge larger than `max_dimension`.
    /// - [`ImgFprintError::ImageTooSmall`]: an edge below `min_dimension`.
    /// - [`ImgFprintError::UnsupportedFormat`] / [`ImgFprintError::DecodeError`]:
    ///   unknown format or corrupt data.
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes), fields(size = image_bytes.len())))]
    pub fn fingerprint(
        &mut self,
        image_bytes: &[u8],
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let config = self.config;
        self.compute_all_hashes(image_bytes, &config)
    }

    /// Fingerprints encoded image bytes with a single algorithm.
    ///
    /// The result is bit-identical to the matching layer of
    /// [`fingerprint`](Self::fingerprint) (e.g. `fingerprint(b)?.phash()`), so
    /// single- and multi-algorithm indexes stay comparable. Saves only the
    /// hashing of the other two layers; decode and resize dominate the cost.
    ///
    /// # Errors
    ///
    /// Same as [`fingerprint`](Self::fingerprint).
    #[cfg_attr(feature = "tracing", instrument(skip(self, image_bytes), fields(size = image_bytes.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_with(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let config = self.config;
        self.compute_single_hash(image_bytes, algorithm, &config, false)
    }

    /// Reads a file (at most `max_input_bytes`) and fingerprints it with all
    /// three algorithms. Equivalent to [`fingerprint`](Self::fingerprint) on
    /// the file's bytes.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::IoError`]: the file cannot be read or is larger
    ///   than `max_input_bytes` (checked while reading, so special files
    ///   such as FIFOs cannot exhaust memory).
    /// - Everything documented on [`fingerprint`](Self::fingerprint).
    pub fn fingerprint_path<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let config = self.config;
        let bytes = read_image_file(path.as_ref(), &config)?;
        self.compute_all_hashes(&bytes, &config)
    }

    /// Reads a file (at most `max_input_bytes`) and fingerprints it with a
    /// single algorithm.
    ///
    /// # Errors
    ///
    /// Same as [`fingerprint_path`](Self::fingerprint_path).
    pub fn fingerprint_path_with<P: AsRef<Path>>(
        &mut self,
        path: P,
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let config = self.config;
        let bytes = read_image_file(path.as_ref(), &config)?;
        self.compute_single_hash(&bytes, algorithm, &config, false)
    }

    /// Fingerprints an already-decoded image (video frame, in-memory
    /// composition, custom decoder output) with all three algorithms.
    ///
    /// Perceptual hashes equal those of [`fingerprint`](Self::fingerprint) on
    /// a lossless encoding of the same pixels. The `exact_hash`, however, is
    /// the BLAKE3 digest of the image's RGB8 pixels (exactly the bytes of
    /// `image.to_rgb8()`), so identical pixels always share an exact hash
    /// regardless of how they were encoded or what color type they use.
    ///
    /// # Errors
    ///
    /// - [`ImgFprintError::ImageTooSmall`] / [`ImgFprintError::InvalidImage`]:
    ///   an edge outside `min_dimension..=max_dimension`.
    /// - [`ImgFprintError::ProcessingError`]: normalization failed.
    pub fn fingerprint_image(
        &mut self,
        image: &DynamicImage,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let config = self.config;
        self.compute_image_hashes(image, &config)
    }

    /// Fingerprints with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config)` (or `set_config`) and call `fingerprint`"
    )]
    pub fn fingerprint_with_preprocess(
        &mut self,
        image_bytes: &[u8],
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        self.compute_all_hashes(image_bytes, preprocess)
    }

    /// Fingerprints a file with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config)` (or `set_config`) and call `fingerprint_path`"
    )]
    pub fn fingerprint_path_with_preprocess<P: AsRef<Path>>(
        &mut self,
        path: P,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let bytes = read_image_file(path.as_ref(), preprocess)?;
        self.compute_all_hashes(&bytes, preprocess)
    }

    /// Fingerprints a decoded image with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config)` (or `set_config`) and call `fingerprint_image`"
    )]
    pub fn fingerprint_image_with_preprocess(
        &mut self,
        image: &DynamicImage,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        self.compute_image_hashes(image, preprocess)
    }

    /// Single-algorithm fingerprint with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config)` (or `set_config`) and call `fingerprint_with`"
    )]
    pub fn fingerprint_with_algorithm_and_preprocess(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
        preprocess: &PreprocessConfig,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        self.compute_single_hash(image_bytes, algorithm, preprocess, false)
    }

    /// Single-algorithm fingerprint using a bilinear instead of Lanczos3
    /// resize for AHash/DHash (PHash always uses Lanczos3).
    ///
    /// Output is **not** bit-compatible with any other entry point.
    #[deprecated(
        since = "0.4.7",
        note = "bit-incompatible with every other entry point for a small end-to-end gain (decode dominates); use `fingerprint_with`"
    )]
    pub fn fingerprint_with_fast(
        &mut self,
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        let config = self.config;
        self.compute_single_hash(image_bytes, algorithm, &config, true)
    }

    /// Fingerprints `images` and invokes `callback` for each result in input
    /// order. With the `parallel` feature the work runs on rayon workers and
    /// this context's buffers are not used.
    #[deprecated(
        since = "0.4.7",
        note = "the inputs are already in memory, so chunking bounds nothing; use `ImageFingerprinter::fingerprint_batch`, or `ImageFingerprinter::fingerprint_paths` for bounded memory"
    )]
    pub fn fingerprint_batch_chunked<S, F>(
        &mut self,
        images: &[(S, Vec<u8>)],
        chunk_size: usize,
        callback: F,
    ) where
        S: Send + Sync + Clone + 'static,
        F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>),
    {
        batch_chunked(images, chunk_size, callback);
    }

    /// BLAKE3-digests `bytes` with the reusable hasher.
    fn update_exact(&mut self, bytes: &[u8]) -> [u8; 32] {
        self.exact_hasher.reset();
        self.exact_hasher.update(bytes);
        *self.exact_hasher.finalize().as_bytes()
    }

    /// BLAKE3 of the image's RGB8 pixel bytes, i.e. of `image.to_rgb8()`.
    ///
    /// RGB8 is hashed in place. Luma8 and RGBA8 are converted through a small
    /// stack buffer instead of a full-frame copy, producing byte-identical
    /// input (Luma maps Y -> (Y,Y,Y); RGBA drops alpha). Only the
    /// `width * height * channels` prefix of the container is hashed, so
    /// over-allocated buffers (allowed by `ImageBuffer::from_raw`) cannot
    /// change the digest.
    fn pixel_exact_hash(&mut self, image: &DynamicImage) -> [u8; 32] {
        const CHUNK_PIXELS: usize = 4096;
        let pixels = image.width() as usize * image.height() as usize;

        self.exact_hasher.reset();
        match image {
            DynamicImage::ImageRgb8(rgb) => {
                self.exact_hasher.update(&rgb.as_raw()[..pixels * 3]);
            }
            DynamicImage::ImageLuma8(gray) => {
                let mut out = [0u8; CHUNK_PIXELS * 3];
                for chunk in gray.as_raw()[..pixels].chunks(CHUNK_PIXELS) {
                    let (dst, _) = out[..chunk.len() * 3].as_chunks_mut::<3>();
                    for (d, &y) in dst.iter_mut().zip(chunk) {
                        *d = [y, y, y];
                    }
                    self.exact_hasher.update(&out[..chunk.len() * 3]);
                }
            }
            DynamicImage::ImageRgba8(rgba) => {
                let mut out = [0u8; CHUNK_PIXELS * 3];
                for chunk in rgba.as_raw()[..pixels * 4].chunks(CHUNK_PIXELS * 4) {
                    let (src, _) = chunk.as_chunks::<4>();
                    let (dst, _) = out[..src.len() * 3].as_chunks_mut::<3>();
                    for (d, s) in dst.iter_mut().zip(src) {
                        *d = [s[0], s[1], s[2]];
                    }
                    self.exact_hasher.update(&out[..src.len() * 3]);
                }
            }
            other => {
                self.exact_hasher.update(other.to_rgb8().as_raw());
            }
        }
        *self.exact_hasher.finalize().as_bytes()
    }

    /// All three layers from encoded bytes; `exact_hash` covers the raw bytes.
    fn compute_all_hashes(
        &mut self,
        image_bytes: &[u8],
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let exact_hash: [u8; 32] = trace_stage!("exact_hash", { self.update_exact(image_bytes) });

        let image = trace_result_stage!("decode", {
            decode_image_with_config(image_bytes, preprocess)
        });
        self.hash_decoded(&image, exact_hash)
    }

    /// All three layers from a decoded image; `exact_hash` covers RGB8 pixels.
    fn compute_image_hashes(
        &mut self,
        image: &DynamicImage,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let (width, height) = image.dimensions();
        validate_dimensions(width, height, preprocess)?;

        // Color types without a native resize lane are converted to RGB8
        // once and shared by the exact hash and the resize (both would
        // otherwise run their own full-frame `to_rgb8()`).
        let converted;
        let image = match image {
            DynamicImage::ImageRgb8(_)
            | DynamicImage::ImageRgba8(_)
            | DynamicImage::ImageLuma8(_) => image,
            other => {
                converted = DynamicImage::ImageRgb8(other.to_rgb8());
                &converted
            }
        };

        let exact_hash = trace_stage!("exact_hash", { self.pixel_exact_hash(image) });
        self.hash_decoded(image, exact_hash)
    }

    /// Normalizes `image` and computes all three perceptual layers.
    fn hash_decoded(
        &mut self,
        image: &DynamicImage,
        exact_hash: [u8; 32],
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let normalized =
            trace_result_stage!("normalize", { self.preprocessor.normalize_as_slice(image) });

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
    /// Runs sequentially on purpose: hashing is ~0.3 ms of CPU per image,
    /// far below the cost of a rayon fork/join. Fanning out *inside* one
    /// image made single-image calls burn ~2.8x more CPU, oversubscribed
    /// batch workers, and let work-stealing re-enter the thread-local
    /// context mid-borrow. Parallelism belongs across images (batch APIs).
    fn compute_all_layers(
        &mut self,
        exact_hash: [u8; 32],
    ) -> Result<(ImageFingerprint, ImageFingerprint, ImageFingerprint), ImgFprintError> {
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

        // Lanczos3 for every algorithm keeps `fingerprint_with(bytes, alg)`
        // bit-identical to the `alg` layer of `fingerprint(bytes)`. The
        // bilinear path only serves the deprecated `fingerprint_with_fast`;
        // PHash always uses Lanczos3 (its DCT needs high-quality input).
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

        let mut block_hashes = [0u64; 16];
        for (hash, block) in block_hashes.iter_mut().zip(blocks.iter()) {
            *hash = compute_phash_from_64x64_with_scratch(block, scratch)?;
        }

        Ok((global_hash, block_hashes))
    }

    fn compute_ahash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
    ) -> (u64, [u64; 16]) {
        let mut hashes = [0u64; 16];
        for (hash, block) in hashes.iter_mut().zip(blocks.iter()) {
            *hash = compute_ahash_from_64x64(block);
        }
        (compute_ahash(global_region), hashes)
    }

    fn compute_dhash_data(
        global_region: &[f32; 32 * 32],
        blocks: &[[f32; 64 * 64]; 16],
    ) -> (u64, [u64; 16]) {
        let mut hashes = [0u64; 16];
        for (hash, block) in hashes.iter_mut().zip(blocks.iter()) {
            *hash = compute_dhash_from_64x64(block);
        }
        (compute_dhash(global_region), hashes)
    }
}

/// Runs `f` over every input in order, on rayon workers with the `parallel`
/// feature (each worker reuses its thread-local context) and sequentially
/// otherwise.
fn run_batch<In, Out, F>(
    inputs: Vec<In>,
    #[cfg_attr(not(feature = "tracing"), allow(unused_variables))] stage: &'static str,
    f: F,
) -> Vec<Out>
where
    In: Send,
    Out: Send,
    F: Fn(&mut FingerprinterContext, In) -> Out + Send + Sync,
{
    #[cfg(feature = "tracing")]
    let start = std::time::Instant::now();

    #[cfg(feature = "parallel")]
    let results: Vec<Out> = {
        use rayon::prelude::*;
        inputs
            .into_par_iter()
            .map(|input| with_shared_ctx(|ctx| f(ctx, input)))
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<Out> =
        with_shared_ctx(|ctx| inputs.into_iter().map(|input| f(ctx, input)).collect());

    #[cfg(feature = "tracing")]
    debug!(
        duration_ms = start.elapsed().as_millis(),
        count = results.len(),
        parallel = cfg!(feature = "parallel"),
        "{stage} completed"
    );

    results
}

/// Shared body of the deprecated `fingerprint_batch_chunked` functions.
fn batch_chunked<S, F>(images: &[(S, Vec<u8>)], chunk_size: usize, mut callback: F)
where
    S: Send + Sync + Clone,
    F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>),
{
    for chunk in images.chunks(chunk_size.max(1)) {
        let refs: Vec<&(S, Vec<u8>)> = chunk.iter().collect();
        for (id, result) in run_batch(refs, "batch_chunked", |ctx, (id, bytes)| {
            (id.clone(), ctx.fingerprint(bytes))
        }) {
            callback(id, result);
        }
    }
}

/// Zero-config entry points backed by a per-thread [`FingerprinterContext`]
/// with the default [`PreprocessConfig`].
///
/// Use [`FingerprinterContext::with_config`] for custom decode guards.
///
/// # Example
///
/// ```rust,no_run
/// use imgfprint::ImageFingerprinter;
///
/// let a = ImageFingerprinter::fingerprint_path("a.jpg")?;
/// let b = ImageFingerprinter::fingerprint_path("b.jpg")?;
/// if a.is_similar(&b, 0.85) {
///     println!("near-duplicates (score {:.3})", a.compare(&b).score);
/// }
/// # Ok::<(), imgfprint::ImgFprintError>(())
/// ```
pub struct ImageFingerprinter;

impl ImageFingerprinter {
    /// Fingerprints encoded image bytes with all three algorithms.
    ///
    /// See [`FingerprinterContext::fingerprint`] for exact-hash semantics and
    /// errors.
    ///
    /// # Errors
    ///
    /// See [`FingerprinterContext::fingerprint`].
    pub fn fingerprint(image_bytes: &[u8]) -> Result<MultiHashFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.fingerprint(image_bytes))
    }

    /// Fingerprints encoded image bytes with a single algorithm; bit-identical
    /// to the matching layer of [`fingerprint`](Self::fingerprint).
    ///
    /// # Errors
    ///
    /// See [`FingerprinterContext::fingerprint`].
    pub fn fingerprint_with(
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.fingerprint_with(image_bytes, algorithm))
    }

    /// Reads a file (at most [`DEFAULT_MAX_INPUT_BYTES`](crate::DEFAULT_MAX_INPUT_BYTES))
    /// and fingerprints it with all three algorithms.
    ///
    /// # Errors
    ///
    /// See [`FingerprinterContext::fingerprint_path`].
    pub fn fingerprint_path<P: AsRef<Path>>(
        path: P,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.fingerprint_path(path))
    }

    /// Reads a file and fingerprints it with a single algorithm.
    ///
    /// # Errors
    ///
    /// See [`FingerprinterContext::fingerprint_path`].
    pub fn fingerprint_path_with<P: AsRef<Path>>(
        path: P,
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.fingerprint_path_with(path, algorithm))
    }

    /// Fingerprints an already-decoded image; the exact hash covers its RGB8
    /// pixels. See [`FingerprinterContext::fingerprint_image`].
    ///
    /// # Errors
    ///
    /// See [`FingerprinterContext::fingerprint_image`].
    pub fn fingerprint_image(image: &DynamicImage) -> Result<MultiHashFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.fingerprint_image(image))
    }

    /// Fingerprints in-memory images, returning results in input order.
    ///
    /// With the `parallel` feature (default) images are spread across rayon
    /// workers, each reusing a per-thread context.
    #[cfg_attr(feature = "tracing", instrument(skip(images), fields(image_count = images.len())))]
    pub fn fingerprint_batch<S>(
        images: &[(S, Vec<u8>)],
    ) -> Vec<(S, Result<MultiHashFingerprint, ImgFprintError>)>
    where
        S: Clone + Send + Sync,
    {
        let refs: Vec<&(S, Vec<u8>)> = images.iter().collect();
        run_batch(refs, "batch", |ctx, (id, bytes)| {
            (id.clone(), ctx.fingerprint(bytes))
        })
    }

    /// Single-algorithm variant of [`fingerprint_batch`](Self::fingerprint_batch).
    #[cfg_attr(feature = "tracing", instrument(skip(images), fields(image_count = images.len(), algorithm = ?algorithm)))]
    pub fn fingerprint_batch_with<S>(
        images: &[(S, Vec<u8>)],
        algorithm: HashAlgorithm,
    ) -> Vec<(S, Result<ImageFingerprint, ImgFprintError>)>
    where
        S: Clone + Send + Sync,
    {
        let refs: Vec<&(S, Vec<u8>)> = images.iter().collect();
        run_batch(refs, "batch_with", |ctx, (id, bytes)| {
            (id.clone(), ctx.fingerprint_with(bytes, algorithm))
        })
    }

    /// Fingerprints many files, returning `(path, result)` in input order.
    ///
    /// Files are read inside the workers, so memory stays bounded by one
    /// file per worker thread no matter how many paths are given. With the
    /// `parallel` feature (default) files are processed across rayon
    /// workers; without it, sequentially. Prefer this over
    /// [`fingerprint_batch`](Self::fingerprint_batch) for directory-scale
    /// jobs, and over [`fingerprint_stream`](Self::fingerprint_stream) when
    /// you want all cores busy.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use imgfprint::ImageFingerprinter;
    ///
    /// let paths: Vec<_> = std::fs::read_dir("photos")?
    ///     .filter_map(|entry| entry.ok().map(|e| e.path()))
    ///     .collect();
    /// for (path, result) in ImageFingerprinter::fingerprint_paths(paths) {
    ///     match result {
    ///         Ok(fp) => println!("{}: {fp}", path.display()),
    ///         Err(e) => eprintln!("{}: {e}", path.display()),
    ///     }
    /// }
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn fingerprint_paths<I, P>(
        paths: I,
    ) -> Vec<(P, Result<MultiHashFingerprint, ImgFprintError>)>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path> + Send,
    {
        run_batch(paths.into_iter().collect(), "paths", |ctx, path: P| {
            let result = ctx.fingerprint_path(path.as_ref());
            (path, result)
        })
    }

    /// Lazily fingerprints paths one at a time on the calling thread.
    ///
    /// Only one file is in memory at a time. For parallel processing use
    /// [`fingerprint_paths`](Self::fingerprint_paths).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use imgfprint::ImageFingerprinter;
    /// use std::path::PathBuf;
    ///
    /// let paths = vec![PathBuf::from("a.jpg"), PathBuf::from("b.png")];
    /// for (path, result) in ImageFingerprinter::fingerprint_stream(paths) {
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
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut ctx = FingerprinterContext::new();
        paths.into_iter().map(move |p| {
            let result = ctx.fingerprint_path(p.as_ref());
            (p, result)
        })
    }

    /// Fingerprints with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config).fingerprint(bytes)`"
    )]
    pub fn fingerprint_with_preprocess(
        image_bytes: &[u8],
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.compute_all_hashes(image_bytes, preprocess))
    }

    /// Fingerprints a file with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config).fingerprint_path(path)`"
    )]
    pub fn fingerprint_path_with_preprocess<P: AsRef<Path>>(
        path: P,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        let bytes = read_image_file(path.as_ref(), preprocess)?;
        with_shared_ctx(|ctx| ctx.compute_all_hashes(&bytes, preprocess))
    }

    /// Fingerprints a decoded image with a one-off [`PreprocessConfig`].
    #[deprecated(
        since = "0.4.7",
        note = "use `FingerprinterContext::with_config(config).fingerprint_image(image)`"
    )]
    pub fn fingerprint_image_with_preprocess(
        image: &DynamicImage,
        preprocess: &PreprocessConfig,
    ) -> Result<MultiHashFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| ctx.compute_image_hashes(image, preprocess))
    }

    /// Single-algorithm fingerprint using a bilinear resize for AHash/DHash.
    ///
    /// Output is **not** bit-compatible with any other entry point.
    #[deprecated(
        since = "0.4.7",
        note = "bit-incompatible with every other entry point for a small end-to-end gain (decode dominates); use `fingerprint_with`"
    )]
    pub fn fingerprint_with_fast(
        image_bytes: &[u8],
        algorithm: HashAlgorithm,
    ) -> Result<ImageFingerprint, ImgFprintError> {
        with_shared_ctx(|ctx| {
            let config = ctx.config;
            ctx.compute_single_hash(image_bytes, algorithm, &config, true)
        })
    }

    /// Compares two single-algorithm fingerprints.
    #[deprecated(since = "0.4.7", note = "use `a.compare(&b)`")]
    #[must_use]
    pub fn compare(a: &ImageFingerprint, b: &ImageFingerprint) -> similarity::Similarity {
        similarity::compute_similarity(a, b)
    }

    /// Generates a semantic embedding via `provider`.
    #[deprecated(since = "0.4.7", note = "call `provider.embed(image)` directly")]
    pub fn semantic_embedding<P: crate::embed::EmbeddingProvider>(
        provider: &P,
        image: &[u8],
    ) -> Result<crate::embed::Embedding, ImgFprintError> {
        provider.embed(image)
    }

    /// Cosine similarity of two embeddings.
    #[deprecated(
        since = "0.4.7",
        note = "use the free function `imgfprint::semantic_similarity`"
    )]
    pub fn semantic_similarity(
        a: &crate::embed::Embedding,
        b: &crate::embed::Embedding,
    ) -> Result<f32, ImgFprintError> {
        crate::embed::semantic_similarity(a, b)
    }

    /// Fingerprints `images` and invokes `callback` for each result in input
    /// order.
    #[deprecated(
        since = "0.4.7",
        note = "the inputs are already in memory, so chunking bounds nothing; use `fingerprint_batch`, or `fingerprint_paths` for bounded memory"
    )]
    pub fn fingerprint_batch_chunked<S, F>(images: &[(S, Vec<u8>)], chunk_size: usize, callback: F)
    where
        S: Send + Sync + Clone + 'static,
        F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>),
    {
        batch_chunked(images, chunk_size, callback);
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
    fn shared_context_survives_reentrancy() {
        // A nested borrow (what rayon work-stealing used to trigger inside
        // the old per-image `rayon::join`) must fall back, not panic.
        let img = create_test_image(64, 64);
        let (outer, inner) = with_shared_ctx(|ctx| {
            let inner = ImageFingerprinter::fingerprint(&img).unwrap();
            (ctx.fingerprint(&img).unwrap(), inner)
        });
        assert_eq!(outer, inner);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn static_api_is_safe_inside_rayon_workers() {
        use rayon::prelude::*;
        // Regression: 0.4.6 panicked with "RefCell already borrowed" here.
        let img = create_test_image(96, 96);
        let expected = ImageFingerprinter::fingerprint(&img).unwrap();
        let all: Vec<_> = (0..128)
            .into_par_iter()
            .map(|_| ImageFingerprinter::fingerprint(&img).unwrap())
            .collect();
        assert!(all.iter().all(|fp| *fp == expected));
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
    #[allow(deprecated)] // covers the deprecated shim until removal
    fn test_fingerprinter_batch_chunked_empty() {
        let images: Vec<(usize, Vec<u8>)> = vec![];
        let mut results = Vec::new();

        ImageFingerprinter::fingerprint_batch_chunked(&images, 2, |id, result| {
            results.push((id, result));
        });

        assert_eq!(results.len(), 0);
    }

    #[test]
    #[allow(deprecated)] // covers the deprecated shim until removal
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
    #[allow(deprecated)] // covers the deprecated shim until removal
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
    #[allow(deprecated)] // covers the deprecated shim until removal
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
    #[allow(deprecated)] // covers the deprecated shim until removal
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
    #[allow(deprecated)] // covers the deprecated shim until removal
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
    #[allow(deprecated)] // covers the deprecated shim until removal
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

        let sim = fp1.compare(&fp2);
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
        let err = FingerprinterContext::with_config(tight)
            .fingerprint_path(&path)
            .unwrap_err();
        assert!(matches!(err, ImgFprintError::IoError(_)), "got: {:?}", err);
        #[allow(deprecated)]
        let legacy = ImageFingerprinter::fingerprint_path_with_preprocess(&path, &tight);
        assert!(matches!(legacy, Err(ImgFprintError::IoError(_))));

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
        let results: Vec<_> = ImageFingerprinter::fingerprint_stream(paths).collect();

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
        let results: Vec<_> = ImageFingerprinter::fingerprint_stream(paths).collect();
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
