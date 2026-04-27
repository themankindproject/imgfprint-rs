# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-04-27

### Added

- **`MultiHashConfig` for fully tunable similarity scoring**: Lets integrators (UCFP, downstream pipelines) shift trade-offs without forking the crate. Knobs:
  - Inter-algorithm weights: `ahash_weight`, `phash_weight`, `dhash_weight` (defaults 0.10 / 0.60 / 0.30 reproduce the previous hardcoded blend)
  - Intra-algorithm weights: `global_weight`, `block_weight` (defaults 0.40 / 0.60)
  - `block_distance_threshold` (default 32 of 64)

  Setting any algorithm weight to `0.0` removes it from the score — covers the "skip this hash" use case without restructuring `MultiHashFingerprint`. Score is clamped to `[0.0, 1.0]`.

  New API: `MultiHashFingerprint::compare_with_config(&self, other, &MultiHashConfig)`. Existing `compare()` and `compare_with_threshold()` are preserved as defaulted wrappers — no behavior change for existing callers. `DEFAULT_*_WEIGHT` and `DEFAULT_BLOCK_DISTANCE_THRESHOLD` constants are re-exported.

- **`PreprocessConfig` for tunable decode-time guards**: Replaces the hardcoded `MAX_INPUT_BYTES` (50 MiB) / `MAX_DIMENSION` (8192) / `MIN_DIMENSION` (32) constants. Tighten on untrusted ingest paths, widen for trusted batch jobs.

  New APIs accept `&PreprocessConfig` on both the bytes and the path entry points:
  - `decode_image_with_config(bytes, &cfg)` — also re-exported at crate root
  - `ImageFingerprinter::fingerprint_with_preprocess(bytes, &cfg)`
  - `ImageFingerprinter::fingerprint_path_with_preprocess(path, &cfg)`
  - `FingerprinterContext::fingerprint_with_preprocess(&mut self, bytes, &cfg)`
  - `FingerprinterContext::fingerprint_path_with_preprocess(&mut self, path, &cfg)`
  - `FingerprinterContext::fingerprint_with_algorithm_and_preprocess(&mut self, bytes, algo, &cfg)`

  The same config gates **both** the pre-read file-size guard and the decode-time dimension/byte guards, so the tightened limits aren't silently bypassed via the path API. `DEFAULT_MAX_INPUT_BYTES`, `DEFAULT_MAX_DIMENSION`, `DEFAULT_MIN_DIMENSION` constants are re-exported.

- **`compute_similarity_with_weights`**: New public function in `core::similarity` exposing the global/block weight split for callers that work with raw `ImageFingerprint`s.

- **`Hash` derive on fingerprint types**: `ImageFingerprint` and `MultiHashFingerprint` now derive `Hash`, enabling their use as `HashMap` / `HashSet` keys for deduplication workflows. All fields are stack-allocated integer arrays, so the derive is a pure structural addition with no runtime cost.

- **`fingerprint_path` convenience methods**: New methods on both `ImageFingerprinter` (static) and `FingerprinterContext` accept an `AsRef<Path>` and handle the file read internally. Variants:
  - `ImageFingerprinter::fingerprint_path(path)` — multi-algorithm
  - `ImageFingerprinter::fingerprint_path_with(path, algorithm)` — single-algorithm
  - `FingerprinterContext::fingerprint_path(&mut self, path)` — multi-algorithm with buffer reuse
  - `FingerprinterContext::fingerprint_path_with(&mut self, path, algorithm)` — single-algorithm with buffer reuse

  File size is validated via `metadata().len()` against the configured `max_input_bytes` *before* any read happens, so oversized files are rejected without being pulled into memory.

- **`ImgFprintError::IoError` variant**: New error variant for I/O failures (missing files, unreadable paths, oversized inputs). Includes a `From<std::io::Error>` impl. Backwards-compatible because `ImgFprintError` is `#[non_exhaustive]`.

### Changed

- **`compare_with_threshold` is now a wrapper over `compare_with_config`** — same scores, same defaults, simpler internals.

### Removed

- Internal `MAX_INPUT_BYTES`, `MAX_DIMENSION`, `MIN_DIMENSION` constants in `imgproc::decode`. Their `DEFAULT_*` public successors carry the same values; behavior is unchanged for callers who never imported the privates.

### Notes for UCFP integrators

Per-algorithm DCT/grid/hash-bit reconfiguration (different `dct_size`, `block_grid`, `hash_bits`) is **not** in 0.4.0 — those would require turning the static `[f32; 32*32]` and `[[f32; 64*64]; 16]` buffers into dynamically-sized vectors through every hash function and the preprocessor. That's a 0.5.0 surgery.

## [0.3.3] - 2026-03-28

### Fixed

- **Double image decode in `decode_image()`**: Eliminated redundant full image decode when checking dimensions. Previously `into_dimensions()` followed by `load_from_memory()` decoded the image twice. Now dimension checking uses the same reader without discarding it.
- **Post-EXIF dimension validation**: Added dimension re-validation after applying EXIF orientation transforms. A 9000x100 image rotated 90° now correctly rejected instead of silently producing out-of-limit dimensions.
- **Per-item context allocation in parallel batch**: `fingerprint_batch` and `fingerprint_batch_with` with the `parallel` feature were creating a new `FingerprinterContext` per image instead of per rayon task. Now uses `map_init` to cache context per worker, reusing preprocessor and hasher across images.
- **NaN/infinity threshold handling in `is_similar`**: Both `ImageFingerprint::is_similar()` and `MultiHashFingerprint::is_similar()` now clamp threshold to [0.0, 1.0] before comparison. Previously, NaN thresholds silently returned incorrect results in release builds (where `debug_assert!` is stripped).
- **Bilinear resample identity case**: Added fast-path `copy_from_slice` when source and destination dimensions are identical, skipping unnecessary floating-point interpolation.

### Changed

- **Performance optimizations**:
  - Use hardware POPCNT via `count_ones()` for Hamming distance (replaces byte-level lookup table)
  - Cache ONNX `RunnableModel` in `LocalProvider` to eliminate per-inference model clone + optimization overhead
  - Removed redundant O(n) finiteness re-validation in `semantic_similarity()` (invariant already guaranteed by `Embedding::new()`)
  - Removed `#[repr(align(64))]` from `ImageFingerprint` and `MultiHashFingerprint` (32 and 192 bytes wasted padding respectively)

- **API improvements**:
  - Added `compute_block_similarity_with_threshold()` for configurable block distance threshold
  - Added `compute_similarity_with_threshold()` for custom block matching strictness
  - Added `MultiHashFingerprint::compare_with_threshold()` for threshold-aware comparison

- **Code quality**:
  - Removed dead code: no-op `apply_orientation()` wrapper in preprocess (EXIF handled in decode module)
  - Renamed `rgb_to_grayscale_simd` to `rgb_to_grayscale` (was not actual SIMD, misleading name)
  - Deduplicated `ImageFingerprinter::fingerprint_batch_chunked` (now delegates to `FingerprinterContext`)
  - Fixed clippy warnings (type_complexity)
  - Added SAFETY documentation for `unsafe set_cpu_extensions()` block explaining CPU feature detection precondition
  - Improved DCT `expect()` messages with specific failure context
  - Clarified `#[allow(dead_code)]` annotation on public `compute_block_similarity()` convenience function
  - Added `#[must_use]` to constructor and accessor methods (`FingerprinterContext::new()`, `HashAlgorithm::hash_bits()`, `HashAlgorithm::max_distance()`, `LocalProviderConfig` presets)
  - Added `Eq` derive to `ImageFingerprint` and `MultiHashFingerprint` (all fields are `Eq`-compatible)
  - Downgraded `#[inline(always)]` to `#[inline]` on `hash_similarity()` and `hamming_distance()` to respect compiler heuristics
  - Added numeric separators to CLIP normalization constants for readability

## [0.3.2] - 2025-03-18

### Changed

- **Performance optimizations**:
  - Cache-line aligned fingerprint structs (64-byte) for CPU cache efficiency
  - Pre-computed popcount lookup table for faster Hamming distance
  - Shared bilinear resampling across AHash, DHash, PHash (removed 3 duplicate implementations)
  - SIMD-friendly chunked grayscale conversion
  - Stack-allocated buffers in PHash DCT (removed RefCell overhead)
  - Use `total_cmp` for f32 comparison (faster than partial_cmp)

- **Bug fixes**:
  - Fixed buffer reclamation order bug in image preprocessing
  - Fixed `apply_orientation()` unnecessarily cloning entire image - now uses `Cow::Borrowed` to avoid expensive clone when no EXIF orientation transform is needed
  - Fixed SIMD CPU extension detection being performed on every `Preprocessor::new()` call - now cached globally using `OnceLock` for better performance
  - **Fixed DCT-II implementation** in PHash: Added proper twiddle factors for mathematically correct DCT-II computation
  - **Fixed duplicate thread-local contexts**: Both `fingerprint()` and `fingerprint_with()` now share a single module-level TLS slot instead of creating separate ones
  - **Fixed unsafe memory operations**: Replaced `unsafe { set_len() }` with safe `resize()` in preprocessing buffers
  - **Fixed EXIF orientation handling**: Now reads EXIF orientation from raw bytes before decoding and applies transformations (rotate/flip) to decoded image
  - **Fixed similarity algorithm consistency**: `MultiHashFingerprint::compare()` now includes block-level similarity (60% weight) for consistent crop resistance with single-algorithm mode
  - **Fixed error handling in DCT**: Changed `unwrap()` to proper `Result` propagation in `dct2_32()`

- **API improvements**:
  - Made `hash_similarity()` and `hamming_distance()` public utilities
  - Added fast-path for exact match in similarity comparison
  - Added compile-time assertions verifying `ImgFprintError` implements `Send + Sync` for safe use in async and multi-threaded contexts
  - **Added model ID support to embeddings**: `Embedding::new_with_model()` allows tagging embeddings with model identifiers to prevent comparing incompatible models
  - **Fixed serde feature gating**: `serde` crate is now properly optional (not compiled unless feature enabled)

- **Code quality**:
  - Fixed clippy warnings (needless_range_loop, manual_clamp)
  - Added `kamadak-exif` dependency for EXIF orientation parsing
  - Updated bincode dev-dependency comment documenting v2 API changes

## [0.3.1] - 2025-03-16

### Added

- **New examples**:
  - `single_algorithm.rs` - Demonstrates using specific hash algorithms
  - `error_handling.rs` - Shows error type pattern matching

- **CONTRIBUTING.md**: Added semver policy documentation
  - Versioning scheme (MAJOR.MINOR.PATCH)
  - Breaking change definitions
  - Deprecation policy (one minor version before removal)
  - MSRV policy

- **Observability**: Added optional `tracing` feature for production debugging
  - `#[instrument]` attributes on all fingerprint methods
  - Timing spans for performance monitoring
  - Batch processing metrics (processed/failed counts)
  - Enable with `cargo build --features tracing`

### Changed

- **Improved API consistency**:
  - Removed misleading `Default` derive from `HashAlgorithm`
  - Added `Debug` derive to `FingerprinterContext` and `Preprocessor`
  - Added module-level documentation to `core`, `hash`, and `imgproc` modules

- **Improved error handling**:
  - Standardized error constructors in `decode.rs`
  - Added `image_too_small()` helper constructor
  - Added detailed `## Errors` sections to each `ImgFprintError` variant with trigger conditions

- **Improved test coverage**:
  - Added 13 edge case tests covering non-square images, boundary sizes, color spaces, and all supported formats

- **Improved embedding safety**:
  - Added NaN/infinity re-validation in `semantic_similarity()`
  - Added performance note to `Embedding::vector()` for O(n) clone warning

## [0.3.0] - 2025-03-15

### Added

- **AHash (Average Hash) implementation**:
  - New `HashAlgorithm::AHash` variant
  - Average-based perceptual hash algorithm
  - Bilinear resampling from 32x32 to 8x8
  - 64-bit hash output with 16 block-level hashes
  - Fastest algorithm (~0.3ms)

- **Three-algorithm fingerprinting**: Compute AHash, PHash, and DHash simultaneously
  - Weighted similarity: 10% AHash + 60% PHash + 30% DHash
  - New `MultiHashFingerprint::ahash()` accessor
  - Parallel computation of all three algorithms

- **Chunked batch processing**: New `fingerprint_batch_chunked()` method
  - Processes images in configurable chunks to limit memory usage
  - Callback-based API for immediate result processing
  - Available on both `ImageFingerprinter` and `FingerprinterContext`
  - Prevents unbounded memory consumption when processing large batches

### Changed

- Replaced SHA2 with BLAKE3 for exact hashing
  - 6-8x faster hashing performance
  - Still provides 32-byte exact match hashes

- **BREAKING**: Renamed `global_phash()` to `global_hash()` in `ImageFingerprint`
  - More accurately reflects that the method returns whichever algorithm was used (PHash or DHash)
  - Updated all documentation and examples accordingly

## [0.2.0] - 2025-03-13

### Added

- **Multi-algorithm fingerprinting support**: Compute both PHash and DHash simultaneously
  - New `HashAlgorithm` enum with `PHash` and `DHash` variants
  - New `MultiHashFingerprint` struct containing both algorithm results
  - Parallel computation of PHash and DHash using rayon for optimal performance
  - Weighted similarity comparison (60% PHash + 40% DHash) for improved accuracy

- **DHash (Difference Hash) implementation**:
  - Gradient-based perceptual hash algorithm
  - Bilinear resampling from 32x32 to 9x8
  - 64-bit hash output with 16 block-level hashes
  - Faster than PHash (~0.5ms vs ~1.5ms)

- **Single algorithm mode**:
  - `ImageFingerprinter::fingerprint_with(algorithm)` for computing specific algorithm only
  - More efficient when only one hash type is needed

- **Batch processing improvements**:
  - `fingerprint_batch()` now returns `MultiHashFingerprint` for each image
  - New `fingerprint_batch_with(algorithm)` for batch processing with specific algorithm

- **Comprehensive test coverage**:
  - Tests for multi-algorithm determinism
  - Tests for single algorithm mode
  - Tests for weighted comparison
  - Tests for algorithm accessors

- **Updated fuzz targets**:
  - `algorithm_fuzz` - Tests individual PHash and DHash algorithms
  - `batch_fuzz` - Tests batch processing
  - Updated existing targets for new API

### Changed

- **BREAKING**: `ImageFingerprinter::fingerprint()` now returns `MultiHashFingerprint` instead of `ImageFingerprint`
  - Migration: Use `fingerprint_with(HashAlgorithm::PHash)` for old behavior
  - Or use `fingerprint().phash()` to get PHash results from multi-hash

- **BREAKING**: `fingerprint_batch()` now returns `Vec<(S, Result<MultiHashFingerprint, ImgFprintError>)>`
  - Migration: Use `fingerprint_batch_with(HashAlgorithm::PHash)` for old behavior

- Updated all examples to use new multi-algorithm API
- Updated documentation with multi-algorithm examples and guides
- Updated benchmarks to compare multi vs single algorithm performance

### Removed

- **BREAKING**: Removed `src/hash/blocks.rs` - generic block hashing function
  - Block hashing is now handled per-algorithm inline in the fingerprinter
  - This was an internal API not exposed publicly

## [0.1.3] - Previous Release

### Features

- SHA256 exact hash for byte-identical detection
- PHash (Perceptual Hash) using DCT-based algorithm
- Block-level hashing with 4x4 grid for crop resistance
- SIMD-accelerated image resizing (AVX2/NEON)
- Parallel batch processing with rayon
- Context API for high-throughput scenarios
- Serde support for JSON/binary serialization
- Semantic embeddings via external providers
- Local ONNX inference (optional feature)

[Unreleased]: https://github.com/themankindproject/imgfprint-rs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/themankindproject/imgfprint-rs/compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com/themankindproject/imgfprint-rs/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/themankindproject/imgfprint-rs/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/themankindproject/imgfprint-rs/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/themankindproject/imgfprint-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/themankindproject/imgfprint-rs/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/themankindproject/imgfprint-rs/releases/tag/v0.1.3
