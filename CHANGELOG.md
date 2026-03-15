# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/themankindproject/imgfprint-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/themankindproject/imgfprint-rs/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/themankindproject/imgfprint-rs/releases/tag/v0.1.3
