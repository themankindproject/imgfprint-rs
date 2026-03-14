# imgfprint

[![Crates.io](https://img.shields.io/crates/v/imgfprint)](https://crates.io/crates/imgfprint)
[![Documentation](https://docs.rs/imgfprint/badge.svg)](https://docs.rs/imgfprint)
[![License](https://img.shields.io/crates/l/imgfprint)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/themankindproject/imgfprint-rs/main.yml)](https://github.com/themankindproject/imgfprint-rs/actions)
![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue)

High-performance image fingerprinting library for Rust with **multi-algorithm perceptual hashing**, **exact matching**, and **semantic embeddings**.

## Overview

`imgfprint` provides multiple complementary approaches to image identification and similarity detection:

| Method | Use Case | Speed | Precision |
|--------|----------|-------|-----------|
| **BLAKE3** | Exact deduplication | ~0.2ms | 100% exact |
| **PHash** | Perceptual similarity | ~1.5ms | DCT-based, resilient to compression |
| **DHash** | Structural similarity | ~0.5ms | Gradient-based, good for crops |
| **Multi** | Combined accuracy | ~1.8ms | Weighted PHash+DHash (60/40) |
| **Semantic** | Content understanding | Local or API | Captures visual meaning |

Perfect for:
- Duplicate image detection
- Similarity search
- Content moderation
- Image deduplication at scale
- Content-based image retrieval (CBIR)

## Features

- **Multi-Algorithm Support** - PHash (DCT-based) + DHash (gradient-based) with weighted combination
- **Deterministic Output** - Same input always produces same fingerprint
- **BLAKE3 Exact Hash** - Byte-identical detection (6-8x faster than SHA256)
- **Block-Level Hashing** - 4x4 grid for crop resistance
- **Semantic Embeddings** - CLIP-style vector representations via external providers or local ONNX models
- **SIMD Acceleration** - AVX2/NEON optimized resizing
- **Parallel Processing** - Multi-core batch operations
- **Zero-Copy APIs** - Minimal allocations in hot paths
- **Serde Support** - JSON/binary serialization
- **Security Hardened** - OOM protection (8192px max), no panics on malformed input
- **Multiple Formats** - PNG, JPEG, GIF, WebP, BMP

## Installation

```toml
[dependencies]
imgfprint = "0.2.0"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | Yes | Serialization support (JSON, binary) |
| `parallel` | Yes | Parallel batch processing with rayon |
| `local-embedding` | No | Local ONNX model inference for semantic embeddings |

Minimal build (no parallel processing):
```toml
[dependencies]
imgfprint = { version = "0.2.0", default-features = false }
```

With local embeddings (requires ONNX model):
```toml
[dependencies]
imgfprint = { version = "0.2.0", features = ["local-embedding"] }
```

## Quick Start

```rust
use imgfprint::ImageFingerprinter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img1 = std::fs::read("photo1.jpg")?;
    let img2 = std::fs::read("photo2.jpg")?;
    
    // Compute all hashes (PHash + DHash) for best accuracy
    let fp1 = ImageFingerprinter::fingerprint(&img1)?;
    let fp2 = ImageFingerprinter::fingerprint(&img2)?;
    
    let sim = fp1.compare(&fp2);
    
    println!("Similarity: {:.2}", sim.score);
    println!("Exact match: {}", sim.exact_match);
    
    if sim.score > 0.8 {
        println!("Images are perceptually similar");
    }
    
    Ok(())
}
```

### Single Algorithm Mode

```rust
use imgfprint::{ImageFingerprinter, HashAlgorithm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img = std::fs::read("photo.jpg")?;
    
    // Use specific algorithm for better speed
    let fp = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::DHash)?;
    
    Ok(())
}
```

## Documentation

For complete API reference and usage examples, see [USAGE.md](USAGE.md).

## Architecture

### Fingerprint Types

#### MultiHashFingerprint (Default)

Contains both PHash and DHash for enhanced accuracy:

```
MultiHashFingerprint
├── exact:       [u8; 32]     // BLAKE3 of original bytes
├── phash:       ImageFingerprint  // PHash results
│   ├── global_hash: u64
│   └── block_hashes: [u64; 16]
└── dhash:       ImageFingerprint  // DHash results
    ├── global_hash: u64
    └── block_hashes: [u64; 16]
```

#### Single Algorithm Mode

```
ImageFingerprint
├── exact:       [u8; 32]     // BLAKE3 of original bytes
├── global_hash: u64         // Algorithm-specific hash (center 32x32)
└── block_hashes: [u64; 16]   // Block-level hashes (4x4 grid, 64x64 each)
```

### Algorithm Pipeline

1. **Decode** - Parse any supported format (PNG, JPEG, GIF, WebP, BMP) into RGB
2. **Normalize** - Resize to 256x256 using SIMD-accelerated Lanczos3 filter
3. **Convert** - RGB to Grayscale (luminance)
4. **Parallel Hash Computation** - Both algorithms computed simultaneously:
   - **PHash**: DCT-based, center 32x32 + 4x4 blocks
   - **DHash**: Gradient-based, resample to 9x8
5. **Exact Hash** - BLAKE3 of original bytes

### Multi-Algorithm Comparison

When using `MultiHashFingerprint`, the similarity score uses weighted combination:

- **60%** PHash similarity (DCT-based, robust to compression)
- **40%** DHash similarity (gradient-based, good for structural changes)

This provides better accuracy than any single algorithm alone.

## Performance

Benchmarked on Intel i5 11th gen (16 GB RAM , 4 cores 8 threads):

| Operation | Time | Throughput |
|-----------|------|------------|
| `fingerprint()` | **1.35ms** | ~740 images/sec |
| `compare()` | **385ns** | 2.6B comparisons/sec |
| `batch()` (10 images) | **6.16ms** | 1,620 images/sec (parallel) |
| `semantic_similarity()` | ~500ns | 2M comparisons/sec |

Run benchmarks:
```bash
cargo bench
```

### Memory Safety

- Maximum image dimension: 8192x8192 (OOM protection)
- Dimension check before full decode
- Pre-allocated buffers in context API
- Zero-copy where possible

## Security

- **OOM Protection**: Maximum image size 8192x8192 pixels (configurable)
- **Deterministic Output**: Same input always produces same output
- **No Panics**: All error conditions return `Result`
- **Fast Hashing**: BLAKE3 computation (6-8x faster than SHA256)
- **Input Validation**: Comprehensive format and size validation

## Comparison with Alternatives

| Feature | imgfprint-rs | imagehash | img_hash |
|---------|-------------|-----------|----------|
| BLAKE3 exact | Yes | No | No |
| PHash | Yes | Yes | Yes |
| DHash | Yes | Yes | Yes |
| Multi-algorithm | Yes | No | No |
| Block hashes | Yes | No | No |
| Semantic embeddings | Yes | No | No |
| Local ONNX inference | Yes | No | No |
| Parallel batch | Yes | No | No |
| SIMD acceleration | Yes | No | No |
| Context API | Yes | No | No |

## Examples

See the `examples/` directory for complete working examples:

- `batch_process.rs` - Process millions of images efficiently
- `compare_images.rs` - Compare two images and show similarity
- `find_duplicates.rs` - Find duplicate images in a directory
- `serialize.rs` - Serialize/deserialize fingerprints to JSON and binary
- `similarity_search.rs` - Perceptual similarity search in a directory
- `semantic_search.rs` - Content-based image search with CLIP embeddings (requires `local-embedding` feature)

Run an example:
```bash
# Compare two images
cargo run --example compare_images -- images/photo1.jpg images/photo2.jpg

# Semantic search with local CLIP model (requires local-embedding feature)
cargo run --example semantic_search --features local-embedding -- model.onnx query.jpg ./images 0.85
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests: `cargo test`
4. Run clippy: `cargo clippy --all-targets -- -D warnings`
5. Run benchmarks: `cargo bench`
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Setup

```bash
# Clone
git clone https://github.com/themankindproject/imgfprint-rs
cd imgfprint-rs

# Run tests
cargo test

# Run with all features
cargo test --all-features

# Generate documentation
cargo doc --no-deps --open
```

## License

MIT License - See LICENSE file for details.
