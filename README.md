# imgfprint-rs

[![Crates.io](https://img.shields.io/crates/v/imgfprint-rs)](https://crates.io/crates/imgfprint-rs)
[![Documentation](https://docs.rs/imgfprint-rs/badge.svg)](https://docs.rs/imgfprint-rs)
[![License](https://img.shields.io/crates/l/imgfprint-rs)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/themankindproject/imgfprint-rs/main.yml)](https://github.com/themankindproject/imgfprint-rs/actions)
![Rust Version](https://img.shields.io/badge/rust-1.70%2B-blue)

High-performance image fingerprinting library for Rust with **perceptual hashing**, **exact matching**, and **semantic embeddings**.

## Overview

`imgfprint-rs` provides multiple complementary approaches to image identification and similarity detection:

| Method | Use Case | Speed | Precision |
|--------|----------|-------|-----------|
| **SHA256** | Exact deduplication | ~1ms | 100% exact |
| **pHash** | Perceptual similarity | ~1-5ms | Resilient to compression, resizing |
| **Semantic** | Content understanding | Provider-dependent | Captures visual meaning |

Perfect for:
- Duplicate image detection
- Similarity search
- Content moderation
- Image deduplication at scale
- Content-based image retrieval (CBIR)

## Features

- **Deterministic Output** - Same input always produces same fingerprint
- **SHA256 Exact Hash** - Byte-identical detection
- **pHash Perceptual Hash** - DCT-based similarity (resilient to compression, resizing, minor edits)
- **Block-Level Hashing** - 4×4 grid for crop resistance
- **Semantic Embeddings** - CLIP-style vector representations via external providers
- **SIMD Acceleration** - AVX2/NEON optimized resizing
- **Parallel Processing** - Multi-core batch operations
- **Zero-Copy APIs** - Minimal allocations in hot paths
- **Serde Support** - JSON/binary serialization
- **Security Hardened** - OOM protection (8192px max), no panics on malformed input
- **Multiple Formats** - PNG, JPEG, GIF, WebP, BMP

## Installation

```toml
[dependencies]
imgfprint-rs = "0.1"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | ✅ | Serialization support (JSON, binary) |
| `parallel` | ✅ | Parallel batch processing with rayon |

Minimal build (no parallel processing):
```toml
[dependencies]
imgfprint-rs = { version = "0.1", default-features = false }
```

## Quick Start

### Basic Fingerprinting

```rust
use imgfprint_rs::ImageFingerprinter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img1 = std::fs::read("photo1.jpg")?;
    let img2 = std::fs::read("photo2.jpg")?;
    
    let fp1 = ImageFingerprinter::fingerprint(&img1)?;
    let fp2 = ImageFingerprinter::fingerprint(&img2)?;
    
    let sim = ImageFingerprinter::compare(&fp1, &fp2);
    
    println!("Similarity: {:.2}", sim.score);
    println!("Exact match: {}", sim.exact_match);
    
    if sim.score > 0.8 {
        println!("Images are perceptually similar");
    }
    
    Ok(())
}
```

### Semantic Embeddings

```rust
use imgfprint_rs::{ImageFingerprinter, EmbeddingProvider, Embedding, ImgFprintError};

// Implement your provider (OpenAI, HuggingFace, local model, etc.)
struct MyClipProvider;

impl EmbeddingProvider for MyClipProvider {
    fn embed(&self, image: &[u8]) -> Result<Embedding, ImgFprintError> {
        // Call your embedding API or local model
        // Return a Vec<f32> with 512-1024 dimensions
        let vector = call_your_embedding_api(image)?;
        Embedding::new(vector)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = MyClipProvider;
    
    let img1 = std::fs::read("cat.jpg")?;
    let img2 = std::fs::read("dog.jpg")?;
    
    let emb1 = ImageFingerprinter::semantic_embedding(&provider, &img1)?;
    let emb2 = ImageFingerprinter::semantic_embedding(&provider, &img2)?;
    
    // Cosine similarity in range [-1.0, 1.0]
    let similarity = ImageFingerprinter::semantic_similarity(&emb1, &emb2)?;
    
    println!("Semantic similarity: {:.4}", similarity);
    // 1.0 = identical content
    // 0.0 = unrelated content
    // -1.0 = opposite content (rare with CLIP)
    
    Ok(())
}
```

## API Reference

### Creating Fingerprints

#### Single Image

```rust
use imgfprint_rs::ImageFingerprinter;

let fp = ImageFingerprinter::fingerprint(&image_bytes)?;
```

#### Batch Processing

Process thousands of images efficiently with automatic parallelization:

```rust
use imgfprint_rs::ImageFingerprinter;

let images: Vec<(String, Vec<u8>)> = vec![
    ("img1.jpg".into(), bytes1),
    ("img2.jpg".into(), bytes2),
    // ... thousands more
];

let results = ImageFingerprinter::fingerprint_batch(&images);

for (id, result) in results {
    match result {
        Ok(fp) => println!("{}: fingerprinted", id),
        Err(e) => println!("{}: error - {}", id, e),
    }
}
```

#### High-Throughput Context

For sustained high-throughput scenarios, use `FingerprinterContext` to enable buffer reuse:

```rust
use imgfprint_rs::FingerprinterContext;

let mut ctx = FingerprinterContext::new();

for path in &image_paths {
    let bytes = std::fs::read(path)?;
    let fp = ctx.fingerprint(&bytes)?;
    // Process fingerprint...
}
```

### Accessing Fingerprint Components

```rust
// Exact SHA256 hash (32 bytes)
let exact: &[u8; 32] = fp.exact_hash();

// Global perceptual hash (center 32×32 region)
let global: u64 = fp.global_phash();

// Block hashes (16 regions in 4×4 grid)
let blocks: &[u64; 16] = fp.block_hashes();
```

### Comparing Fingerprints

#### Using Similarity Scores

```rust
use imgfprint_rs::ImageFingerprinter;

let sim = ImageFingerprinter::compare(&fp1, &fp2);

println!("Score: {:.2}", sim.score);           // 0.0 - 1.0
println!("Exact match: {}", sim.exact_match);   // true/false
println!("Hamming distance: {}", sim.perceptual_distance); // 0-64
```

#### Direct Distance Methods

```rust
// Hamming distance between global hashes (0-64)
let dist = fp1.distance(&fp2);

// Check against threshold
if fp1.is_similar(&fp2, 0.85) {
    println!("Images are similar (threshold: 0.85)");
}
```

### Semantic Embeddings

#### Creating Embeddings

```rust
use imgfprint_rs::{ImageFingerprinter, EmbeddingProvider, Embedding};

// Your provider implementation
struct HuggingFaceProvider { /* ... */ }

impl EmbeddingProvider for HuggingFaceProvider {
    fn embed(&self, image: &[u8]) -> Result<Embedding, ImgFprintError> {
        // Call HuggingFace Inference API
        let response = self.client.post("https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32")
            .body(image.to_vec())
            .send()?;
        
        let vector: Vec<f32> = response.json()?;
        Embedding::new(vector)
    }
}
```

#### Computing Cosine Similarity

```rust
use imgfprint_rs::{semantic_similarity, Embedding};

let emb1 = Embedding::new(vec![0.1, 0.2, 0.3, 0.4])?;
let emb2 = Embedding::new(vec![0.2, 0.3, 0.4, 0.5])?;

// Returns f32 in range [-1.0, 1.0]
let sim = semantic_similarity(&emb1, &emb2)?;
```

## Architecture

### Fingerprint Structure (168 bytes)

```
ImageFingerprint
├── exact:      [u8; 32]     // SHA256 of original bytes
├── global_phash: u64        // DCT-based hash (center 32×32)
└── block_hashes: [u64; 16]  // DCT hashes (4×4 grid, 64×64 each)
```

### Algorithm Pipeline

1. **Decode** - Parse any supported format (PNG, JPEG, GIF, WebP, BMP) into RGB
2. **Normalize** - Resize to 256×256 using SIMD-accelerated Lanczos3 filter
3. **Convert** - RGB → Grayscale (luminance)
4. **Global Hash** - Extract center 32×32 → DCT → pHash
5. **Block Hashes** - Split into 4×4 grid (64×64 blocks) → DCT → 16 pHashes
6. **Exact Hash** - SHA256 of original bytes

### Similarity Computation

The similarity score is a weighted combination:

- **40%** Global perceptual hash similarity
- **60%** Block-level hash similarity (crop-resistant)

Block distances > 32 are filtered (handles cropping by ignoring missing blocks).

### Embedding Validation

Embeddings are validated on creation:
- ✅ Non-empty vector
- ✅ All values finite (no NaN or infinity)
- ✅ Consistent dimensions for comparison

## Performance

Benchmarked on AMD Ryzen 9 5900X (single core unless noted):

| Operation | Time | Throughput |
|-----------|------|------------|
| `fingerprint()` | **1.35ms** | ~740 images/sec |
| `compare()` | **385ns** | 2.6B comparisons/sec |
| `batch()` (10 images) | **6.16ms** | 1,620 images/sec (parallel) |
| `semantic_similarity()` | ~500ns | 2M comparisons/sec |

> **Note**: `semantic_similarity()` performance depends on embedding dimension (typically 512-1024). The above is for 512-dim embeddings.

Run benchmarks:
```bash
cargo bench
```

### Memory Safety

- Maximum image dimension: 8192×8192 (OOM protection)
- Dimension check before full decode
- Pre-allocated buffers in context API
- Zero-copy where possible

## Error Handling

```rust
use imgfprint_rs::{ImageFingerprinter, ImgFprintError};

match ImageFingerprinter::fingerprint(&bytes) {
    Ok(fp) => { /* success */ }
    
    // Image-related errors
    Err(ImgFprintError::InvalidImage(msg)) => { /* empty or too large */ }
    Err(ImgFprintError::UnsupportedFormat(msg)) => { /* format not supported */ }
    Err(ImgFprintError::DecodeError(msg)) => { /* corrupted data */ }
    Err(ImgFprintError::ImageTooSmall(msg)) => { /* < 32×32 pixels */ }
    
    // Processing errors
    Err(ImgFprintError::ProcessingError(msg)) => { /* internal error */ }
    
    // Embedding errors (new in 0.2)
    Err(ImgFprintError::EmbeddingDimensionMismatch { expected, actual }) => { /* dimension mismatch */ }
    Err(ImgFprintError::ProviderError(msg)) => { /* external provider failed */ }
    Err(ImgFprintError::InvalidEmbedding(msg)) => { /* empty vector or NaN */ }
}
```

## Serialization

### JSON

```rust
use imgfprint_rs::ImageFingerprinter;
use serde_json;

let fp = ImageFingerprinter::fingerprint(&bytes)?;

// Serialize
let json = serde_json::to_string(&fp)?;

// Deserialize
let fp: ImageFingerprint = serde_json::from_str(&json)?;
```

### Binary

```rust
use bincode;

// Serialize
let bytes = bincode::serialize(&fp)?;

// Deserialize
let fp: ImageFingerprint = bincode::deserialize(&bytes)?;
```

## Security

- **OOM Protection**: Maximum image size 8192×8192 pixels (configurable)
- **Deterministic Output**: Same input always produces same output
- **No Panics**: All error conditions return `Result`
- **Constant-Time Hashing**: SHA256 computation
- **Input Validation**: Comprehensive format and size validation

## Comparison with Alternatives

| Feature | imgfprint-rs | imagehash | img_hash |
|---------|-------------|-----------|----------|
| SHA256 exact | ✅ | ❌ | ❌ |
| pHash | ✅ | ✅ | ✅ |
| Block hashes | ✅ | ❌ | ❌ |
| Semantic embeddings | ✅ | ❌ | ❌ |
| Parallel batch | ✅ | ❌ | ❌ |
| SIMD acceleration | ✅ | ❌ | ❌ |
| Context API | ✅ | ❌ | ❌ |

## Examples

See the `examples/` directory for complete working examples:

- `batch_process.rs` - Process millions of images efficiently
- `compare_images.rs` - Compare two images and show similarity
- `find_duplicates.rs` - Find duplicate images in a directory
- `semantic_search.rs` - Content-based image search with embeddings

Run an example:
```bash
cargo run --example compare_images -- images/photo1.jpg images/photo2.jpg
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
