# imgfprint Usage Guide

> Complete API reference and examples for high-performance image fingerprinting

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [ImageFingerprinter](#imagefingerprinter)
    - [Multi-Algorithm Mode (Default)](#multi-algorithm-mode-default)
    - [Single Algorithm Mode](#single-algorithm-mode)
  - [FingerprinterContext](#fingerprintercontext)
  - [ImageFingerprint](#imagefingerprint)
  - [MultiHashFingerprint](#multihashfingerprint)
  - [Similarity](#similarity)
- [Advanced Usage](#advanced-usage)
  - [Batch Processing](#batch-processing)
  - [Semantic Embeddings](#semantic-embeddings)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)
- [Feature Flags](#feature-flags)

---

## Quick Start

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
imgfprint = "0.3.1"
```

### Basic Example (Multi-Algorithm)

Compare three images using AHash, PHash, and DHash for best accuracy:

```rust
use imgfprint::ImageFingerprinter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load images
    let img1 = std::fs::read("photo1.jpg")?;
    let img2 = std::fs::read("photo2.jpg")?;
    
    // Generate fingerprints (computes AHash, PHash, and DHash in parallel)
    let fp1 = ImageFingerprinter::fingerprint(&img1)?;
    let fp2 = ImageFingerprinter::fingerprint(&img2)?;
    
    // Compare using weighted combination (10% AHash, 60% PHash, 30% DHash)
    let sim = fp1.compare(&fp2);
    
    println!("Similarity: {:.2}", sim.score);
    println!("Exact match: {}", sim.exact_match);
    
    if sim.score > 0.8 {
        println!("Images are perceptually similar");
    }
    
    Ok(())
}
```

### Single Algorithm Example

For better performance when you only need one algorithm:

```rust
use imgfprint::{ImageFingerprinter, HashAlgorithm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let img1 = std::fs::read("photo1.jpg")?;
    let img2 = std::fs::read("photo2.jpg")?;
    
    // Use specific algorithm for better speed
    let fp1 = ImageFingerprinter::fingerprint_with(&img1, HashAlgorithm::AHash)?;
    let fp2 = ImageFingerprinter::fingerprint_with(&img2, HashAlgorithm::AHash)?;
    
    let sim = ImageFingerprinter::compare(&fp1, &fp2);
    
    println!("Similarity: {:.2}", sim.score);
    
    Ok(())
}
```

---

## Core API

### ImageFingerprinter

Static methods for computing and comparing fingerprints. The primary entry point for most use cases.

#### Multi-Algorithm Mode (Default)

##### `fingerprint()`

```rust
pub fn fingerprint(
    image_bytes: &[u8]
) -> Result<MultiHashFingerprint, ImgFprintError>
```

Computes **AHash, PHash, and DHash** in parallel for maximum accuracy.

**Returns:**
- `MultiHashFingerprint` containing:
  - BLAKE3 hash of original bytes (exact matching)
  - AHash results: global + 16 block hashes
  - PHash results: global + 16 block hashes
  - DHash results: global + 16 block hashes

**Example:**
```rust
use imgfprint::ImageFingerprinter;

let image_bytes = std::fs::read("image.png")?;
let fingerprint = ImageFingerprinter::fingerprint(&image_bytes)?;

// Access individual algorithms
let ahash = fingerprint.ahash();
let phash = fingerprint.phash();
let dhash = fingerprint.dhash();
```

#### Single Algorithm Mode

##### `fingerprint_with()`

```rust
pub fn fingerprint_with(
    image_bytes: &[u8],
    algorithm: HashAlgorithm
) -> Result<ImageFingerprint, ImgFprintError>
```

Computes a single perceptual hash using the specified algorithm.

**Algorithms:**
- `HashAlgorithm::AHash` - Average Hash (fastest, simplest)
- `HashAlgorithm::PHash` - DCT-based (robust to compression)
- `HashAlgorithm::DHash` - Gradient-based (good for structural changes)

**Example:**
```rust
use imgfprint::{ImageFingerprinter, HashAlgorithm};

// Use only AHash (fastest)
let fp = ImageFingerprinter::fingerprint_with(
    &image_bytes,
    HashAlgorithm::AHash
)?;
```

#### `compare()`

```rust
pub fn compare(
    a: &ImageFingerprint,
    b: &ImageFingerprint
) -> Similarity
```

Compares two fingerprints and returns a similarity score.

**Note:** For `MultiHashFingerprint`, use the `compare()` method directly on the struct for weighted multi-algorithm comparison.

**Returns:**
- `Similarity` struct with:
  - `score`: 0.0 (different) to 1.0 (identical)
  - `exact_match`: true if BLAKE3 hashes match
  - `perceptual_distance`: Hamming distance 0-64

**Example:**
```rust
let sim = ImageFingerprinter::compare(&fp1, &fp2);

println!("Score: {:.2}", sim.score);              // 0.0 - 1.0
println!("Exact match: {}", sim.exact_match);      // true/false
println!("Distance: {}", sim.perceptual_distance); // 0-64
```

**Similarity Thresholds:**

| Score Range | Interpretation |
|-------------|----------------|
| 1.0 | Identical images |
| 0.8 - 1.0 | Same image, minor edits |
| 0.5 - 0.8 | Visually similar |
| < 0.5 | Different images |

#### `fingerprint_batch()`

```rust
pub fn fingerprint_batch<S>(
    images: &[(S, Vec<u8>)]
) -> Vec<(S, Result<ImageFingerprint, ImgFprintError>)>
where
    S: Send + Sync + Clone + 'static
```

Processes multiple images in parallel (requires `parallel` feature).

**Example:**
```rust
let images = vec![
    ("img1.jpg".to_string(), std::fs::read("img1.jpg")?),
    ("img2.jpg".to_string(), std::fs::read("img2.jpg")?),
    ("img3.jpg".to_string(), std::fs::read("img3.jpg")?),
];

let results = ImageFingerprinter::fingerprint_batch(&images);

for (id, result) in results {
    match result {
        Ok(fp) => println!("{}: fingerprinted successfully", id),
        Err(e) => println!("{}: error - {}", id, e),
    }
}
```

**Performance Note:** With the `parallel` feature enabled (default), batch processing uses all available CPU cores for 2-3x speedup.

#### `fingerprint_batch_chunked()`

```rust
pub fn fingerprint_batch_chunked<S, F>(
    images: &[(S, Vec<u8>)],
    chunk_size: usize,
    callback: F
)
where
    S: Send + Sync + Clone + 'static,
    F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>)
```

Processes multiple images in chunks to limit memory usage. Invokes the callback for each result as it completes.

**Arguments:**
- `images` - Slice of (id, image_bytes) pairs
- `chunk_size` - Number of images to process per chunk
- `callback` - Function called with each result

**Example:**
```rust
let images = vec![
    ("img1.jpg".to_string(), std::fs::read("img1.jpg")?),
    ("img2.jpg".to_string(), std::fs::read("img2.jpg")?),
    // ... thousands of images
];

let mut processed = 0;
ImageFingerprinter::fingerprint_batch_chunked(&images, 100, |id, result| {
    match result {
        Ok(fp) => println!("{}: fingerprinted", id),
        Err(e) => println!("{}: error - {}", id, e),
    }
    processed += 1;
});

println!("Processed {} images", processed);
```

**Use case:** Prevents unbounded memory consumption when processing large batches (e.g., 10,000+ images).

---

### FingerprinterContext

High-performance context for repeated fingerprinting with buffer reuse. Recommended for processing thousands of images.

#### `new()`

```rust
pub fn new() -> Self
```

Creates a new context with cached resources.

**Example:**
```rust
use imgfprint::FingerprinterContext;

let mut ctx = FingerprinterContext::new();
```

#### `fingerprint()`

```rust
pub fn fingerprint(
    &mut self,
    image_bytes: &[u8]
) -> Result<MultiHashFingerprint, ImgFprintError>
```

Computes **all hashes** (AHash + PHash + DHash) in parallel using internal buffer reuse.

**Example:**
```rust
use imgfprint::FingerprinterContext;

let mut ctx = FingerprinterContext::new();

// Process thousands of images efficiently
for path in image_paths {
    let bytes = std::fs::read(path)?;
    let fp = ctx.fingerprint(&bytes)?;
    // fp is MultiHashFingerprint
}
```

**Performance:** Context API saves approximately 260KB of allocations per image compared to the static API. Essential for high-throughput scenarios (10M+ images/day).

#### `fingerprint_with()`

```rust
pub fn fingerprint_with(
    &mut self,
    image_bytes: &[u8],
    algorithm: HashAlgorithm
) -> Result<ImageFingerprint, ImgFprintError>
```

Computes a **single algorithm** fingerprint using internal buffer reuse.

**Example:**
```rust
use imgfprint::{FingerprinterContext, HashAlgorithm};

let mut ctx = FingerprinterContext::new();

// Process only DHash for speed
for path in image_paths {
    let bytes = std::fs::read(path)?;
    let fp = ctx.fingerprint_with(&bytes, HashAlgorithm::DHash)?;
    // Process fingerprint...
}
```

#### `fingerprint_batch_chunked()`

```rust
pub fn fingerprint_batch_chunked<S, F>(
    &mut self,
    images: &[(S, Vec<u8>)],
    chunk_size: usize,
    callback: F
)
where
    S: Send + Sync + Clone + 'static,
    F: FnMut(S, Result<MultiHashFingerprint, ImgFprintError>)
```

Processes images in chunks with buffer reuse. Combines memory efficiency of streaming with better cache utilization.

**Example:**
```rust
use imgfprint::FingerprinterContext;

let mut ctx = FingerprinterContext::new();

let images = vec![
    ("img1.jpg".to_string(), std::fs::read("img1.jpg")?),
    ("img2.jpg".to_string(), std::fs::read("img2.jpg")?),
    // ... thousands of images
];

ctx.fingerprint_batch_chunked(&images, 50, |id, result| {
    if let Ok(fp) = result {
        // Store or process fingerprint
        db.store(&id, fp)?;
    }
});
```

---

### ImageFingerprint

A perceptual fingerprint containing multiple hash layers for robust comparison.

#### Accessors

##### `exact_hash()`

```rust
pub fn exact_hash(&self) -> &[u8; 32]
```

Returns the BLAKE3 hash of original image bytes.

**Use case:** Exact deduplication before perceptual comparison (much faster).

**Example:**
```rust
let exact: &[u8; 32] = fp.exact_hash();

// Convert to hex string
let hex = hex::encode(exact);
println!("Exact hash: {}", hex);
```

##### `global_hash()`

```rust
pub fn global_hash(&self) -> u64
```

Returns the global perceptual hash from the center 32x32 region. The algorithm used (PHash or DHash) depends on which was specified when creating the fingerprint.

**Example:**
```rust
let hash: u64 = fp.global_hash();
println!("Perceptual hash: {:016x}", hash);
```

##### `block_hashes()`

```rust
pub fn block_hashes(&self) -> &[u64; 16]
```

Returns 16 block-level hashes from a 4x4 grid.

**Use case:** Crop-resistant comparison by matching partial regions.

**Example:**
```rust
let blocks: &[u64; 16] = fp.block_hashes();

// Access individual blocks
for (i, hash) in blocks.iter().enumerate() {
    println!("Block {}: {:016x}", i, hash);
}
```

#### Comparison Methods

##### `distance()`

```rust
pub fn distance(&self, other: &ImageFingerprint) -> u32
```

Computes Hamming distance between global hashes.

**Returns:** 0 (identical) to 64 (completely different)

**Example:**
```rust
let dist = fp1.distance(&fp2);
println!("Hamming distance: {} (lower is more similar)", dist);

// Convert to similarity percentage
let similarity = 1.0 - (dist as f32 / 64.0);
```

##### `is_similar()`

```rust
pub fn is_similar(&self, other: &ImageFingerprint, threshold: f32) -> bool
```

Checks if this fingerprint is similar to another within a threshold.

**Parameters:**
- `other`: Fingerprint to compare against
- `threshold`: Similarity threshold 0.0 to 1.0 (default: 0.8)

**Example:**
```rust
// Check if images are 80% similar
if fp1.is_similar(&fp2, 0.8) {
    println!("Images are similar!");
}

// Stricter matching (90%)
if fp1.is_similar(&fp2, 0.9) {
    println!("Images are nearly identical!");
}

// Loose matching (50%)
if fp1.is_similar(&fp2, 0.5) {
    println!("Images share some visual elements");
}
```

**Note:** A threshold of 1.0 requires exact match (both BLAKE3 and perceptual hash identical).

---

### MultiHashFingerprint

A multi-algorithm fingerprint containing AHash, PHash, and DHash results for enhanced accuracy.

#### Accessors

##### `exact_hash()`

```rust
pub fn exact_hash(&self) -> &[u8; 32]
```

Returns the BLAKE3 hash of original image bytes.

##### `ahash()`

```rust
pub fn ahash(&self) -> &ImageFingerprint
```

Returns the AHash-based fingerprint (global + block hashes).

##### `phash()`

```rust
pub fn phash(&self) -> &ImageFingerprint
```

Returns the PHash-based fingerprint (global + block hashes).

##### `dhash()`

```rust
pub fn dhash(&self) -> &ImageFingerprint
```

Returns the DHash-based fingerprint (global + block hashes).

**Example:**
```rust
let multi = ImageFingerprinter::fingerprint(&image_bytes)?;

// Access individual algorithms
let ahash = multi.ahash();
let phash = multi.phash();
let dhash = multi.dhash();

println!("AHash: {:016x}", ahash.global_hash());
println!("PHash: {:016x}", phash.global_hash());
println!("DHash: {:016x}", dhash.global_hash());
```

##### `get()`

```rust
pub fn get(&self, algorithm: HashAlgorithm) -> &ImageFingerprint
```

Returns the fingerprint for a specific algorithm.

**Example:**
```rust
use imgfprint::HashAlgorithm;

let fp = multi.get(HashAlgorithm::PHash);
```

#### Comparison Methods

##### `compare()`

```rust
pub fn compare(&self, other: &MultiHashFingerprint) -> Similarity
```

Compares two multi-hash fingerprints using weighted combination.

**Weights:**
- **10%** AHash similarity (average hash, fastest)
- **60%** PHash similarity (DCT-based, robust to compression)
- **30%** DHash similarity (gradient-based, good for structural changes)

**Example:**
```rust
let multi1 = ImageFingerprinter::fingerprint(&img1)?;
let multi2 = ImageFingerprinter::fingerprint(&img2)?;

// Weighted comparison
let sim = multi1.compare(&multi2);
println!("Combined similarity: {:.2}", sim.score);
```

##### `is_similar()`

```rust
pub fn is_similar(&self, other: &MultiHashFingerprint, threshold: f32) -> bool
```

Checks if this fingerprint is similar to another within a threshold.

**Example:**
```rust
if multi1.is_similar(&multi2, 0.8) {
    println!("Images are similar!");
}
```

---

### Similarity

Similarity score between two image fingerprints.

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `f32` | Similarity 0.0 (different) to 1.0 (identical) |
| `exact_match` | `bool` | True if BLAKE3 hashes match |
| `perceptual_distance` | `u32` | Hamming distance 0-64 |

#### Methods

##### `perfect()`

```rust
pub fn perfect() -> Self
```

Returns a perfect similarity score (1.0, exact match).

**Example:**
```rust
let perfect = imgfprint::Similarity::perfect();
assert_eq!(perfect.score, 1.0);
assert!(perfect.exact_match);
```

---

## Advanced Usage

### Batch Processing

Process thousands of images efficiently using two strategies:

#### Strategy 1: Parallel Batch (Best for Many Images)

Uses all CPU cores for maximum throughput:

```rust
use imgfprint::ImageFingerprinter;
use std::path::Path;

fn process_parallel(image_paths: &[&Path]) -> Result<(), Box<dyn std::error::Error>> {
    let images: Vec<_> = image_paths
        .iter()
        .map(|p| (p.to_string_lossy().to_string(), std::fs::read(p).unwrap()))
        .collect();
    
    let results = ImageFingerprinter::fingerprint_batch(&images);
    
    for (path, result) in results {
        match result {
            Ok(fp) => {
                // Store in database, index, etc.
                db.store(&path, fp)?;
            }
            Err(e) => eprintln!("Failed to process {}: {}", path, e),
        }
    }
    
    Ok(())
}
```

**Speedup:** 2-3x on multi-core systems.

#### Strategy 2: Streaming with Context (Best for Memory-Constrained)

Lower memory usage, ideal for processing large datasets:

```rust
use imgfprint::FingerprinterContext;
use std::path::Path;

fn process_streaming(image_paths: &[&Path]) -> Result<(), Box<dyn std::error::Error>> {
    let mut ctx = FingerprinterContext::new();
    
    for path in image_paths {
        let bytes = std::fs::read(path)?;
        let fp = ctx.fingerprint(&bytes)?;
        db.store(&path.to_string_lossy(), fp)?;
    }
    
    Ok(())
}
```

**Benefit:** Minimal memory footprint, processes images one at a time.

#### Strategy 3: Chunked Batch (Best for Large Batches)

Combines memory efficiency with callback-based processing for very large datasets:

```rust
use imgfprint::ImageFingerprinter;
use std::path::Path;

fn process_chunked(image_paths: &[&Path]) -> Result<usize, Box<dyn std::error::Error>> {
    let images: Vec<_> = image_paths
        .iter()
        .map(|p| (p.to_string_lossy().to_string(), std::fs::read(p).unwrap()))
        .collect();
    
    let mut count = 0;
    
    // Process in chunks of 100 to limit memory
    ImageFingerprinter::fingerprint_batch_chunked(&images, 100, |id, result| {
        match result {
            Ok(fp) => {
                db.store(&id, fp).unwrap();
                count += 1;
            }
            Err(e) => eprintln!("Failed to process {}: {}", id, e),
        }
    });
    
    Ok(count)
}
```

**Benefit:** Bounded memory usage (100 images max), callback enables immediate processing/streaming results.

### Semantic Embeddings

Work with CLIP-style semantic embeddings (requires custom provider implementation).

#### Custom Provider Example

```rust
use imgfprint::{Embedding, EmbeddingProvider, ImgFprintError};

struct MyClipProvider {
    client: reqwest::Client,
    api_key: String,
}

impl EmbeddingProvider for MyClipProvider {
    fn embed(&self, image: &[u8]) -> Result<Embedding, ImgFprintError> {
        let response = self.client
            .post("https://api.example.com/embed")
            .header("Authorization", &self.api_key)
            .body(image.to_vec())
            .send()
            .map_err(|e| ImgFprintError::ProviderError(e.to_string()))?;
        
        let vector: Vec<f32> = response.json()
            .map_err(|e| ImgFprintError::ProviderError(e.to_string()))?;
        
        Embedding::new(vector)
    }
}

// Usage
let provider = MyClipProvider { 
    client: reqwest::Client::new(),
    api_key: "your-api-key".to_string(),
};

let img1 = std::fs::read("cat.jpg")?;
let img2 = std::fs::read("dog.jpg")?;

let emb1 = provider.embed(&img1)?;
let emb2 = provider.embed(&img2)?;

let similarity = imgfprint::semantic_similarity(&emb1, &emb2)?;
println!("Semantic similarity: {:.4}", similarity);
```

**Note:** The library does not include built-in embedding providers. You must implement `EmbeddingProvider` for your use case (OpenAI CLIP, HuggingFace, local models, etc.).

#### Local ONNX Models

With the `local-embedding` feature:

```toml
[dependencies]
imgfprint = { version = "0.3.1", features = ["local-embedding"] }
```

```rust
use imgfprint::{LocalProvider, LocalProviderConfig};

// Load CLIP model
let provider = LocalProvider::from_file("clip-vit-base-patch32.onnx")?;

// Or with custom configuration
let config = LocalProviderConfig::clip_vit_large_patch14();
let provider = LocalProvider::from_file_with_config("clip.onnx", config)?;

// Generate embedding
let image = std::fs::read("photo.jpg")?;
let embedding = provider.embed(&image)?;

println!("Generated {}-dimensional embedding", embedding.len());
```

---

## Error Handling

All operations return `Result<T, ImgFprintError>`. The library never panics on malformed input.

### Creating Errors (For SDK Users)

When implementing custom providers or extending the library, use the optimized error constructors:

```rust
use imgfprint::ImgFprintError;

let err = ImgFprintError::decode_error("invalid format");
let err = ImgFprintError::invalid_image("dimensions too large");
let err = ImgFprintError::processing_error("resize failed");

let size = 1024;
let err = ImgFprintError::invalid_image(format!("image too large: {} bytes", size));
```

### Complete Error Handling Example

```rust
use imgfprint::{ImageFingerprinter, ImgFprintError};

match ImageFingerprinter::fingerprint(&bytes) {
    Ok(fp) => {
        // Use fingerprint
    }
    
    // Image-related errors
    Err(ImgFprintError::InvalidImage(msg)) => {
        eprintln!("Invalid image: {}", msg);
    }
    Err(ImgFprintError::UnsupportedFormat(msg)) => {
        eprintln!("Unsupported format: {}", msg);
    }
    Err(ImgFprintError::DecodeError(msg)) => {
        eprintln!("Decode failed: {}", msg);
    }
    Err(ImgFprintError::ImageTooSmall(msg)) => {
        eprintln!("Image too small: {} (minimum 32x32)", msg);
    }
    
    // Processing errors
    Err(ImgFprintError::ProcessingError(msg)) => {
        eprintln!("Processing error: {}", msg);
    }
    
    // Embedding errors
    Err(ImgFprintError::EmbeddingDimensionMismatch { expected, actual }) => {
        eprintln!("Dimension mismatch: expected {}, got {}", expected, actual);
    }
    Err(ImgFprintError::ProviderError(msg)) => {
        eprintln!("Provider error: {}", msg);
    }
    Err(ImgFprintError::InvalidEmbedding(msg)) => {
        eprintln!("Invalid embedding: {}", msg);
    }
}
```

---

## Performance Tips

### 1. Use Context API for High Throughput

**Avoid:** Creating new allocations for each image
```rust
// Slow: Allocates buffers for each image
for path in paths {
    let fp = ImageFingerprinter::fingerprint(&std::fs::read(path)?)?;
}
```

**Prefer:** Buffer reuse with context
```rust
// Fast: Reuses buffers
let mut ctx = FingerprinterContext::new();
for path in paths {
    let fp = ctx.fingerprint(&std::fs::read(path)?)?;
}
```

**Savings:** ~260KB allocations per image eliminated.

### 2. Use Batch Processing for Parallel Workloads

```rust
// Process 100 images in parallel using all CPU cores
let results = ImageFingerprinter::fingerprint_batch(&images);
```

**Speedup:** 2-3x on multi-core systems.

### 3. Check Exact Hash First

```rust
// Fast path: exact byte match (BLAKE3 comparison)
if fp1.exact_hash() == fp2.exact_hash() {
    return Similarity::perfect();
}

// Slow path: perceptual comparison (DCT-based)
let sim = ImageFingerprinter::compare(&fp1, &fp2);
```

### 4. Handle Common Formats Efficiently

Supported formats: PNG, JPEG, GIF, WebP, BMP

| Format | Decode Speed | Best For |
|--------|--------------|----------|
| PNG | Fast | Synthetic images, screenshots |
| JPEG | Medium | Photos (use `zune-jpeg` for 2-3x speedup) |
| WebP | Slow | Web-optimized images |
| GIF | Fast | Simple graphics |

### 5. Image Size Guidelines

| Metric | Value | Notes |
|--------|-------|-------|
| Minimum | 32x32 pixels | Enforced |
| Maximum | 8192x8192 pixels | OOM protection |
| Optimal | 256x512 to 1024x1024 | Best performance |

**Note:** Larger images are downscaled to 256x256 internally, so feeding very large images wastes decode time.

---

## Feature Flags

Configure the library for your needs:

```toml
[dependencies]
# Minimal build (no parallel processing)
imgfprint = { version = "0.3.1", default-features = false }

# Default (serialization + parallel processing)
imgfprint = "0.3.1"

# With local ONNX inference
imgfprint = { version = "0.3.1", features = ["local-embedding"] }
```

### Available Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | Yes | Serialization support (JSON, binary) via serde |
| `parallel` | Yes | Parallel batch processing using rayon |
| `local-embedding` | No | Local ONNX model inference for embeddings |
| `tracing` | No | Observability hooks for production debugging |

### Tracing Feature

Enable the `tracing` feature to add performance instrumentation:

```toml
[dependencies]
imgfprint = { version = "0.3.1", features = ["tracing"] }
```

```rust
use imgfprint::ImageFingerprinter;

// Initialize a tracing subscriber with env filter
// RUST_LOG env var controls the log level: info, debug, trace
std::env::set_var("RUST_LOG", "info");
tracing_subscriber::fmt::init();

// Now all fingerprint operations emit trace events
let fp = ImageFingerprinter::fingerprint(&data).unwrap();
// Outputs: TRACE fingerprinter: fingerprint completed duration_ms=1.23
```

**Environment Variables:**
- `RUST_LOG=info` - Show info level and above
- `RUST_LOG=debug` - Show debug level and above  
- `RUST_LOG=trace` - Show all trace events (most verbose)
- `RUST_LOG=imgfprint=debug` - Show only imgfprint debug logs

**Traced Operations:**
- `fingerprint()` - tracks image size, duration
- `fingerprint_with()` - tracks image size, algorithm, duration
- `fingerprint_batch()` - tracks image count, parallel/sequential, duration
- `fingerprint_batch_chunked()` - tracks chunk_size, processed count, failed count, duration

**Note:** If you don't see output, ensure you've set `RUST_LOG` before calling `tracing_subscriber::fmt::init()`. The subscriber uses an env filter by default which requires this environment variable.

---

## License

MIT License - See LICENSE file for details.

## Links

- [Crates.io](https://crates.io/crates/imgfprint)
- [Documentation](https://docs.rs/imgfprint)
- [Repository](https://github.com/themankindproject/imgfprint-rs)
