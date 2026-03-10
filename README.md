# imgfprint-rs

A production-grade Rust library for high-performance perceptual image fingerprinting.

![Rust](https://img.shields.io/badge/rust-1.70%2B-blue)
[![Crates.io](https://img.shields.io/crates/v/imgfprint-rs)](https://crates.io/crates/imgfprint-rs)
[![License](https://img.shields.io/crates/l/imgfprint-rs)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/themankindproject/imgfprint-rs/main.yml)](https://github.com/themankindproject/imgfprint-rs/actions)

Fast, deterministic image fingerprinting for deduplication, similarity detection, and content identification in production systems.

## Features

- **SHA256 Exact Hash** - Byte-identical detection
- **pHash Perceptual Hash** - DCT-based similarity detection
- **Block-level Hashing** - 4x4 grid for crop resistance
- **Parallel Processing** - Multi-core optimized
- **Serde Support** - JSON/binary serialization
- **Dimension Protection** - OOM attack prevention (8192px max)
- **Multiple Formats** - PNG, JPEG, GIF, WebP, BMP

## Quick Start

```rust
use imgfprint_rs::ImageFingerprinter;

let fp1 = ImageFingerprinter::fingerprint(&img1_bytes)?;
let fp2 = ImageFingerprinter::fingerprint(&img2_bytes)?;

let sim = ImageFingerprinter::compare(&fp1, &fp2);

if sim.score > 0.8 {
    println!("Images are similar!");
}
```

## Installation

```toml
# Cargo.toml
[dependencies]
imgfprint-rs = "0.1"
```

### Features

| Feature | Default | Description |
|---------|---------|-------------|
| `serde` | ✅ | JSON/binary serialization |
| `parallel` | ✅ | Multi-core block processing |

Disable features for minimal builds:
```toml
[dependencies]
imgfprint-rs = { version = "0.1", default-features = false }
```

## API

### Fingerprint

```rust
// Create fingerprint from image bytes
let fp = ImageFingerprinter::fingerprint(&image_bytes)?;

// Access components
fp.exact_hash();      // &[u8; 32] - SHA256
fp.global_phash();   // u64 - perceptual hash
fp.block_hashes();   // &[u64; 16] - block hashes
```

### Compare

```rust
use imgfprint_rs::ImageFingerprinter;

let sim = ImageFingerprinter::compare(&fp1, &fp2);

// sim.score:           0.0 - 1.0 similarity
// sim.exact_match:     true if SHA256 matches
// sim.perceptual_distance: 0-64 Hamming distance
```

### Distance Methods

```rust
// Direct distance calculation
let dist = fp1.distance(&fp2);  // 0-64 Hamming distance

// Threshold-based check
if fp1.is_similar(&fp2, 0.8) {  // threshold 0.0-1.0
    println!("similar!");
}
```

### Batch Processing

```rust
let images = vec![
    ("img1.jpg".to_string(), bytes1),
    ("img2.jpg".to_string(), bytes2),
    // ... millions of images
];

let results = ImageFingerprinter::fingerprint_batch(&images);

for (id, result) in results {
    match result {
        Ok(fp) => println!("{}: OK", id),
        Err(e) => println!("{}: Error - {}", id, e),
    }
}
```

## Fingerprint Structure

```
ImageFingerprint (168 bytes)
├── exact:      [u8; 32]  SHA256 of original bytes
├── global_phash: u64      Perceptual hash (center 32x32)
└── block_hashes: [u64; 16] Perceptual hashes (4x4 grid)
```

### Algorithm

1. Decode image (any format → RGB)
2. Normalize to 256×256 grayscale
3. Extract center 32×32 → global pHash (DCT)
4. Split into 4×4 blocks (64×64 each) → block pHashes
5. SHA256 original bytes → exact hash

### Comparison

- **40%** global perceptual hash similarity
- **60%** block-level hash similarity

This provides robustness to:
- Resizing
- Compression
- Minor color adjustments
- Cropping (via block hashes)

## Performance

```
fingerprint():     ~1-5ms (300x300 image, single core)
compare():         ~100ns
batch (10 images): ~8ms (parallel)
```

Run benchmarks:
```bash
cargo bench
```

## Error Handling

```rust
use imgfprint_rs::{ImageFingerprinter, ImgFprintError};

match ImageFingerprinter::fingerprint(&bytes) {
    Ok(fp) => { /* use fingerprint */ }
    Err(ImgFprintError::InvalidImage(msg)) => { /* empty or too large */ }
    Err(ImgFprintError::UnsupportedFormat(msg)) => { /* bad format */ }
    Err(ImgFprintError::DecodeError(msg)) => { /* corrupted */ }
    Err(ImgFprintError::ProcessingError(msg)) => { /* internal */ }
}
```

## Serialization

```rust
use imgfprint_rs::{ImageFingerprinter, Similarity};

// JSON
let json = serde_json::to_string(&fp)?;

// Binary
let bytes = bincode::serialize(&fp)?;
```

## Security

- Maximum image dimension: 8192×8192 pixels
- Dimension check before full decode (OOM protection)
- No panics on malformed input
- Deterministic output across platforms

## Production Checklist

- [x] SHA256 exact matching
- [x] Perceptual hashing (pHash/DCT)
- [x] Crop-resistant block hashing
- [x] Parallel processing
- [x] Batch API
- [x] Serde serialization
- [x] Error handling
- [x] Dimension limits
- [x] Benchmarks
- [x] Deterministic output

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

## Contributing

Contributions welcome! Please ensure:
- All tests pass: `cargo test`
- No warnings: `cargo clippy`
- Benchmarks run: `cargo bench`
