//! Perceptual hashing algorithms.
//!
//! This module contains implementations of three hash algorithms:
//! - [`ahash`](ahash): Average-based hashing (fastest)
//! - [`phash`](phash): DCT-based hashing (most robust)
//! - [`dhash`](dhash): Gradient-based hashing (good for crops)
//!
//! Each algorithm produces a 64-bit hash that can be compared using
//! Hamming distance. See [`HashAlgorithm`](algorithms::HashAlgorithm) for details.

pub mod ahash;
pub mod algorithms;
pub mod dhash;
pub mod phash;
