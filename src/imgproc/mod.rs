//! Image processing utilities.
//!
//! This module provides low-level image processing operations:
//! - [`decode`](decode): Image decoding from various formats
//! - [`preprocess`](preprocess): Image normalization and feature extraction
//!
//! These are used internally by the fingerprinting pipeline but may
//! also be used directly for custom processing workflows.

pub mod decode;
pub mod preprocess;
