//! Core fingerprinting functionality.
//!
//! This module provides the main entry points for image fingerprinting:
//! - [`ImageFingerprinter`](super::core::fingerprinter::ImageFingerprinter) - High-level API
//! - [`FingerprinterContext`](super::core::fingerprinter::FingerprinterContext) - Reusable context for performance
//! - [`ImageFingerprint`](super::core::fingerprint::ImageFingerprint) - Single algorithm fingerprint
//! - [`MultiHashFingerprint`](super::core::fingerprint::MultiHashFingerprint) - Multi-algorithm fingerprint
//! - [`Similarity`](super::core::similarity::Similarity) - Similarity scoring

pub mod fingerprint;
pub mod fingerprinter;
pub mod similarity;
