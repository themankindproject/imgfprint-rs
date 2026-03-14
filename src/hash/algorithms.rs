//! Perceptual hashing algorithms for image fingerprinting.

/// Available perceptual hash algorithms.
///
/// Each algorithm has different characteristics suitable for different use cases:
/// - **AHash**: Average Hash, fastest, compares pixels to mean
/// - **PHash**: DCT-based, robust to minor visual changes, slower
/// - **DHash**: Gradient-based, fast, good for detecting structural changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HashAlgorithm {
    /// Average Hash using mean pixel value threshold.
    ///
    /// Simplest and fastest algorithm. Resizes to 8x8 and compares
    /// each pixel to the average brightness.
    AHash,

    /// Perceptual Hash using Discrete Cosine Transform.
    ///
    /// Most robust to compression artifacts and minor adjustments.
    /// Computationally intensive due to 2D DCT.
    #[default]
    PHash,

    /// Difference Hash using horizontal gradients.
    ///
    /// Fast algorithm that compares adjacent pixels horizontally.
    /// Excellent for detecting cropping and structural changes.
    DHash,
}

impl HashAlgorithm {
    /// Returns the bit length of hashes produced by this algorithm.
    pub const fn hash_bits(&self) -> u32 {
        match self {
            HashAlgorithm::AHash => 64,
            HashAlgorithm::PHash => 64,
            HashAlgorithm::DHash => 64,
        }
    }

    /// Returns the maximum Hamming distance for this algorithm.
    pub const fn max_distance(&self) -> u32 {
        self.hash_bits()
    }
}
