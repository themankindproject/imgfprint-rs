//! Perceptual hashing algorithms for image fingerprinting.

/// Available perceptual hash algorithms.
///
/// Each algorithm has different characteristics suitable for different use cases:
/// - **AHash**: Average Hash, fastest, compares pixels to mean
/// - **PHash**: DCT-based, robust to minor visual changes, slower
/// - **DHash**: Gradient-based, fast, good for detecting structural changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    PHash,

    /// Difference Hash using horizontal gradients.
    ///
    /// Fast algorithm that compares adjacent pixels horizontally.
    /// Excellent for detecting cropping and structural changes.
    DHash,
}

impl std::fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashAlgorithm::AHash => f.write_str("AHash"),
            HashAlgorithm::PHash => f.write_str("PHash"),
            HashAlgorithm::DHash => f.write_str("DHash"),
        }
    }
}

impl HashAlgorithm {
    /// Returns the bit length of hashes produced by this algorithm.
    #[must_use]
    pub const fn hash_bits(&self) -> u32 {
        match self {
            HashAlgorithm::AHash => 64,
            HashAlgorithm::PHash => 64,
            HashAlgorithm::DHash => 64,
        }
    }

    /// Returns the maximum Hamming distance for this algorithm.
    #[must_use]
    pub const fn max_distance(&self) -> u32 {
        self.hash_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [HashAlgorithm; 3] = [
        HashAlgorithm::AHash,
        HashAlgorithm::PHash,
        HashAlgorithm::DHash,
    ];

    #[test]
    fn test_hash_algorithm_bits_and_distance() {
        // All three algorithms emit 64-bit hashes today; the table keeps the
        // per-variant coverage if one ever diverges.
        for algo in ALL {
            assert_eq!(algo.hash_bits(), 64, "{algo:?}");
            assert_eq!(algo.max_distance(), 64, "{algo:?}");
            assert_eq!(algo.max_distance(), algo.hash_bits(), "{algo:?}");
        }
    }

    #[test]
    fn test_hash_algorithm_copy_clone_eq() {
        for algo in ALL {
            // Copy: still usable after move; Clone: round-trips.
            let moved = algo;
            assert_eq!(algo, moved);
            assert_eq!(algo.clone(), algo);
        }
        assert_eq!(HashAlgorithm::AHash, HashAlgorithm::AHash);
        assert_eq!(HashAlgorithm::PHash, HashAlgorithm::PHash);
        assert_eq!(HashAlgorithm::DHash, HashAlgorithm::DHash);

        assert_ne!(HashAlgorithm::AHash, HashAlgorithm::PHash);
        assert_ne!(HashAlgorithm::AHash, HashAlgorithm::DHash);
        assert_ne!(HashAlgorithm::PHash, HashAlgorithm::DHash);
    }

    #[test]
    fn test_hash_algorithm_debug_display() {
        for algo in ALL {
            let name = format!("{algo:?}");
            assert_eq!(algo.to_string(), name);
        }
        assert_eq!(format!("{:?}", HashAlgorithm::AHash), "AHash");
    }

    #[test]
    fn test_hash_algorithm_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        HashAlgorithm::AHash.hash(&mut hasher1);
        HashAlgorithm::AHash.hash(&mut hasher2);

        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn test_hash_algorithm_match_expression() {
        for algo in ALL {
            match algo {
                HashAlgorithm::AHash => assert_eq!(algo.hash_bits(), 64),
                HashAlgorithm::PHash => assert_eq!(algo.hash_bits(), 64),
                HashAlgorithm::DHash => assert_eq!(algo.hash_bits(), 64),
            }
        }
    }
}
