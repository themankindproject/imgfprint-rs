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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_algorithm_ahash_bits() {
        assert_eq!(HashAlgorithm::AHash.hash_bits(), 64);
    }

    #[test]
    fn test_hash_algorithm_phash_bits() {
        assert_eq!(HashAlgorithm::PHash.hash_bits(), 64);
    }

    #[test]
    fn test_hash_algorithm_dhash_bits() {
        assert_eq!(HashAlgorithm::DHash.hash_bits(), 64);
    }

    #[test]
    fn test_hash_algorithm_ahash_max_distance() {
        assert_eq!(HashAlgorithm::AHash.max_distance(), 64);
    }

    #[test]
    fn test_hash_algorithm_phash_max_distance() {
        assert_eq!(HashAlgorithm::PHash.max_distance(), 64);
    }

    #[test]
    fn test_hash_algorithm_dhash_max_distance() {
        assert_eq!(HashAlgorithm::DHash.max_distance(), 64);
    }

    #[test]
    fn test_hash_algorithm_clone() {
        let ahash = HashAlgorithm::AHash;
        let phash = HashAlgorithm::PHash;
        let dhash = HashAlgorithm::DHash;
        
        assert_eq!(ahash.clone(), HashAlgorithm::AHash);
        assert_eq!(phash.clone(), HashAlgorithm::PHash);
        assert_eq!(dhash.clone(), HashAlgorithm::DHash);
    }

    #[test]
    fn test_hash_algorithm_copy() {
        let ahash = HashAlgorithm::AHash;
        let phash = HashAlgorithm::PHash;
        let dhash = HashAlgorithm::DHash;
        
        let ahash_copy = ahash;
        let phash_copy = phash;
        let dhash_copy = dhash;
        
        assert_eq!(ahash, ahash_copy);
        assert_eq!(phash, phash_copy);
        assert_eq!(dhash, dhash_copy);
    }

    #[test]
    fn test_hash_algorithm_partial_eq() {
        assert_eq!(HashAlgorithm::AHash, HashAlgorithm::AHash);
        assert_eq!(HashAlgorithm::PHash, HashAlgorithm::PHash);
        assert_eq!(HashAlgorithm::DHash, HashAlgorithm::DHash);
        
        assert_ne!(HashAlgorithm::AHash, HashAlgorithm::PHash);
        assert_ne!(HashAlgorithm::AHash, HashAlgorithm::DHash);
        assert_ne!(HashAlgorithm::PHash, HashAlgorithm::DHash);
    }

    #[test]
    fn test_hash_algorithm_debug() {
        assert_eq!(format!("{:?}", HashAlgorithm::AHash), "AHash");
        assert_eq!(format!("{:?}", HashAlgorithm::PHash), "PHash");
        assert_eq!(format!("{:?}", HashAlgorithm::DHash), "DHash");
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
    fn test_hash_algorithm_in_vec() {
        let algorithms = vec![
            HashAlgorithm::AHash,
            HashAlgorithm::PHash,
            HashAlgorithm::DHash,
        ];
        
        assert_eq!(algorithms.len(), 3);
        assert_eq!(algorithms[0], HashAlgorithm::AHash);
        assert_eq!(algorithms[1], HashAlgorithm::PHash);
        assert_eq!(algorithms[2], HashAlgorithm::DHash);
    }

    #[test]
    fn test_hash_algorithm_match_expression() {
        for algo in [HashAlgorithm::AHash, HashAlgorithm::PHash, HashAlgorithm::DHash] {
            match algo {
                HashAlgorithm::AHash => assert_eq!(algo.hash_bits(), 64),
                HashAlgorithm::PHash => assert_eq!(algo.hash_bits(), 64),
                HashAlgorithm::DHash => assert_eq!(algo.hash_bits(), 64),
            }
        }
    }
}
