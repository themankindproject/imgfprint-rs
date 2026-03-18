use crate::core::similarity::{hash_similarity, Similarity};
use crate::hash::algorithms::HashAlgorithm;

/// Weights for weighted combination of algorithm similarities.
///
/// Default: `AHash` 10%, `PHash` 60%, `DHash` 30%
///
/// These weights were determined empirically to maximize accuracy across
/// diverse image transformations:
/// - `PHash` receives highest weight (60%) due to superior robustness to compression
/// - `DHash` receives moderate weight (30%) for structural change detection
/// - `AHash` receives lowest weight (10%) as a fast baseline
const AHASH_WEIGHT: f32 = 0.10;
const PHASH_WEIGHT: f32 = 0.60;
const DHASH_WEIGHT: f32 = 0.30;

/// A perceptual fingerprint containing multiple hash layers for robust comparison.
///
/// Fingerprints are deterministic and comparable across platforms. The structure
/// includes exact hashing for identical detection and perceptual hashing for
/// similarity detection with resistance to resizing, compression, and cropping.
///
/// # Cache Alignment
/// This struct is cache-line aligned (64 bytes) for optimal performance on
/// modern CPUs. The total size is 192 bytes (3 × 64-byte cache lines).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, PartialEq)]
#[repr(align(64))]
pub struct ImageFingerprint {
    pub(crate) exact: [u8; 32],
    pub(crate) global_hash: u64,
    pub(crate) block_hashes: [u64; 16],
}

impl ImageFingerprint {
    #[inline]
    pub(crate) fn new(exact: [u8; 32], global_hash: u64, block_hashes: [u64; 16]) -> Self {
        Self {
            exact,
            global_hash,
            block_hashes,
        }
    }

    /// Returns the BLAKE3 hash of the original image bytes.
    ///
    /// Two images with identical byte content will have matching exact hashes.
    /// Use this for exact deduplication before perceptual comparison.
    #[inline]
    #[must_use]
    pub fn exact_hash(&self) -> &[u8; 32] {
        &self.exact
    }

    /// Returns the global perceptual hash from the center 32x32 region.
    ///
    /// This hash captures the overall structure of the image and is robust
    /// to minor changes in compression and color adjustments. The algorithm
    /// used (`PHash` or `DHash`) depends on which was specified when creating the fingerprint.
    #[inline]
    #[must_use]
    pub fn global_hash(&self) -> u64 {
        self.global_hash
    }

    /// Returns the 16 block-level perceptual hashes from a 4x4 grid.
    ///
    /// Block hashes enable crop-resistant comparison by matching partial
    /// regions between images. Each hash covers a 64x64 pixel region.
    #[inline]
    #[must_use]
    pub fn block_hashes(&self) -> &[u64; 16] {
        &self.block_hashes
    }

    /// Computes the Hamming distance between this and another fingerprint's global hash.
    ///
    /// Returns a value from 0 (identical) to 64 (completely different).
    #[inline]
    #[must_use]
    pub fn distance(&self, other: &ImageFingerprint) -> u32 {
        (self.global_hash ^ other.global_hash).count_ones()
    }

    /// Checks if this fingerprint is similar to another within a threshold.
    ///
    /// # Arguments
    /// * `other` - The fingerprint to compare against
    /// * `threshold` - Similarity threshold from 0.0 to 1.0 (default: 0.8)
    ///
    /// # Example
    /// ```ignore
    /// // Use ImageFingerprinter::fingerprint() to create fingerprints first
    /// use imgfprint::ImageFingerprinter;
    ///
    /// let fp1 = ImageFingerprinter::fingerprint(&std::fs::read("image1.jpg")?).unwrap();
    /// let fp2 = ImageFingerprinter::fingerprint(&std::fs::read("image2.jpg")?).unwrap();
    ///
    /// if fp1.is_similar(&fp2, 0.8) {
    ///     println!("Images are similar!");
    /// }
    /// ```
    #[doc(alias = "compare")]
    #[doc(alias = "match")]
    #[must_use]
    pub fn is_similar(&self, other: &ImageFingerprint, threshold: f32) -> bool {
        debug_assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in range [0.0, 1.0], got {threshold}"
        );
        if self.exact == other.exact {
            return true;
        }
        let dist = self.distance(other);
        let similarity = hash_similarity(dist);
        similarity >= threshold
    }
}

/// A multi-algorithm fingerprint containing hashes from multiple perceptual algorithms.
///
/// Provides enhanced similarity detection by combining results from multiple
/// hash algorithms with weighted combination for improved accuracy.
///
/// # Cache Alignment
/// This struct is cache-line aligned (64 bytes) for optimal performance.
/// Total size is 640 bytes (10 × 64-byte cache lines).
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, PartialEq)]
#[repr(align(64))]
pub struct MultiHashFingerprint {
    pub(crate) exact: [u8; 32],
    pub(crate) ahash: ImageFingerprint,
    pub(crate) phash: ImageFingerprint,
    pub(crate) dhash: ImageFingerprint,
}

impl MultiHashFingerprint {
    pub(crate) fn new(
        exact: [u8; 32],
        ahash: ImageFingerprint,
        phash: ImageFingerprint,
        dhash: ImageFingerprint,
    ) -> Self {
        Self {
            exact,
            ahash,
            phash,
            dhash,
        }
    }

    /// Returns the BLAKE3 hash of the original image bytes.
    #[inline]
    #[must_use]
    pub fn exact_hash(&self) -> &[u8; 32] {
        &self.exact
    }

    /// Returns the AHash-based fingerprint.
    #[inline]
    #[must_use]
    pub fn ahash(&self) -> &ImageFingerprint {
        &self.ahash
    }

    /// Returns the PHash-based fingerprint.
    #[inline]
    #[must_use]
    pub fn phash(&self) -> &ImageFingerprint {
        &self.phash
    }

    /// Returns the DHash-based fingerprint.
    #[inline]
    #[must_use]
    pub fn dhash(&self) -> &ImageFingerprint {
        &self.dhash
    }

    /// Returns the fingerprint for a specific algorithm.
    #[must_use]
    pub fn get(&self, algorithm: HashAlgorithm) -> &ImageFingerprint {
        match algorithm {
            HashAlgorithm::AHash => &self.ahash,
            HashAlgorithm::PHash => &self.phash,
            HashAlgorithm::DHash => &self.dhash,
        }
    }

    /// Compares two multi-hash fingerprints using weighted combination.
    ///
    /// Uses weighted combination of algorithm similarities:
    /// - 10% `AHash` similarity (average hash, fastest)
    /// - 60% `PHash` similarity (DCT-based, robust to compression)
    /// - 30% `DHash` similarity (gradient-based, good for structural changes)
    ///
    /// Each algorithm's similarity includes both global and block-level hashing
    /// for improved crop resistance. Block hashes are weighted at 60% and global
    /// hashes at 40% within each algorithm.
    ///
    /// Returns a Similarity struct with combined score and component distances.
    ///
    /// Uses constant-time comparison to prevent timing attacks on exact match.
    ///
    /// # Arguments
    /// * `other` - The fingerprint to compare against
    #[must_use]
    pub fn compare(&self, other: &MultiHashFingerprint) -> Similarity {
        use crate::core::similarity::{compute_similarity, hamming_distance};
        use subtle::ConstantTimeEq;

        let exact_match = self.exact.ct_eq(&other.exact).into();

        // Fast path: exact match returns immediately
        if exact_match {
            return Similarity {
                score: 1.0,
                exact_match: true,
                perceptual_distance: 0,
            };
        }

        // Compute per-algorithm similarities including block-level comparison
        let ahash_sim = compute_similarity(&self.ahash, &other.ahash).score;
        let phash_sim = compute_similarity(&self.phash, &other.phash).score;
        let dhash_sim = compute_similarity(&self.dhash, &other.dhash).score;

        let weighted_score =
            ahash_sim * AHASH_WEIGHT + phash_sim * PHASH_WEIGHT + dhash_sim * DHASH_WEIGHT;

        // Use global hash distances for perceptual_distance (backward compatibility)
        let ahash_dist = hamming_distance(self.ahash.global_hash, other.ahash.global_hash);
        let phash_dist = hamming_distance(self.phash.global_hash, other.phash.global_hash);
        let dhash_dist = hamming_distance(self.dhash.global_hash, other.dhash.global_hash);

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let avg_distance = ((ahash_dist as f32 * AHASH_WEIGHT)
            + (phash_dist as f32 * PHASH_WEIGHT)
            + (dhash_dist as f32 * DHASH_WEIGHT)) as u32;

        // Score is already in valid range due to hash_similarity returning [0, 1]
        // and weights summing to 1.0, so clamping is only needed for floating-point errors
        Similarity {
            score: weighted_score.clamp(0.0, 1.0),
            exact_match: false,
            perceptual_distance: avg_distance,
        }
    }

    /// Checks if this fingerprint is similar to another within a threshold.
    ///
    /// Uses the weighted combination score from compare().
    #[must_use]
    pub fn is_similar(&self, other: &MultiHashFingerprint, threshold: f32) -> bool {
        debug_assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );
        self.compare(other).score >= threshold
    }
}
