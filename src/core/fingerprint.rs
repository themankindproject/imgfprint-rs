use crate::core::similarity::Similarity;
use crate::hash::algorithms::HashAlgorithm;

/// Weights for weighted combination of algorithm similarities.
///
/// Default: AHash 10%, PHash 60%, DHash 30%
const AHASH_WEIGHT: f32 = 0.10;
const PHASH_WEIGHT: f32 = 0.60;
const DHASH_WEIGHT: f32 = 0.30;

/// A perceptual fingerprint containing multiple hash layers for robust comparison.
///
/// Fingerprints are deterministic and comparable across platforms. The structure
/// includes exact hashing for identical detection and perceptual hashing for
/// similarity detection with resistance to resizing, compression, and cropping.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, PartialEq)]
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
    pub fn exact_hash(&self) -> &[u8; 32] {
        &self.exact
    }

    /// Returns the global perceptual hash from the center 32x32 region.
    ///
    /// This hash captures the overall structure of the image and is robust
    /// to minor changes in compression and color adjustments. The algorithm
    /// used (PHash or DHash) depends on which was specified when creating the fingerprint.
    #[inline]
    pub fn global_hash(&self) -> u64 {
        self.global_hash
    }

    /// Returns the 16 block-level perceptual hashes from a 4x4 grid.
    ///
    /// Block hashes enable crop-resistant comparison by matching partial
    /// regions between images. Each hash covers a 64x64 pixel region.
    #[inline]
    pub fn block_hashes(&self) -> &[u64; 16] {
        &self.block_hashes
    }

    /// Computes the Hamming distance between this and another fingerprint's global hash.
    ///
    /// Returns a value from 0 (identical) to 64 (completely different).
    #[inline(always)]
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
    pub fn is_similar(&self, other: &ImageFingerprint, threshold: f32) -> bool {
        debug_assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );
        if self.exact == other.exact {
            return true;
        }
        let dist = self.distance(other);
        let similarity = 1.0 - (dist as f32 / 64.0);
        similarity >= threshold
    }
}

/// A multi-algorithm fingerprint containing hashes from multiple perceptual algorithms.
///
/// Provides enhanced similarity detection by combining results from multiple
/// hash algorithms with weighted combination for improved accuracy.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, PartialEq)]
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
    pub fn exact_hash(&self) -> &[u8; 32] {
        &self.exact
    }

    /// Returns the AHash-based fingerprint.
    #[inline]
    pub fn ahash(&self) -> &ImageFingerprint {
        &self.ahash
    }

    /// Returns the PHash-based fingerprint.
    #[inline]
    pub fn phash(&self) -> &ImageFingerprint {
        &self.phash
    }

    /// Returns the DHash-based fingerprint.
    #[inline]
    pub fn dhash(&self) -> &ImageFingerprint {
        &self.dhash
    }

    /// Returns the fingerprint for a specific algorithm.
    pub fn get(&self, algorithm: HashAlgorithm) -> &ImageFingerprint {
        match algorithm {
            HashAlgorithm::AHash => &self.ahash,
            HashAlgorithm::PHash => &self.phash,
            HashAlgorithm::DHash => &self.dhash,
        }
    }

    /// Compares two multi-hash fingerprints using weighted combination.
    ///
    /// Uses weighted combination:
    /// - 10% AHash similarity (average hash, fastest)
    /// - 60% PHash similarity (DCT-based, robust to compression)
    /// - 30% DHash similarity (gradient-based, good for structural changes)
    ///
    /// Returns a Similarity struct with combined score and component distances.
    ///
    /// # Arguments
    /// * `other` - The fingerprint to compare against
    pub fn compare(&self, other: &MultiHashFingerprint) -> Similarity {
        if self.exact == other.exact {
            return Similarity {
                score: 1.0,
                exact_match: true,
                perceptual_distance: 0,
            };
        }

        let ahash_dist = self.ahash.distance(&other.ahash);
        let phash_dist = self.phash.distance(&other.phash);
        let dhash_dist = self.dhash.distance(&other.dhash);

        let ahash_sim = 1.0 - (ahash_dist as f32 / 64.0);
        let phash_sim = 1.0 - (phash_dist as f32 / 64.0);
        let dhash_sim = 1.0 - (dhash_dist as f32 / 64.0);

        let weighted_score =
            ahash_sim * AHASH_WEIGHT + phash_sim * PHASH_WEIGHT + dhash_sim * DHASH_WEIGHT;

        let avg_distance = ((ahash_dist as f32 * AHASH_WEIGHT)
            + (phash_dist as f32 * PHASH_WEIGHT)
            + (dhash_dist as f32 * DHASH_WEIGHT)) as u32;

        Similarity {
            score: weighted_score.clamp(0.0, 1.0),
            exact_match: false,
            perceptual_distance: avg_distance,
        }
    }

    /// Checks if this fingerprint is similar to another within a threshold.
    ///
    /// Uses the weighted combination score from compare().
    pub fn is_similar(&self, other: &MultiHashFingerprint, threshold: f32) -> bool {
        debug_assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );
        self.compare(other).score >= threshold
    }
}
