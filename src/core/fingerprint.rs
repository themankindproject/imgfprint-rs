use crate::core::similarity::{hash_similarity, Similarity};
use crate::hash::algorithms::HashAlgorithm;

/// Default weight for `AHash` in the combined score (10%).
pub const DEFAULT_AHASH_WEIGHT: f32 = 0.10;
/// Default weight for `PHash` in the combined score (60%).
pub const DEFAULT_PHASH_WEIGHT: f32 = 0.60;
/// Default weight for `DHash` in the combined score (30%).
pub const DEFAULT_DHASH_WEIGHT: f32 = 0.30;
/// Default weight for the global hash inside each per-algorithm similarity (40%).
pub const DEFAULT_GLOBAL_WEIGHT: f32 = 0.40;
/// Default weight for the block-level hashes inside each per-algorithm similarity (60%).
pub const DEFAULT_BLOCK_WEIGHT: f32 = 0.60;
/// Default maximum Hamming distance for a block to count as a valid match (32 of 64).
pub const DEFAULT_BLOCK_DISTANCE_THRESHOLD: u32 = 32;

/// Tunable weights and thresholds for [`MultiHashFingerprint::compare_with_config`].
///
/// Lets an integrator (UCFP, downstream pipelines) shift the trade-off without
/// forking the crate. Defaults reproduce the historic 10/60/30 algorithm
/// blend and the 40/60 global/block split that plain
/// [`compare`](MultiHashFingerprint::compare) uses.
///
/// Weights do not need to sum to 1.0 — the final score is clamped to
/// `[0.0, 1.0]`. Setting any algorithm weight to `0.0` removes it from the
/// score (cheaper than introducing skip flags).
///
/// # Example
///
/// ```rust,no_run
/// use imgfprint::{ImageFingerprinter, MultiHashConfig};
///
/// # fn run(a: &[u8], b: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
/// // PHash-only scoring — useful when AHash/DHash aren't trusted on this corpus.
/// let cfg = MultiHashConfig {
///     ahash_weight: 0.0,
///     phash_weight: 1.0,
///     dhash_weight: 0.0,
///     ..MultiHashConfig::default()
/// };
///
/// let fp1 = ImageFingerprinter::fingerprint(a)?;
/// let fp2 = ImageFingerprinter::fingerprint(b)?;
/// let sim = fp1.compare_with_config(&fp2, &cfg);
/// # let _ = sim;
/// # Ok(())
/// # }
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiHashConfig {
    /// Weight applied to the `AHash` per-algorithm similarity.
    pub ahash_weight: f32,
    /// Weight applied to the `PHash` per-algorithm similarity.
    pub phash_weight: f32,
    /// Weight applied to the `DHash` per-algorithm similarity.
    pub dhash_weight: f32,
    /// Weight on the global 32x32 hash inside each per-algorithm similarity.
    pub global_weight: f32,
    /// Weight on the block-level hashes inside each per-algorithm similarity.
    pub block_weight: f32,
    /// Maximum Hamming distance (0–64) for a block to count toward similarity.
    /// Lower = stricter; higher = looser.
    pub block_distance_threshold: u32,
}

impl Default for MultiHashConfig {
    fn default() -> Self {
        Self {
            ahash_weight: DEFAULT_AHASH_WEIGHT,
            phash_weight: DEFAULT_PHASH_WEIGHT,
            dhash_weight: DEFAULT_DHASH_WEIGHT,
            global_weight: DEFAULT_GLOBAL_WEIGHT,
            block_weight: DEFAULT_BLOCK_WEIGHT,
            block_distance_threshold: DEFAULT_BLOCK_DISTANCE_THRESHOLD,
        }
    }
}

/// A perceptual fingerprint containing multiple hash layers for robust comparison.
///
/// Fingerprints are deterministic and comparable across platforms. The structure
/// includes exact hashing for identical detection and perceptual hashing for
/// similarity detection with resistance to resizing, compression, and cropping.
///
/// # Binary layout
///
/// `#[repr(C)]` with no padding bytes (168 bytes total: 32 + 8 + 128). Implements
/// [`bytemuck::Pod`] / [`bytemuck::Zeroable`] so a `&[ImageFingerprint]` can be
/// zero-copy cast to `&[u8]` for mmap-based persistence:
///
/// ```rust
/// use imgfprint::ImageFingerprint;
/// # fn ex(fps: &[ImageFingerprint]) -> &[u8] {
/// bytemuck::cast_slice(fps)
/// # }
/// ```
///
/// `Copy` is derived because `bytemuck::Pod` requires it; the trade-off is
/// that move-by-value silently memcpys 168 bytes. Prefer borrowing
/// (`&ImageFingerprint`) in hot loops where this matters.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
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

    /// Returns the on-disk format version this fingerprint was computed under.
    ///
    /// Equal to [`crate::FORMAT_VERSION`]. Persist alongside fingerprint bytes
    /// (or in a sidecar manifest) and refuse comparison across mismatched
    /// versions to guard against algorithm-version drift.
    #[inline]
    #[must_use]
    pub const fn format_version() -> u32 {
        crate::FORMAT_VERSION
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
    /// # Panics
    /// Panics in debug mode if threshold is not in [0.0, 1.0].
    /// In release mode, out-of-range or NaN thresholds return false (never similar).
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
        // Clamp threshold to valid range for release builds to handle NaN/out-of-range gracefully
        let clamped_threshold = threshold.clamp(0.0, 1.0);
        let dist = self.distance(other);
        let similarity = hash_similarity(dist);
        similarity >= clamped_threshold
    }
}

/// A multi-algorithm fingerprint containing hashes from multiple perceptual algorithms.
///
/// Provides enhanced similarity detection by combining results from multiple
/// hash algorithms with weighted combination for improved accuracy.
///
/// # Binary layout
///
/// `#[repr(C)]` with no padding bytes (536 bytes total: 32 + 3 × 168). Implements
/// [`bytemuck::Pod`] / [`bytemuck::Zeroable`] for zero-copy cast to `&[u8]`.
/// See [`ImageFingerprint`] for an example.
///
/// Stable layout is enforced at compile time via a `const _` size assertion;
/// any accidental layout drift fails the build.
///
/// `Copy` is derived for `bytemuck::Pod` compatibility; move-by-value silently
/// memcpys 536 bytes. Prefer borrowing (`&MultiHashFingerprint`) in hot loops.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MultiHashFingerprint {
    pub(crate) exact: [u8; 32],
    pub(crate) ahash: ImageFingerprint,
    pub(crate) phash: ImageFingerprint,
    pub(crate) dhash: ImageFingerprint,
}

// Layout-stability gate. If anyone accidentally introduces padding or reorders
// fields in a way that changes the binary size, the build fails here. UCFP and
// any other consumer relying on bytemuck::cast_slice would otherwise get
// silently broken artefacts.
const _: () = {
    assert!(
        core::mem::size_of::<ImageFingerprint>() == 168,
        "ImageFingerprint binary layout drifted"
    );
    assert!(
        core::mem::size_of::<MultiHashFingerprint>() == 536,
        "MultiHashFingerprint binary layout drifted"
    );
};

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

    /// Returns the on-disk format version this fingerprint was computed under.
    ///
    /// Equal to [`crate::FORMAT_VERSION`]. Persist alongside fingerprint bytes
    /// (or in a sidecar manifest) and refuse comparison across mismatched
    /// versions to guard against algorithm-version drift.
    #[inline]
    #[must_use]
    pub const fn format_version() -> u32 {
        crate::FORMAT_VERSION
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

    /// Compares two multi-hash fingerprints using the default weighted combination.
    ///
    /// Equivalent to [`compare_with_config`](Self::compare_with_config) called
    /// with [`MultiHashConfig::default()`]:
    /// - 10% `AHash` / 60% `PHash` / 30% `DHash` per-algorithm blend
    /// - Within each algorithm, 40% global hash + 60% block-level hashes
    /// - Block distance threshold of 32 (Hamming, out of 64)
    ///
    /// Use [`compare_with_config`](Self::compare_with_config) to tune any of these.
    #[must_use]
    pub fn compare(&self, other: &MultiHashFingerprint) -> Similarity {
        self.compare_with_threshold(other, 32)
    }

    /// Compares two multi-hash fingerprints with a custom block distance threshold.
    ///
    /// # Arguments
    /// * `other` - The fingerprint to compare against
    /// * `block_threshold` - Maximum Hamming distance for a block to count as a match (0-64).
    ///   Lower = stricter (fewer blocks qualify), higher = looser. Default is 32.
    #[must_use]
    pub fn compare_with_threshold(
        &self,
        other: &MultiHashFingerprint,
        block_threshold: u32,
    ) -> Similarity {
        let cfg = MultiHashConfig {
            block_distance_threshold: block_threshold,
            ..MultiHashConfig::default()
        };
        self.compare_with_config(other, &cfg)
    }

    /// Compares two multi-hash fingerprints using a fully configurable weight
    /// and threshold set.
    ///
    /// All knobs from [`MultiHashConfig`] are honored; defaults reproduce
    /// [`compare`](Self::compare). See [`MultiHashConfig`] for examples.
    #[must_use]
    pub fn compare_with_config(
        &self,
        other: &MultiHashFingerprint,
        config: &MultiHashConfig,
    ) -> Similarity {
        use crate::core::similarity::{compute_similarity_with_weights, hamming_distance};
        use subtle::ConstantTimeEq;

        let exact_match = self.exact.ct_eq(&other.exact).into();

        if exact_match {
            return Similarity {
                score: 1.0,
                exact_match: true,
                perceptual_distance: 0,
            };
        }

        let ahash_sim = compute_similarity_with_weights(
            &self.ahash,
            &other.ahash,
            config.global_weight,
            config.block_weight,
            config.block_distance_threshold,
        )
        .score;
        let phash_sim = compute_similarity_with_weights(
            &self.phash,
            &other.phash,
            config.global_weight,
            config.block_weight,
            config.block_distance_threshold,
        )
        .score;
        let dhash_sim = compute_similarity_with_weights(
            &self.dhash,
            &other.dhash,
            config.global_weight,
            config.block_weight,
            config.block_distance_threshold,
        )
        .score;

        let weighted_score = ahash_sim * config.ahash_weight
            + phash_sim * config.phash_weight
            + dhash_sim * config.dhash_weight;

        let ahash_dist = hamming_distance(self.ahash.global_hash, other.ahash.global_hash);
        let phash_dist = hamming_distance(self.phash.global_hash, other.phash.global_hash);
        let dhash_dist = hamming_distance(self.dhash.global_hash, other.dhash.global_hash);

        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let avg_distance = ((ahash_dist as f32 * config.ahash_weight)
            + (phash_dist as f32 * config.phash_weight)
            + (dhash_dist as f32 * config.dhash_weight)) as u32;

        Similarity {
            score: weighted_score.clamp(0.0, 1.0),
            exact_match: false,
            perceptual_distance: avg_distance,
        }
    }

    /// Checks if this fingerprint is similar to another within a threshold.
    ///
    /// Uses the weighted combination score from compare().
    ///
    /// # Panics
    /// Panics in debug mode if threshold is not in [0.0, 1.0].
    /// In release mode, out-of-range or NaN thresholds return false.
    #[must_use]
    pub fn is_similar(&self, other: &MultiHashFingerprint, threshold: f32) -> bool {
        debug_assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in range [0.0, 1.0], got {}",
            threshold
        );
        let clamped_threshold = threshold.clamp(0.0, 1.0);
        self.compare(other).score >= clamped_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(global: u64, blocks_word: u64) -> ImageFingerprint {
        ImageFingerprint::new([0u8; 32], global, [blocks_word; 16])
    }

    fn multi(exact: [u8; 32], a_global: u64, p_global: u64, d_global: u64) -> MultiHashFingerprint {
        // Mirror production: per-algo ImageFingerprint.exact == outer exact.
        MultiHashFingerprint::new(
            exact,
            ImageFingerprint::new(exact, a_global, [a_global; 16]),
            ImageFingerprint::new(exact, p_global, [p_global; 16]),
            ImageFingerprint::new(exact, d_global, [d_global; 16]),
        )
    }

    #[test]
    fn multi_hash_config_default_matches_compare() {
        let a = multi([1u8; 32], 0xAAAA, 0xBBBB, 0xCCCC);
        let b = multi([2u8; 32], 0xAAAA, 0xBBB0, 0xCCC0);
        let default_score = a.compare(&b).score;
        let cfg_score = a.compare_with_config(&b, &MultiHashConfig::default()).score;
        assert!(
            (default_score - cfg_score).abs() < 1e-6,
            "{default_score} vs {cfg_score}"
        );
    }

    #[test]
    fn multi_hash_config_phash_only_ignores_other_algorithms() {
        // a and b share PHash exactly but differ wildly on AHash and DHash.
        let a = multi([1u8; 32], 0x0000_0000, 0x1234_5678, 0x0000_0000);
        let b = multi([2u8; 32], u64::MAX, 0x1234_5678, u64::MAX);

        let default_score = a.compare(&b).score;
        let phash_only = MultiHashConfig {
            ahash_weight: 0.0,
            phash_weight: 1.0,
            dhash_weight: 0.0,
            ..MultiHashConfig::default()
        };
        let phash_score = a.compare_with_config(&b, &phash_only).score;

        // PHash-only score must be 1.0 (perfect PHash match).
        assert!((phash_score - 1.0).abs() < 1e-6, "got {phash_score}");
        // Default score gets dragged down by AHash/DHash divergence.
        assert!(
            default_score < phash_score,
            "{default_score} >= {phash_score}"
        );
    }

    #[test]
    fn multi_hash_config_exact_match_is_always_one() {
        let a = multi([7u8; 32], 0xAAAA, 0xBBBB, 0xCCCC);
        let weird = MultiHashConfig {
            ahash_weight: 0.0,
            phash_weight: 0.0,
            dhash_weight: 0.0,
            global_weight: 0.0,
            block_weight: 0.0,
            block_distance_threshold: 0,
        };
        let s = a.compare_with_config(&a, &weird);
        assert!(s.exact_match);
        assert_eq!(s.score, 1.0);
    }

    #[test]
    fn multi_hash_config_score_clamped_to_unit_interval() {
        // Inflated weights would naively produce > 1.0; final score must clamp.
        let a = multi([1u8; 32], 0, 0, 0);
        let b = multi([2u8; 32], 0, 0, 0);
        let cfg = MultiHashConfig {
            ahash_weight: 5.0,
            phash_weight: 5.0,
            dhash_weight: 5.0,
            global_weight: 10.0,
            block_weight: 10.0,
            block_distance_threshold: 32,
        };
        let s = a.compare_with_config(&b, &cfg);
        assert!(s.score <= 1.0 && s.score >= 0.0, "got {}", s.score);
    }

    #[test]
    fn fingerprint_unused_helper_compiles() {
        // Keeps `fp()` referenced so the helper test util doesn't bit-rot.
        let _ = fp(0x1234, 0xABCD);
    }

    #[test]
    fn format_version_is_one() {
        assert_eq!(crate::FORMAT_VERSION, 1);
        assert_eq!(ImageFingerprint::format_version(), 1);
        assert_eq!(MultiHashFingerprint::format_version(), 1);
    }

    #[test]
    fn image_fingerprint_layout_is_stable() {
        assert_eq!(core::mem::size_of::<ImageFingerprint>(), 168);
        assert_eq!(core::mem::align_of::<ImageFingerprint>(), 8);
    }

    #[test]
    fn multi_hash_fingerprint_layout_is_stable() {
        assert_eq!(core::mem::size_of::<MultiHashFingerprint>(), 536);
        assert_eq!(core::mem::align_of::<MultiHashFingerprint>(), 8);
    }

    #[test]
    fn image_fingerprint_cast_slice_roundtrips() {
        let fps = vec![
            ImageFingerprint::new([1u8; 32], 0xAAAA_BBBB_CCCC_DDDD, [0x1234; 16]),
            ImageFingerprint::new([2u8; 32], 0xDEAD_BEEF_CAFE_BABE, [0xFEDC; 16]),
            ImageFingerprint::new([3u8; 32], 0, [0; 16]),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&fps);
        assert_eq!(bytes.len(), 3 * 168);

        let back: &[ImageFingerprint] = bytemuck::cast_slice(bytes);
        assert_eq!(back.len(), fps.len());
        assert_eq!(back, &fps[..]);
    }

    #[test]
    fn multi_hash_fingerprint_cast_slice_roundtrips() {
        let fps = vec![
            multi([1u8; 32], 0x1111, 0x2222, 0x3333),
            multi([2u8; 32], 0xAAAA, 0xBBBB, 0xCCCC),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&fps);
        assert_eq!(bytes.len(), 2 * 536);

        let back: &[MultiHashFingerprint] = bytemuck::cast_slice(bytes);
        assert_eq!(back.len(), fps.len());
        assert_eq!(back, &fps[..]);
    }

    #[test]
    fn fingerprint_zeroed_is_valid() {
        // Zeroable means an all-zero bit pattern is a valid value of the type.
        let z: MultiHashFingerprint = bytemuck::Zeroable::zeroed();
        assert_eq!(*z.exact_hash(), [0u8; 32]);
        assert_eq!(z.ahash().global_hash(), 0);
    }
}
