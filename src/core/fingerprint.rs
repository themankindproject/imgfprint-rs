use crate::core::similarity::Similarity;
use crate::error::ImgFprintError;
use crate::hash::algorithms::HashAlgorithm;

/// Writes a byte slice as lowercase hex.
fn write_hex(f: &mut core::fmt::Formatter<'_>, bytes: &[u8]) -> core::fmt::Result {
    for byte in bytes {
        write!(f, "{:02x}", byte)?;
    }
    Ok(())
}

/// `score >= threshold` for a threshold in `[0.0, 1.0]`; any other threshold
/// (negative, above 1.0, NaN) never matches.
#[inline]
fn meets_threshold(score: f32, threshold: f32) -> bool {
    (0.0..=1.0).contains(&threshold) && score >= threshold
}

/// Magic prefix of the [`ImageFingerprint::to_bytes`] /
/// [`MultiHashFingerprint::to_bytes`] encodings.
const CODEC_MAGIC: [u8; 2] = *b"IF";
/// Codec kind byte: single-algorithm [`ImageFingerprint`].
const CODEC_KIND_SINGLE: u8 = 1;
/// Codec kind byte: [`MultiHashFingerprint`].
const CODEC_KIND_MULTI: u8 = 3;
/// Header length: magic (2) + format version (1) + kind (1).
const CODEC_HEADER_LEN: usize = 4;
/// Encoded payload of one [`ImageFingerprint`]: exact (32) + global (8) + 16 blocks (128).
const SINGLE_PAYLOAD_LEN: usize = 32 + 8 + 16 * 8;

fn codec_header(kind: u8) -> [u8; CODEC_HEADER_LEN] {
    #[allow(clippy::cast_possible_truncation)] // asserted below: FORMAT_VERSION fits in u8
    let version = crate::FORMAT_VERSION as u8;
    [CODEC_MAGIC[0], CODEC_MAGIC[1], version, kind]
}

const _: () = assert!(crate::FORMAT_VERSION <= u8::MAX as u32);

/// Validates length, magic, version, and kind; returns the payload.
fn check_codec_header<'a>(
    bytes: &'a [u8],
    kind: u8,
    expected_len: usize,
    type_name: &str,
) -> Result<&'a [u8], ImgFprintError> {
    if bytes.len() != expected_len {
        return Err(ImgFprintError::InvalidFingerprint(format!(
            "{type_name} encoding must be {expected_len} bytes, got {}",
            bytes.len()
        )));
    }
    if bytes[..2] != CODEC_MAGIC {
        return Err(ImgFprintError::InvalidFingerprint(
            "missing imgfprint magic bytes".to_string(),
        ));
    }
    if u32::from(bytes[2]) != crate::FORMAT_VERSION {
        return Err(ImgFprintError::InvalidFingerprint(format!(
            "format version {} is not supported (this build reads version {}); recompute the fingerprint",
            bytes[2],
            crate::FORMAT_VERSION
        )));
    }
    if bytes[3] != kind {
        return Err(ImgFprintError::InvalidFingerprint(format!(
            "encoding holds kind {}, not a {type_name}",
            bytes[3]
        )));
    }
    Ok(&bytes[CODEC_HEADER_LEN..])
}

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

impl MultiHashConfig {
    /// Validates the configuration, rejecting values that would produce
    /// meaningless or NaN similarity scores.
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError::InvalidConfig`] when:
    /// - any weight (`ahash_weight`, `phash_weight`, `dhash_weight`,
    ///   `global_weight`, `block_weight`) is NaN, infinite, or negative —
    ///   NaN weights poison the score (NaN propagates through the weighted
    ///   sum and makes `is_similar` silently return `false`), infinite
    ///   weights saturate the score at `1.0` regardless of content, and
    ///   negative weights invert similarity semantics;
    /// - `block_distance_threshold` exceeds 64 (the maximum Hamming distance
    ///   between two 64-bit hashes).
    ///
    /// # Example
    ///
    /// ```rust
    /// use imgfprint::MultiHashConfig;
    ///
    /// let cfg = MultiHashConfig::default();
    /// assert!(cfg.validate().is_ok());
    ///
    /// let bad = MultiHashConfig { phash_weight: f32::NAN, ..cfg };
    /// assert!(bad.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), crate::error::ImgFprintError> {
        let weights = [
            ("ahash_weight", self.ahash_weight),
            ("phash_weight", self.phash_weight),
            ("dhash_weight", self.dhash_weight),
            ("global_weight", self.global_weight),
            ("block_weight", self.block_weight),
        ];
        for (name, value) in weights {
            if value.is_nan() {
                return Err(crate::error::ImgFprintError::invalid_config(format!(
                    "{name} is NaN"
                )));
            }
            if value.is_infinite() {
                return Err(crate::error::ImgFprintError::invalid_config(format!(
                    "{name} is infinite ({value})"
                )));
            }
            if value < 0.0 {
                return Err(crate::error::ImgFprintError::invalid_config(format!(
                    "{name} is negative ({value})"
                )));
            }
        }
        if self.block_distance_threshold > 64 {
            return Err(crate::error::ImgFprintError::invalid_config(format!(
                "block_distance_threshold ({}) exceeds maximum 64",
                self.block_distance_threshold
            )));
        }
        Ok(())
    }

    /// Returns a sanitized copy of this config that is safe to score with.
    ///
    /// - NaN or infinite weights become `0.0` (the algorithm is excluded from
    ///   the score).
    /// - Negative weights are clamped to `0.0`.
    /// - `block_distance_threshold` is clamped to `0..=64`.
    ///
    /// Use this when configs come from untrusted sources (e.g. deserialized
    /// user input) and you prefer best-effort scoring over rejection.
    /// [`validate`](Self::validate) is the strict alternative.
    #[must_use]
    pub fn sanitized(&self) -> Self {
        let sanitize = |v: f32| {
            if !v.is_finite() || v < 0.0 {
                0.0
            } else {
                v
            }
        };
        Self {
            ahash_weight: sanitize(self.ahash_weight),
            phash_weight: sanitize(self.phash_weight),
            dhash_weight: sanitize(self.dhash_weight),
            global_weight: sanitize(self.global_weight),
            block_weight: sanitize(self.block_weight),
            block_distance_threshold: self.block_distance_threshold.min(64),
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
    /// BLAKE3 hash for exact-match detection.
    ///
    /// **Semantics depend on the construction path:**
    /// - When produced by [`ImageFingerprinter::fingerprint`] (or any method
    ///   accepting `&[u8]` file bytes), this is the BLAKE3 digest of the raw
    ///   compressed file bytes. Different encodings of the same pixels yield
    ///   different values.
    /// - When produced by [`ImageFingerprinter::fingerprint_image`] (accepting
    ///   a decoded `DynamicImage`), this is the BLAKE3 digest of the RGB8
    ///   pixel buffer. Identical pixels always yield the same value.
    pub(crate) exact: [u8; 32],
    pub(crate) global_hash: u64,
    pub(crate) block_hashes: [u64; 16],
}

impl ImageFingerprint {
    /// Length of [`to_bytes`](Self::to_bytes) output: 4-byte header + 168-byte payload.
    pub const ENCODED_LEN: usize = CODEC_HEADER_LEN + SINGLE_PAYLOAD_LEN;

    #[inline]
    pub(crate) fn new(exact: [u8; 32], global_hash: u64, block_hashes: [u64; 16]) -> Self {
        Self {
            exact,
            global_hash,
            block_hashes,
        }
    }

    /// Returns the BLAKE3 exact-match hash.
    ///
    /// Covers the raw input bytes for byte entry points and the RGB8 pixels
    /// for [`ImageFingerprinter::fingerprint_image`]. Equal exact hashes mean
    /// identical input; use it for exact deduplication before perceptual
    /// comparison.
    ///
    /// [`ImageFingerprinter::fingerprint_image`]: crate::ImageFingerprinter::fingerprint_image
    #[inline]
    #[must_use]
    pub fn exact_hash(&self) -> &[u8; 32] {
        &self.exact
    }

    /// Returns the on-disk format version this fingerprint was computed under.
    #[deprecated(since = "0.4.7", note = "use the `imgfprint::FORMAT_VERSION` constant")]
    #[inline]
    #[must_use]
    pub const fn format_version() -> u32 {
        crate::FORMAT_VERSION
    }

    /// Returns the global perceptual hash of the center 32x32 region.
    ///
    /// Captures the overall structure of the image; which algorithm produced
    /// it depends on how the fingerprint was created.
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

    /// Returns a coarse locality key for fast bucket-based deduplication.
    ///
    /// Extracts the top `bucket_bits` bits from the global perceptual hash,
    /// producing a key that places perceptually similar images into the same
    /// bucket. Use as the first stage of a multi-stage deduplication index
    /// where the full block-by-block comparison is only run against candidates
    /// sharing the same coarse key.
    ///
    /// # Arguments
    ///
    /// * `bucket_bits` — Number of high-order bits to extract (0–64).
    ///   More bits → more buckets → fewer false candidates per bucket but
    ///   higher risk of splitting true near-duplicates across buckets.
    ///   Typical values: 8–16 for million-image corpora.
    ///
    /// # Behavior
    ///
    /// - `coarse_key(0)` returns `0` (single bucket — everything matches).
    /// - `coarse_key(64)` returns the full `global_hash` (finest granularity).
    /// - Values > 64 are clamped to 64 (full hash) in all build modes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use imgfprint::ImageFingerprint;
    ///
    /// fn build_index(fingerprints: &[ImageFingerprint]) -> HashMap<u64, Vec<usize>> {
    ///     let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    ///     for (idx, fp) in fingerprints.iter().enumerate() {
    ///         buckets.entry(fp.coarse_key(16)).or_default().push(idx);
    ///     }
    ///     buckets
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub fn coarse_key(&self, bucket_bits: u32) -> u64 {
        let bits = bucket_bits.min(64);
        if bits == 0 {
            0
        } else {
            self.global_hash >> (64 - bits)
        }
    }

    /// Computes the Hamming distance between this and another fingerprint's global hash.
    ///
    /// Returns a value from 0 (identical) to 64 (completely different).
    #[inline]
    #[must_use]
    pub fn distance(&self, other: &ImageFingerprint) -> u32 {
        (self.global_hash ^ other.global_hash).count_ones()
    }

    /// Compares two single-algorithm fingerprints.
    ///
    /// The score blends 40% global-hash similarity with 60% block similarity
    /// (blocks farther than 32 bits apart are ignored); an exact-hash match
    /// scores `1.0`. Only compare fingerprints made with the same algorithm.
    #[must_use]
    pub fn compare(&self, other: &ImageFingerprint) -> Similarity {
        crate::core::similarity::compute_similarity(self, other)
    }

    /// Returns `true` if [`compare`](Self::compare) scores at least
    /// `threshold`.
    ///
    /// `threshold` must lie in `[0.0, 1.0]`; any other value (negative,
    /// above 1.0, NaN) returns `false`.
    #[doc(alias = "match")]
    #[must_use]
    pub fn is_similar(&self, other: &ImageFingerprint, threshold: f32) -> bool {
        meets_threshold(self.compare(other).score, threshold)
    }

    /// Encodes the fingerprint as a compact, versioned, endian-independent
    /// byte string for storage or transport (database column, cache, RPC).
    ///
    /// Layout: `b"IF"`, [`FORMAT_VERSION`](crate::FORMAT_VERSION) (1 byte),
    /// kind (1 byte), then the exact hash and the 17 hashes as little-endian
    /// `u64`s. [`from_bytes`](Self::from_bytes) is the exact inverse.
    ///
    /// ```rust
    /// # use imgfprint::{ImageFingerprinter, HashAlgorithm, ImageFingerprint};
    /// # let png = {
    /// #     let img = image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, 7]));
    /// #     let mut buf = Vec::new();
    /// #     img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
    /// #     buf
    /// # };
    /// let fp = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::PHash)?;
    /// let stored: [u8; ImageFingerprint::ENCODED_LEN] = fp.to_bytes();
    /// assert_eq!(ImageFingerprint::from_bytes(&stored)?, fp);
    /// # Ok::<(), imgfprint::ImgFprintError>(())
    /// ```
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::ENCODED_LEN] {
        let mut out = [0u8; Self::ENCODED_LEN];
        out[..CODEC_HEADER_LEN].copy_from_slice(&codec_header(CODEC_KIND_SINGLE));
        self.write_payload(&mut out[CODEC_HEADER_LEN..]);
        out
    }

    /// Decodes bytes produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// [`ImgFprintError::InvalidFingerprint`] if the length, magic bytes,
    /// kind, or format version do not match. A version mismatch means the
    /// fingerprint was computed by an incompatible algorithm revision and
    /// must be recomputed rather than compared.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ImgFprintError> {
        let payload = check_codec_header(
            bytes,
            CODEC_KIND_SINGLE,
            Self::ENCODED_LEN,
            "ImageFingerprint",
        )?;
        Ok(Self::read_payload(payload))
    }

    /// Writes the 168-byte little-endian payload into `out`.
    fn write_payload(&self, out: &mut [u8]) {
        out[..32].copy_from_slice(&self.exact);
        out[32..40].copy_from_slice(&self.global_hash.to_le_bytes());
        for (dst, hash) in out[40..SINGLE_PAYLOAD_LEN]
            .chunks_exact_mut(8)
            .zip(&self.block_hashes)
        {
            dst.copy_from_slice(&hash.to_le_bytes());
        }
    }

    /// Reads a 168-byte little-endian payload (length checked by the caller).
    fn read_payload(payload: &[u8]) -> Self {
        let u64_at = |offset: usize| {
            let mut word = [0u8; 8];
            word.copy_from_slice(&payload[offset..offset + 8]);
            u64::from_le_bytes(word)
        };
        let mut exact = [0u8; 32];
        exact.copy_from_slice(&payload[..32]);
        Self {
            exact,
            global_hash: u64_at(32),
            block_hashes: core::array::from_fn(|i| u64_at(40 + i * 8)),
        }
    }
}

impl core::fmt::Display for ImageFingerprint {
    /// Formats the fingerprint as hex: `exact:global:block0,block1,...,block15`
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Exact hash as hex
        write_hex(f, &self.exact)?;
        write!(f, ":{:016x}:", self.global_hash)?;
        for (i, h) in self.block_hashes.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{:016x}", h)?;
        }
        Ok(())
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
    /// BLAKE3 hash for exact-match detection.
    ///
    /// **Semantics depend on the construction path:**
    /// - Via [`ImageFingerprinter::fingerprint`]: BLAKE3 of raw compressed
    ///   file bytes. Two files with identical pixels but different encodings
    ///   produce different values.
    /// - Via [`ImageFingerprinter::fingerprint_image`]: BLAKE3 of the decoded
    ///   RGB8 pixel buffer. Identical pixels always produce the same value,
    ///   regardless of the original encoding.
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
    /// Length of [`to_bytes`](Self::to_bytes) output: 4-byte header + 536-byte payload.
    pub const ENCODED_LEN: usize = CODEC_HEADER_LEN + 32 + 3 * SINGLE_PAYLOAD_LEN;

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

    /// Returns the BLAKE3 exact-match hash (see [`ImageFingerprint::exact_hash`]).
    #[inline]
    #[must_use]
    pub fn exact_hash(&self) -> &[u8; 32] {
        &self.exact
    }

    /// Returns the on-disk format version this fingerprint was computed under.
    #[deprecated(since = "0.4.7", note = "use the `imgfprint::FORMAT_VERSION` constant")]
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
    /// Scores of unrelated images cluster around 0.55 (two independent 64-bit
    /// hashes differ in ~32 bits), so thresholds are best chosen from the
    /// measured guidance in the crate docs: ~0.85 for near-duplicates.
    #[must_use]
    pub fn compare(&self, other: &MultiHashFingerprint) -> Similarity {
        self.compare_with_config(other, &MultiHashConfig::default())
    }

    /// Compares two multi-hash fingerprints with a custom block distance threshold.
    #[deprecated(
        since = "0.4.7",
        note = "use `compare_with_config(other, &MultiHashConfig { block_distance_threshold, ..Default::default() })`"
    )]
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
    ///
    /// The config is sanitized before scoring: NaN/infinite/negative weights
    /// are treated as `0.0` and `block_distance_threshold` is clamped to `0..=64`,
    /// so a malformed config can never produce a NaN score. Use
    /// [`MultiHashConfig::validate`] if you need to reject bad configs
    /// instead.
    #[must_use]
    pub fn compare_with_config(
        &self,
        other: &MultiHashFingerprint,
        config: &MultiHashConfig,
    ) -> Similarity {
        use crate::core::similarity::{compute_score_only, hamming_distance};

        if self.exact == other.exact {
            return Similarity {
                score: 1.0,
                exact_match: true,
                perceptual_distance: 0,
            };
        }

        // Sanitize once up front; identity for valid configs.
        let config = config.sanitized();

        let ahash_sim = compute_score_only(
            &self.ahash,
            &other.ahash,
            config.global_weight,
            config.block_weight,
            config.block_distance_threshold,
        );
        let phash_sim = compute_score_only(
            &self.phash,
            &other.phash,
            config.global_weight,
            config.block_weight,
            config.block_distance_threshold,
        );
        let dhash_sim = compute_score_only(
            &self.dhash,
            &other.dhash,
            config.global_weight,
            config.block_weight,
            config.block_distance_threshold,
        );

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
        let avg_distance = {
            let weight_sum = config.ahash_weight + config.phash_weight + config.dhash_weight;
            let raw = (ahash_dist as f32 * config.ahash_weight)
                + (phash_dist as f32 * config.phash_weight)
                + (dhash_dist as f32 * config.dhash_weight);
            if weight_sum > 0.0 {
                (raw / weight_sum) as u32
            } else {
                0
            }
        };

        Similarity {
            score: weighted_score.clamp(0.0, 1.0),
            exact_match: false,
            perceptual_distance: avg_distance,
        }
    }

    /// Returns `true` if [`compare`](Self::compare) scores at least
    /// `threshold`.
    ///
    /// `threshold` must lie in `[0.0, 1.0]`; any other value (negative,
    /// above 1.0, NaN) returns `false`. See the crate-level docs for measured
    /// threshold guidance (~0.85 catches re-encodes, resizes, and small edits
    /// while keeping unrelated images out).
    #[must_use]
    pub fn is_similar(&self, other: &MultiHashFingerprint, threshold: f32) -> bool {
        meets_threshold(self.compare(other).score, threshold)
    }

    /// Encodes the fingerprint as a compact, versioned, endian-independent
    /// byte string for storage or transport (database column, cache, RPC).
    ///
    /// Layout: `b"IF"`, [`FORMAT_VERSION`](crate::FORMAT_VERSION) (1 byte),
    /// kind (1 byte), the exact hash, then the AHash, PHash, and DHash layers
    /// (each: exact hash + 17 little-endian `u64`s).
    /// [`from_bytes`](Self::from_bytes) is the exact inverse.
    ///
    /// Unlike a raw [`bytemuck`] cast this is portable across endianness and
    /// carries its format version, so stale fingerprints are rejected instead
    /// of silently mis-compared after an algorithm change.
    ///
    /// ```rust
    /// # use imgfprint::{ImageFingerprinter, MultiHashFingerprint};
    /// # let png = {
    /// #     let img = image::RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8, y as u8, 7]));
    /// #     let mut buf = Vec::new();
    /// #     img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png).unwrap();
    /// #     buf
    /// # };
    /// let fp = ImageFingerprinter::fingerprint(&png)?;
    /// let stored = fp.to_bytes(); // [u8; MultiHashFingerprint::ENCODED_LEN]
    /// assert_eq!(MultiHashFingerprint::from_bytes(&stored)?, fp);
    /// # Ok::<(), imgfprint::ImgFprintError>(())
    /// ```
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::ENCODED_LEN] {
        let mut out = [0u8; Self::ENCODED_LEN];
        out[..CODEC_HEADER_LEN].copy_from_slice(&codec_header(CODEC_KIND_MULTI));
        out[CODEC_HEADER_LEN..CODEC_HEADER_LEN + 32].copy_from_slice(&self.exact);
        let layers = &mut out[CODEC_HEADER_LEN + 32..];
        for (dst, layer) in
            layers
                .chunks_exact_mut(SINGLE_PAYLOAD_LEN)
                .zip([&self.ahash, &self.phash, &self.dhash])
        {
            layer.write_payload(dst);
        }
        out
    }

    /// Decodes bytes produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// [`ImgFprintError::InvalidFingerprint`] if the length, magic bytes,
    /// kind, or format version do not match.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ImgFprintError> {
        let payload = check_codec_header(
            bytes,
            CODEC_KIND_MULTI,
            Self::ENCODED_LEN,
            "MultiHashFingerprint",
        )?;
        let mut exact = [0u8; 32];
        exact.copy_from_slice(&payload[..32]);
        let layer = |i: usize| {
            let start = 32 + i * SINGLE_PAYLOAD_LEN;
            ImageFingerprint::read_payload(&payload[start..start + SINGLE_PAYLOAD_LEN])
        };
        Ok(Self::new(exact, layer(0), layer(1), layer(2)))
    }
}

impl core::fmt::Display for MultiHashFingerprint {
    /// Formats as `exact_hex|ahash_global|phash_global|dhash_global`
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write_hex(f, &self.exact)?;
        write!(
            f,
            "|{:016x}|{:016x}|{:016x}",
            self.ahash.global_hash, self.phash.global_hash, self.dhash.global_hash
        )
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
    #[allow(deprecated)] // pins the deprecated shim until removal
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

    #[test]
    fn image_fingerprint_display() {
        let fp = ImageFingerprint::new([0xABu8; 32], 0x1234_5678_9ABC_DEF0, [0xFF; 16]);
        let s = format!("{}", fp);
        assert!(s.starts_with("abababab"));
        assert!(s.contains(":123456789abcdef0:"));
        assert!(s.contains("00000000000000ff"));
    }

    #[test]
    fn multi_hash_fingerprint_display() {
        let m = multi([0x01u8; 32], 0xAAAA, 0xBBBB, 0xCCCC);
        let s = format!("{}", m);
        assert!(s.starts_with("01010101"));
        assert!(s.contains("|000000000000aaaa|"));
        assert!(s.contains("|000000000000bbbb|"));
        assert!(s.ends_with("000000000000cccc"));
    }

    #[test]
    fn is_similar_uses_block_hashes() {
        // Two fingerprints with identical global hash but very different blocks
        let fp1 = ImageFingerprint::new([1u8; 32], 0x1234, [0u64; 16]);
        let fp2 = ImageFingerprint::new([2u8; 32], 0x1234, [u64::MAX; 16]);
        // Global distance is 0, but blocks are maximally different (excluded by threshold)
        // With block weighting, score should be less than 1.0
        assert!(!fp1.is_similar(&fp2, 1.0));
        // But with a low threshold it should still pass
        assert!(fp1.is_similar(&fp2, 0.3));
    }

    #[test]
    fn perceptual_distance_bounded_with_inflated_weights() {
        let a = multi([1u8; 32], 0, 0xFFFF_FFFF_FFFF_FFFF, 0);
        let b = multi([2u8; 32], 0, 0, 0);
        let cfg = MultiHashConfig {
            ahash_weight: 0.0,
            phash_weight: 5.0,
            dhash_weight: 0.0,
            ..MultiHashConfig::default()
        };
        let s = a.compare_with_config(&b, &cfg);
        // Distance should be normalized: 64 * 5.0 / 5.0 = 64 (not 320)
        assert!(s.perceptual_distance <= 64, "got {}", s.perceptual_distance);
    }

    #[test]
    fn perceptual_distance_zero_weights() {
        let a = multi([1u8; 32], 0xFFFF, 0xFFFF, 0xFFFF);
        let b = multi([2u8; 32], 0, 0, 0);
        let cfg = MultiHashConfig {
            ahash_weight: 0.0,
            phash_weight: 0.0,
            dhash_weight: 0.0,
            ..MultiHashConfig::default()
        };
        let s = a.compare_with_config(&b, &cfg);
        assert_eq!(s.perceptual_distance, 0);
    }

    #[test]
    fn coarse_key_bit_extraction_table() {
        // (hash, bits, expected): top-N-bits extraction, edge widths, clamping.
        let hash = 0xDEAD_BEEF_CAFE_BABE;
        for (h, bits, expected) in [
            (hash, 0u32, 0u64),
            (hash, 1, (hash >> 63) & 1),
            (hash, 8, 0xDE),
            (hash, 16, 0xDEAD),
            (hash, 32, 0xDEAD_BEEF),
            (hash, 64, hash),
            (hash, 65, hash),
            (hash, u32::MAX, hash),
            (0x8000_0000_0000_0000, 1, 1),
            (0x7FFF_FFFF_FFFF_FFFF, 1, 0),
            (0xAB00_0000_0000_0000, 8, 0xAB),
        ] {
            assert_eq!(fp(h, 0).coarse_key(bits), expected, "bits={bits}");
        }

        // Near-identical hashes share the coarse bucket; extraction is pure.
        let f = fp(0x1234_5678_9ABC_DEF0, 0);
        assert_eq!(f.coarse_key(16), f.coarse_key(16));
        let f1 = fp(0xAAAA_BBBB_0000_0001, 0);
        let f2 = fp(0xAAAA_BBBB_FFFF_FFFE, 0);
        assert_eq!(f1.coarse_key(32), f2.coarse_key(32));
    }
}
