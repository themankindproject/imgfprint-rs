/// A perceptual fingerprint containing multiple hash layers for robust comparison.
///
/// Fingerprints are deterministic and comparable across platforms. The structure
/// includes exact hashing for identical detection and perceptual hashing for
/// similarity detection with resistance to resizing, compression, and cropping.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ImageFingerprint {
    pub(crate) exact: [u8; 32],
    pub(crate) global_phash: u64,
    pub(crate) block_hashes: [u64; 16],
}

impl ImageFingerprint {
    #[inline]
    pub(crate) fn new(exact: [u8; 32], global_phash: u64, block_hashes: [u64; 16]) -> Self {
        Self {
            exact,
            global_phash,
            block_hashes,
        }
    }

    /// Returns the SHA256 hash of the original image bytes.
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
    /// to minor changes in compression and color adjustments.
    #[inline]
    pub fn global_phash(&self) -> u64 {
        self.global_phash
    }

    /// Returns the 16 block-level perceptual hashes from a 4x4 grid.
    ///
    /// Block hashes enable crop-resistant comparison by matching partial
    /// regions between images. Each hash covers a 64x64 pixel region.
    #[inline]
    pub fn block_hashes(&self) -> &[u64; 16] {
        &self.block_hashes
    }
}
