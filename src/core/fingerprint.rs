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

    /// Computes the Hamming distance between this and another fingerprint's global hash.
    ///
    /// Returns a value from 0 (identical) to 64 (completely different).
    #[inline]
    pub fn distance(&self, other: &ImageFingerprint) -> u32 {
        (self.global_phash ^ other.global_phash).count_ones()
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
    /// use imgfprint_rs::ImageFingerprinter;
    ///
    /// let fp1 = ImageFingerprinter::fingerprint(&std::fs::read("image1.jpg")?).unwrap();
    /// let fp2 = ImageFingerprinter::fingerprint(&std::fs::read("image2.jpg")?).unwrap();
    ///
    /// if fp1.is_similar(&fp2, 0.8) {
    ///     println!("Images are similar!");
    /// }
    /// ```
    pub fn is_similar(&self, other: &ImageFingerprint, threshold: f32) -> bool {
        if self.exact == other.exact {
            return true;
        }
        let dist = self.distance(other);
        let similarity = 1.0 - (dist as f32 / 64.0);
        similarity >= threshold
    }
}
