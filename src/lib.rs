//! High-performance, deterministic image fingerprinting library.
//!
//! Provides perceptual hashing with crop resistance for image
//! deduplication and similarity detection in production systems.
//!
//! Supports multiple algorithms (`PHash`, `DHash`) with automatic parallel
//! computation for improved accuracy.
//!
//! ## Quick Start
//!
//! Compute all hashes (recommended for best accuracy):
//!
//! ```rust
//! use imgfprint::ImageFingerprinter;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let fp1 = ImageFingerprinter::fingerprint(&std::fs::read("img1.jpg")?)?;
//! let fp2 = ImageFingerprinter::fingerprint(&std::fs::read("img2.jpg")?)?;
//!
//! let sim = fp1.compare(&fp2);
//! println!("Similarity: {:.2}", sim.score);
//! # Ok(())
//! # }
//! ```
//!
//! Compute specific algorithm only:
//!
//! ```rust
//! use imgfprint::{ImageFingerprinter, HashAlgorithm};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let fp = ImageFingerprinter::fingerprint_with(
//!     &std::fs::read("img1.jpg")?,
//!     HashAlgorithm::DHash
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## High-Throughput Usage
//!
//! For processing many images, use [`FingerprinterContext`] to enable
//! buffer reuse and avoid repeated allocations:
//!
//! ```rust
//! use imgfprint::FingerprinterContext;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut ctx = FingerprinterContext::new();
//!
//! // Process multiple images efficiently
//! for path in &["img1.jpg", "img2.jpg", "img3.jpg"] {
//!     let fp = ctx.fingerprint(&std::fs::read(path)?)?;
//!     // Use fingerprint...
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Features
//!
//! - **SIMD-accelerated resize**: Uses AVX2/NEON instructions for 3-4x faster resizing
//! - **Cached DCT plans**: Reuses DCT computation plans across all calls
//! - **Buffer reuse**: Context API minimizes allocations in high-throughput scenarios
//! - **Parallel processing**: Batch operations use rayon for multi-core speedup

#![deny(missing_docs)]

mod core;
mod embed;
mod error;
mod hash;
mod imgproc;

pub use core::fingerprint::{ImageFingerprint, MultiHashFingerprint};
pub use core::fingerprinter::{FingerprinterContext, ImageFingerprinter};
pub use core::similarity::Similarity;
pub use embed::{semantic_similarity, Embedding, EmbeddingProvider};
pub use hash::algorithms::HashAlgorithm;

#[cfg(feature = "local-embedding")]
pub use embed::{LocalProvider, LocalProviderConfig};

pub use error::ImgFprintError;

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_image() -> Vec<u8> {
        let img = ImageBuffer::from_fn(300, 300, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn test_multi_hash_deterministic() {
        let img = create_test_image();
        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();

        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        assert_eq!(fp1.phash().global_hash(), fp2.phash().global_hash());
        assert_eq!(fp1.dhash().global_hash(), fp2.dhash().global_hash());
    }

    #[test]
    fn test_single_algorithm_deterministic() {
        let img = create_test_image();

        let fp1 = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::PHash).unwrap();
        let fp2 = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::PHash).unwrap();

        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        assert_eq!(fp1.global_hash(), fp2.global_hash());
        assert_eq!(fp1.block_hashes(), fp2.block_hashes());

        let fp3 = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::DHash).unwrap();
        let fp4 = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::DHash).unwrap();

        assert_eq!(fp3.global_hash(), fp4.global_hash());
    }

    #[test]
    fn test_multi_hash_identical_images() {
        let img = create_test_image();
        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();
        let sim = fp1.compare(&fp2);

        assert!(sim.exact_match);
        assert_eq!(sim.score, 1.0);
        assert_eq!(sim.perceptual_distance, 0);
    }

    #[test]
    fn test_single_hash_identical_images() {
        let img = create_test_image();
        let fp1 = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::PHash).unwrap();
        let fp2 = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::PHash).unwrap();
        let sim = ImageFingerprinter::compare(&fp1, &fp2);

        assert!(sim.exact_match);
        assert_eq!(sim.score, 1.0);
        assert_eq!(sim.perceptual_distance, 0);
    }

    #[test]
    fn test_similar_images_multi_hash() {
        let img1: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([100, 150, 200]));
        let mut buf1 = Vec::new();
        img1.write_to(
            &mut std::io::Cursor::new(&mut buf1),
            image::ImageFormat::Png,
        )
        .unwrap();

        let img2: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([102, 148, 198]));
        let mut buf2 = Vec::new();
        img2.write_to(
            &mut std::io::Cursor::new(&mut buf2),
            image::ImageFormat::Png,
        )
        .unwrap();

        let fp1 = ImageFingerprinter::fingerprint(&buf1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&buf2).unwrap();
        let sim = fp1.compare(&fp2);

        assert!(!sim.exact_match);
        assert!(sim.score > 0.5, "Similar images: got {}", sim.score);
    }

    #[test]
    fn test_similar_images_single_hash() {
        let img1: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([100, 150, 200]));
        let mut buf1 = Vec::new();
        img1.write_to(
            &mut std::io::Cursor::new(&mut buf1),
            image::ImageFormat::Png,
        )
        .unwrap();

        let img2: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([102, 148, 198]));
        let mut buf2 = Vec::new();
        img2.write_to(
            &mut std::io::Cursor::new(&mut buf2),
            image::ImageFormat::Png,
        )
        .unwrap();

        let fp1 = ImageFingerprinter::fingerprint_with(&buf1, HashAlgorithm::PHash).unwrap();
        let fp2 = ImageFingerprinter::fingerprint_with(&buf2, HashAlgorithm::PHash).unwrap();
        let sim = ImageFingerprinter::compare(&fp1, &fp2);

        assert!(!sim.exact_match);
        assert!(sim.score > 0.5, "Similar images: got {}", sim.score);
    }

    #[test]
    fn test_different_images() {
        let img1: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |x, _| Rgb([x as u8, x as u8, x as u8]));
        let mut buf1 = Vec::new();
        img1.write_to(
            &mut std::io::Cursor::new(&mut buf1),
            image::ImageFormat::Png,
        )
        .unwrap();

        let img2: ImageBuffer<image::Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(300, 300, |x, _| {
            if x % 2 == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            }
        });
        let mut buf2 = Vec::new();
        img2.write_to(
            &mut std::io::Cursor::new(&mut buf2),
            image::ImageFormat::Png,
        )
        .unwrap();

        let fp1 = ImageFingerprinter::fingerprint(&buf1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&buf2).unwrap();
        let sim = fp1.compare(&fp2);

        assert!(!sim.exact_match);
        assert!(
            sim.perceptual_distance > 0,
            "Different patterns: got {}",
            sim.perceptual_distance
        );
    }

    #[test]
    fn test_resized_image() {
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(400, 400, |x, y| {
            Rgb([
                ((x * y) % 256) as u8,
                ((x * y) % 256) as u8,
                ((x * y) % 256) as u8,
            ])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let resized =
            image::imageops::resize(&img, 200, 200, image::imageops::FilterType::Lanczos3);
        let mut buf_resized = Vec::new();
        resized
            .write_to(
                &mut std::io::Cursor::new(&mut buf_resized),
                image::ImageFormat::Png,
            )
            .unwrap();

        let fp1 = ImageFingerprinter::fingerprint(&buf).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&buf_resized).unwrap();
        let sim = fp1.compare(&fp2);

        assert!(sim.score > 0.6, "Resized similarity: got {}", sim.score);
        assert!(!sim.exact_match);
    }

    #[test]
    fn test_compressed_image() {
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([128, 64, 200]));
        let mut buf = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buf),
            image::ImageFormat::Jpeg,
        )
        .unwrap();

        let fp1 = ImageFingerprinter::fingerprint(&buf).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&buf).unwrap();
        let sim = fp1.compare(&fp2);

        assert!(sim.score > 0.8, "JPEG compression: got {}", sim.score);
    }

    #[test]
    fn test_error_empty_input() {
        let result = ImageFingerprinter::fingerprint(&[]);
        assert!(matches!(result, Err(ImgFprintError::InvalidImage(_))));
    }

    #[test]
    fn test_error_invalid_data() {
        let result = ImageFingerprinter::fingerprint(b"not an image");
        assert!(matches!(
            result,
            Err(ImgFprintError::DecodeError(_) | ImgFprintError::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn test_fingerprint_accessors() {
        let fp = ImageFingerprinter::fingerprint(&create_test_image()).unwrap();
        assert_eq!(fp.exact_hash().len(), 32);
        assert_eq!(fp.phash().block_hashes().len(), 16);
        assert_eq!(fp.dhash().block_hashes().len(), 16);
    }

    #[test]
    fn test_single_fingerprint_accessors() {
        let fp = ImageFingerprinter::fingerprint_with(&create_test_image(), HashAlgorithm::PHash)
            .unwrap();
        assert_eq!(fp.exact_hash().len(), 32);
        assert_eq!(fp.block_hashes().len(), 16);
    }

    #[test]
    fn test_error_image_too_small() {
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(31, 31, |_, _| Rgb([128, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let result = ImageFingerprinter::fingerprint(&buf);
        assert!(
            matches!(result, Err(ImgFprintError::ImageTooSmall(_))),
            "Expected ImageTooSmall error, got {:?}",
            result
        );
    }

    #[test]
    fn test_error_image_too_large() {
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(8193, 100, |_, _| Rgb([128, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let result = ImageFingerprinter::fingerprint(&buf);
        assert!(
            matches!(result, Err(ImgFprintError::InvalidImage(_))),
            "Expected InvalidImage error for oversized image, got {:?}",
            result
        );
    }

    #[test]
    fn test_minimum_valid_image() {
        let img: ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(32, 32, |_, _| Rgb([128, 128, 128]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let result = ImageFingerprinter::fingerprint(&buf);
        assert!(
            result.is_ok(),
            "32x32 image should be valid, got error: {:?}",
            result
        );
    }

    #[test]
    fn test_rgba_image_handling() {
        use image::Rgba;
        let img: ImageBuffer<image::Rgba<u8>, Vec<u8>> = ImageBuffer::from_fn(300, 300, |x, y| {
            let alpha = if (x + y) % 2 == 0 { 255 } else { 128 };
            Rgba([128, 64, 200, alpha])
        });
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let result = ImageFingerprinter::fingerprint(&buf);
        assert!(
            result.is_ok(),
            "RGBA image should be processable: {:?}",
            result
        );
    }

    #[test]
    fn test_grayscale_image_handling() {
        use image::Luma;
        let img: ImageBuffer<image::Luma<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |x, y| Luma([((x + y) % 256) as u8]));
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();

        let result = ImageFingerprinter::fingerprint(&buf);
        assert!(
            result.is_ok(),
            "Grayscale image should be processable: {:?}",
            result
        );
    }

    #[test]
    fn test_determinism_across_multiple_calls() {
        let img = create_test_image();

        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp3 = ImageFingerprinter::fingerprint(&img).unwrap();

        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        assert_eq!(fp1.exact_hash(), fp3.exact_hash());
        assert_eq!(fp1.phash().global_hash(), fp2.phash().global_hash());
        assert_eq!(fp1.dhash().global_hash(), fp3.dhash().global_hash());
    }

    #[test]
    fn test_context_api_determinism() {
        let img = create_test_image();

        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();

        let mut ctx = FingerprinterContext::new();
        let fp2 = ctx.fingerprint(&img).unwrap();

        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        assert_eq!(fp1.phash().global_hash(), fp2.phash().global_hash());
        assert_eq!(fp1.dhash().global_hash(), fp2.dhash().global_hash());
    }

    #[test]
    fn test_multi_hash_similarity() {
        let img = create_test_image();
        let fp = ImageFingerprinter::fingerprint(&img).unwrap();

        assert!(fp.is_similar(&fp, 1.0));
        assert!(fp.is_similar(&fp, 0.0));
    }

    #[test]
    fn test_single_hash_similarity() {
        let img = create_test_image();
        let fp = ImageFingerprinter::fingerprint_with(&img, HashAlgorithm::PHash).unwrap();

        assert!(fp.is_similar(&fp, 1.0));
        assert!(fp.is_similar(&fp, 0.0));
        assert_eq!(fp.distance(&fp), 0);
    }

    #[test]
    fn test_corrupted_image_handling() {
        let mut corrupted = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        corrupted.extend_from_slice(&[0u8; 100]);

        let result = ImageFingerprinter::fingerprint(&corrupted);
        assert!(
            matches!(
                result,
                Err(ImgFprintError::DecodeError(_) | ImgFprintError::InvalidImage(_))
            ),
            "Corrupted image should return error, got {:?}",
            result
        );
    }

    #[test]
    fn test_determinism_1000_iterations() {
        let img = create_test_image();
        let expected_fp = ImageFingerprinter::fingerprint(&img).unwrap();

        for i in 0..1000 {
            let fp = ImageFingerprinter::fingerprint(&img).unwrap();
            assert_eq!(
                fp.exact_hash(),
                expected_fp.exact_hash(),
                "Exact hash mismatch at iteration {}",
                i
            );
            assert_eq!(
                fp.phash().global_hash(),
                expected_fp.phash().global_hash(),
                "PHash mismatch at iteration {}",
                i
            );
            assert_eq!(
                fp.dhash().global_hash(),
                expected_fp.dhash().global_hash(),
                "DHash mismatch at iteration {}",
                i
            );
        }
    }

    #[test]
    fn test_determinism_various_sizes() {
        use image::{ImageBuffer, Rgb};

        let sizes = [
            (32u32, 32u32),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
        ];

        for (width, height) in sizes.iter() {
            let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
                ImageBuffer::from_fn(*width, *height, |x, y| {
                    Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
                });
            let mut buf = Vec::new();
            img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();

            let fp1 = ImageFingerprinter::fingerprint(&buf).unwrap();
            let fp2 = ImageFingerprinter::fingerprint(&buf).unwrap();

            assert_eq!(
                fp1.exact_hash(),
                fp2.exact_hash(),
                "Size {}x{}: exact hash not deterministic",
                width,
                height
            );
            assert_eq!(
                fp1.phash().global_hash(),
                fp2.phash().global_hash(),
                "Size {}x{}: phash not deterministic",
                width,
                height
            );
            assert_eq!(
                fp1.dhash().global_hash(),
                fp2.dhash().global_hash(),
                "Size {}x{}: dhash not deterministic",
                width,
                height
            );
        }
    }

    #[test]
    fn test_determinism_with_context_reuse() {
        let img = create_test_image();
        let mut ctx = FingerprinterContext::new();

        let expected_fp = ctx.fingerprint(&img).unwrap();

        for i in 0..100 {
            let fp = ctx.fingerprint(&img).unwrap();
            assert_eq!(
                fp.exact_hash(),
                expected_fp.exact_hash(),
                "Context reuse: exact hash mismatch at iteration {}",
                i
            );
            assert_eq!(
                fp.phash().global_hash(),
                expected_fp.phash().global_hash(),
                "Context reuse: phash mismatch at iteration {}",
                i
            );
            assert_eq!(
                fp.dhash().global_hash(),
                expected_fp.dhash().global_hash(),
                "Context reuse: dhash mismatch at iteration {}",
                i
            );
        }
    }

    #[test]
    fn test_determinism_across_similar_images() {
        use image::{ImageBuffer, Rgb};

        let img1: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([100, 150, 200]));
        let img2: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(300, 300, |_, _| Rgb([102, 148, 198]));

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();
        img1.write_to(
            &mut std::io::Cursor::new(&mut buf1),
            image::ImageFormat::Png,
        )
        .unwrap();
        img2.write_to(
            &mut std::io::Cursor::new(&mut buf2),
            image::ImageFormat::Png,
        )
        .unwrap();

        let fp1 = ImageFingerprinter::fingerprint(&buf1).unwrap();
        let fp1_again = ImageFingerprinter::fingerprint(&buf1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&buf2).unwrap();

        assert_eq!(fp1.phash().global_hash(), fp1_again.phash().global_hash());

        let sim1 = fp1.compare(&fp2);
        let sim2 = fp1.compare(&fp2);

        assert_eq!(sim1.score, sim2.score, "Similarity score not deterministic");
        assert_eq!(
            sim1.perceptual_distance, sim2.perceptual_distance,
            "Perceptual distance not deterministic"
        );
    }

    #[test]
    fn test_algorithm_accessor() {
        let img = create_test_image();
        let fp = ImageFingerprinter::fingerprint(&img).unwrap();

        let phash_fp = fp.get(HashAlgorithm::PHash);
        let dhash_fp = fp.get(HashAlgorithm::DHash);

        assert_eq!(phash_fp.global_hash(), fp.phash().global_hash());
        assert_eq!(dhash_fp.global_hash(), fp.dhash().global_hash());
    }
}
