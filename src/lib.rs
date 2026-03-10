//! High-performance, deterministic image fingerprinting library.
//!
//! Provides perceptual hashing (pHash) with crop resistance for image
//! deduplication and similarity detection in production systems.
//!
//! ## Example
//!
//! ```rust
//! use imgfprint_rs::ImageFingerprinter;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let fp1 = ImageFingerprinter::fingerprint(&std::fs::read("img1.jpg")?)?;
//! let fp2 = ImageFingerprinter::fingerprint(&std::fs::read("img2.jpg")?)?;
//!
//! let sim = ImageFingerprinter::compare(&fp1, &fp2);
//! println!("Similarity: {:.2}", sim.score);
//! # Ok(())
//! # }
//! ```

mod core;
mod error;
mod hash;
mod imgproc;

pub use core::fingerprint::ImageFingerprint;
pub use core::fingerprinter::ImageFingerprinter;
pub use core::similarity::Similarity;
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
    fn test_deterministic_fingerprints() {
        let img = create_test_image();
        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();

        assert_eq!(fp1.exact_hash(), fp2.exact_hash());
        assert_eq!(fp1.global_phash(), fp2.global_phash());
        assert_eq!(fp1.block_hashes(), fp2.block_hashes());
    }

    #[test]
    fn test_identical_images() {
        let img = create_test_image();
        let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();
        let sim = ImageFingerprinter::compare(&fp1, &fp2);

        assert!(sim.exact_match);
        assert_eq!(sim.score, 1.0);
        assert_eq!(sim.perceptual_distance, 0);
    }

    #[test]
    fn test_similar_images() {
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
        let sim = ImageFingerprinter::compare(&fp1, &fp2);

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
        let sim = ImageFingerprinter::compare(&fp1, &fp2);

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
        let sim = ImageFingerprinter::compare(&fp1, &fp2);

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
        assert_eq!(fp.block_hashes().len(), 16);
    }
}
