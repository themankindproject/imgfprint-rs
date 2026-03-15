#![allow(dead_code)]

use imgfprint::ImageFingerprinter;
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_compare_reflexivity(ref img_bytes in any_image()) {
        let fp = ImageFingerprinter::fingerprint(img_bytes).unwrap();
        let sim = fp.compare(&fp);
        prop_assert!((sim.score - 1.0).abs() < 1e-6, "Similarity with self should be 1.0");
        prop_assert!(sim.exact_match, "Exact match should be true for same fingerprint");
    }

    #[test]
    fn prop_compare_symmetry(ref img1 in any_image(), ref img2 in any_image()) {
        let fp1 = ImageFingerprinter::fingerprint(img1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(img2).unwrap();
        let sim1 = fp1.compare(&fp2);
        let sim2 = fp2.compare(&fp1);
        prop_assert!((sim1.score - sim2.score).abs() < 1e-6, "Comparison should be symmetric");
    }

    #[test]
    fn prop_score_in_range(ref img1 in any_image(), ref img2 in any_image()) {
        let fp1 = ImageFingerprinter::fingerprint(img1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(img2).unwrap();
        let sim = fp1.compare(&fp2);
        prop_assert!((0.0..=1.0).contains(&sim.score), "Score should be in [0.0, 1.0]");
    }

    #[test]
    fn prop_distance_in_range(ref img1 in any_image(), ref img2 in any_image()) {
        let fp1 = ImageFingerprinter::fingerprint(img1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(img2).unwrap();
        let dist = fp1.phash().distance(fp2.phash());
        prop_assert!((0..=64).contains(&dist), "Hamming distance should be in [0, 64]");
    }

    #[test]
    fn prop_is_similar_with_high_threshold(ref img in any_image()) {
        let fp = ImageFingerprinter::fingerprint(img).unwrap();
        prop_assert!(fp.is_similar(&fp, 0.8), "Image should be similar to itself");
    }

    #[test]
    fn prop_different_images_can_be_similar(ref img1 in any_image(), ref img2 in any_image()) {
        let fp1 = ImageFingerprinter::fingerprint(img1).unwrap();
        let fp2 = ImageFingerprinter::fingerprint(img2).unwrap();

        let sim = fp1.compare(&fp2);

        if fp1.exact_hash() == fp2.exact_hash() {
            prop_assert!((sim.score - 1.0).abs() < 1e-6);
        }
    }
}

fn any_image() -> impl Strategy<Value = Vec<u8>> {
    (32u32..=512u32, 32u32..=512u32, 1u8..=3u8).prop_map(|(width, height, color_channels)| {
        let img = match color_channels {
            1 => image::DynamicImage::ImageLuma8(image::GrayImage::new(width, height)),
            2 => image::DynamicImage::ImageLumaA8(image::GrayAlphaImage::new(width, height)),
            _ => image::DynamicImage::ImageRgb8(image::RgbImage::new(width, height)),
        };

        let mut bytes = Vec::new();
        use std::io::Cursor;
        img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        bytes
    })
}
