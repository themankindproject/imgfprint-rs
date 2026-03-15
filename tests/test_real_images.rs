use imgfprint::{FingerprinterContext, HashAlgorithm, ImageFingerprinter};
use std::io::Cursor;

fn encode_png(img: &image::RgbImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .unwrap();
    bytes
}

#[test]
fn test_png_fingerprint() {
    let img = image::RgbImage::new(256, 256);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_determinism() {
    let img = image::RgbImage::new(256, 256);
    let bytes = encode_png(&img);

    let fp1 = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    let fp2 = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");

    assert_eq!(fp1.exact_hash(), fp2.exact_hash());
    let sim = fp1.compare(&fp2);
    assert!(sim.exact_match);
    assert!((sim.score - 1.0).abs() < 1e-6);
}

#[test]
fn test_different_images_different_hashes() {
    let mut img1 = image::RgbImage::new(256, 256);
    let mut img2 = image::RgbImage::new(256, 256);

    for pixel in img1.pixels_mut() {
        pixel.0 = [100, 100, 100];
    }
    for pixel in img2.pixels_mut() {
        pixel.0 = [200, 200, 200];
    }

    let bytes1 = encode_png(&img1);
    let bytes2 = encode_png(&img2);

    let fp1 = ImageFingerprinter::fingerprint(&bytes1).expect("fingerprint should succeed");
    let fp2 = ImageFingerprinter::fingerprint(&bytes2).expect("fingerprint should succeed");

    assert_ne!(fp1.exact_hash(), fp2.exact_hash());
}

#[test]
fn test_single_algorithm_mode() {
    let img = image::RgbImage::new(256, 256);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint_with(&bytes, HashAlgorithm::PHash)
        .expect("fingerprint should succeed");

    assert!(fp.global_hash() != 0);
}

#[test]
fn test_context_api() {
    let mut ctx = FingerprinterContext::new();

    let img = image::RgbImage::new(256, 256);
    let bytes = encode_png(&img);

    let fp = ctx.fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));

    let fp2 = ctx
        .fingerprint(&bytes)
        .expect("second fingerprint should succeed");
    assert_eq!(fp.exact_hash(), fp2.exact_hash());
}

#[test]
fn test_similar_images_high_similarity() {
    let mut img1 = image::RgbImage::new(256, 256);
    let mut img2 = image::RgbImage::new(256, 256);

    for (p1, p2) in img1.pixels_mut().zip(img2.pixels_mut()) {
        p1.0 = [100, 100, 100];
        p2.0 = [100, 100, 101];
    }

    let bytes1 = encode_png(&img1);
    let bytes2 = encode_png(&img2);

    let fp1 = ImageFingerprinter::fingerprint(&bytes1).expect("fingerprint should succeed");
    let fp2 = ImageFingerprinter::fingerprint(&bytes2).expect("fingerprint should succeed");

    let sim = fp1.compare(&fp2);
    assert!(
        sim.score > 0.9,
        "Similar images should have high similarity"
    );
}

#[test]
fn test_multihash_fingerprint_accessors() {
    let img = image::RgbImage::new(256, 256);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");

    let _ahash = fp.ahash();
    let _phash = fp.phash();
    let _dhash = fp.dhash();

    let _by_algorithm = fp.get(HashAlgorithm::PHash);
}

#[test]
fn test_is_similar_method() {
    let mut img1 = image::RgbImage::new(256, 256);
    let mut img2 = image::RgbImage::new(256, 256);

    for (p1, p2) in img1.pixels_mut().zip(img2.pixels_mut()) {
        p1.0 = [100, 100, 100];
        p2.0 = [100, 100, 100];
    }

    let bytes1 = encode_png(&img1);
    let bytes2 = encode_png(&img2);

    let fp1 = ImageFingerprinter::fingerprint(&bytes1).expect("fingerprint should succeed");
    let fp2 = ImageFingerprinter::fingerprint(&bytes2).expect("fingerprint should succeed");

    assert!(fp1.is_similar(&fp2, 0.8));
}
