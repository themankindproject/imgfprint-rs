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

#[test]
fn test_non_square_image_landscape() {
    let img = image::RgbImage::new(1920, 1080);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_non_square_image_portrait() {
    let img = image::RgbImage::new(1080, 1920);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_minimum_boundary_32x32() {
    let img = image::RgbImage::new(32, 32);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_maximum_boundary_8192x8192() {
    let img = image::RgbImage::new(8192, 8192);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_grayscale_image() {
    let img = image::GrayImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_rgba_image() {
    let img = image::RgbaImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_rgb_image() {
    let img = image::RgbImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_jpeg_format() {
    let img = image::RgbImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Jpeg,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_gif_format() {
    let img = image::RgbImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Gif,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_webp_format() {
    let img = image::RgbImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::WebP,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_bmp_format() {
    let img = image::RgbImage::new(256, 256);
    let mut bytes = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Bmp,
    )
    .unwrap();

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_small_image_32x32_minimum() {
    let img = image::RgbImage::new(32, 32);
    let bytes = encode_png(&img);

    let fp = ImageFingerprinter::fingerprint(&bytes).expect("fingerprint should succeed");
    assert!(!fp.exact_hash().iter().all(|&b| b == 0));
}

#[test]
fn test_small_image_too_small_error() {
    let img = image::RgbImage::new(7, 7);
    let bytes = encode_png(&img);

    let result = ImageFingerprinter::fingerprint(&bytes);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("too small"));
}
