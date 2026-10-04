//! Optional input formats (`tiff`, `ico`, `pnm`, `qoi` features).
//!
//! Every format is lossless here, so each must fingerprint exactly like the
//! same pixels encoded as PNG (exact hashes differ: they cover file bytes).
//!
//! TGA is deliberately not covered: it has no magic bytes, so
//! `with_guessed_format` can never identify it from a byte buffer and no
//! entry point in this crate could reach the decoder.

#![cfg(any(feature = "tiff", feature = "ico", feature = "pnm", feature = "qoi"))]

use image::{DynamicImage, ImageFormat, RgbaImage};
use imgfprint::{HashAlgorithm, ImageFingerprinter};

fn sample() -> DynamicImage {
    DynamicImage::ImageRgba8(RgbaImage::from_fn(64, 48, |x, y| {
        image::Rgba([(x * 4) as u8, (y * 5) as u8, ((x ^ y) * 3) as u8, 255])
    }))
}

fn encode(img: &DynamicImage, format: ImageFormat) -> Vec<u8> {
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), format)
        .unwrap();
    buf
}

fn assert_matches_png(format: ImageFormat) {
    let img = sample();
    let reference = ImageFingerprinter::fingerprint(&encode(&img, ImageFormat::Png)).unwrap();
    let fp = ImageFingerprinter::fingerprint(&encode(&img, format))
        .unwrap_or_else(|e| panic!("{format:?}: {e}"));
    for alg in [
        HashAlgorithm::AHash,
        HashAlgorithm::PHash,
        HashAlgorithm::DHash,
    ] {
        assert_eq!(
            fp.get(alg).global_hash(),
            reference.get(alg).global_hash(),
            "{format:?} {alg}"
        );
        assert_eq!(
            fp.get(alg).block_hashes(),
            reference.get(alg).block_hashes(),
            "{format:?} {alg}"
        );
    }
}

#[cfg(feature = "tiff")]
#[test]
fn tiff_decodes() {
    assert_matches_png(ImageFormat::Tiff);
}

#[cfg(feature = "ico")]
#[test]
fn ico_decodes() {
    assert_matches_png(ImageFormat::Ico);
}

#[cfg(feature = "pnm")]
#[test]
fn pnm_decodes() {
    // PNM has no alpha channel; compare against an opaque RGB reference.
    let rgb = DynamicImage::ImageRgb8(sample().to_rgb8());
    let reference = ImageFingerprinter::fingerprint(&encode(&rgb, ImageFormat::Png)).unwrap();
    let fp = ImageFingerprinter::fingerprint(&encode(&rgb, ImageFormat::Pnm)).unwrap();
    assert_eq!(fp.phash().block_hashes(), reference.phash().block_hashes());
    assert_eq!(fp.dhash().global_hash(), reference.dhash().global_hash());
}

#[cfg(feature = "qoi")]
#[test]
fn qoi_decodes() {
    assert_matches_png(ImageFormat::Qoi);
}
