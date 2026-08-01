//! Tests demonstrating the divergence between `fingerprint()` (file-byte exact hash)
//! and `fingerprint_image()` (pixel-buffer exact hash).
//!
//! Key invariant: for the same pixel data, perceptual hashes are identical
//! regardless of the API used, but the exact hashes differ because they hash
//! different inputs (compressed bytes vs. raw RGB8 pixels).

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use imgfprint::ImageFingerprinter;
use std::io::Cursor;

/// Helper: create a deterministic 64×64 test image.
fn make_test_image() -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    ImageBuffer::from_fn(64, 64, |x, y| Rgb([(x * 4) as u8, (y * 4) as u8, 128u8]))
}

/// Helper: encode an image to PNG bytes.
fn encode_png(img: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<u8> {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)
        .expect("PNG encoding should not fail");
    buf
}

/// `fingerprint_image()` produces the same exact hash for identical pixel data,
/// regardless of how many times it's called.
#[test]
fn fingerprint_image_exact_hash_is_deterministic() {
    let img = make_test_image();
    let dynamic = DynamicImage::ImageRgb8(img);

    let fp1 = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();
    let fp2 = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();

    assert_eq!(
        fp1.exact_hash(),
        fp2.exact_hash(),
        "fingerprint_image() must produce the same exact hash for the same pixels"
    );
}

/// `fingerprint()` hashes raw file bytes, so its exact hash differs from
/// `fingerprint_image()` which hashes decoded RGB8 pixels.
#[test]
fn fingerprint_vs_fingerprint_image_exact_hashes_differ() {
    let img = make_test_image();
    let dynamic = DynamicImage::ImageRgb8(img.clone());
    let png_bytes = encode_png(&img);

    let fp_bytes = ImageFingerprinter::fingerprint(&png_bytes).unwrap();
    let fp_image = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();

    // Exact hashes MUST differ — one hashes PNG bytes, the other hashes raw pixels
    assert_ne!(
        fp_bytes.exact_hash(),
        fp_image.exact_hash(),
        "fingerprint() and fingerprint_image() must produce different exact hashes \
         because they hash different data (file bytes vs. RGB8 pixels)"
    );
}

/// Despite different exact hashes, perceptual hashes are identical across both APIs
/// because they both normalize to the same pixel representation before hashing.
#[test]
fn perceptual_hashes_match_across_both_apis() {
    let img = make_test_image();
    let dynamic = DynamicImage::ImageRgb8(img.clone());
    let png_bytes = encode_png(&img);

    let fp_bytes = ImageFingerprinter::fingerprint(&png_bytes).unwrap();
    let fp_image = ImageFingerprinter::fingerprint_image(&dynamic).unwrap();

    assert_eq!(
        fp_bytes.phash().global_hash(),
        fp_image.phash().global_hash(),
        "PHash global hash should be the same regardless of API"
    );
    assert_eq!(
        fp_bytes.dhash().global_hash(),
        fp_image.dhash().global_hash(),
        "DHash global hash should be the same regardless of API"
    );
    assert_eq!(
        fp_bytes.ahash().global_hash(),
        fp_image.ahash().global_hash(),
        "AHash global hash should be the same regardless of API"
    );
}

/// Encoding the same pixels to PNG twice produces the same file bytes
/// (PNG encoding in the `image` crate is deterministic), so `fingerprint()`
/// exact hashes match for two identical PNG encodings of the same image.
///
/// However, if we modify the raw bytes slightly (simulating a non-deterministic
/// encoder or different metadata), the exact hashes will diverge while
/// `fingerprint_image()` on the decoded pixels still matches.
#[test]
fn different_file_bytes_same_pixels_diverge_on_fingerprint() {
    let img = make_test_image();

    // Encode to PNG twice — the `image` crate's PNG encoder is deterministic,
    // so we manually perturb one copy to simulate different file bytes
    // (e.g., different encoder, timestamps, or compression level).
    let png_bytes_original = encode_png(&img);
    let mut png_bytes_modified = png_bytes_original.clone();

    // Append trailing garbage after the IEND chunk.
    // PNG parsers ignore data after IEND, so the decoded pixels are identical.
    png_bytes_modified.extend_from_slice(b"\x00\x00\x00\x00GARBAGE_AFTER_IEND");

    // Verify both decode to the same pixels
    let decoded_original =
        image::load_from_memory_with_format(&png_bytes_original, ImageFormat::Png).unwrap();
    let decoded_modified =
        image::load_from_memory_with_format(&png_bytes_modified, ImageFormat::Png).unwrap();
    assert_eq!(
        decoded_original.to_rgb8().as_raw(),
        decoded_modified.to_rgb8().as_raw(),
        "Both PNG byte sequences should decode to identical pixels"
    );

    // fingerprint() hashes the raw bytes — different bytes → different exact hash
    let fp_original = ImageFingerprinter::fingerprint(&png_bytes_original).unwrap();
    let fp_modified = ImageFingerprinter::fingerprint(&png_bytes_modified).unwrap();

    assert_ne!(
        fp_original.exact_hash(),
        fp_modified.exact_hash(),
        "fingerprint() exact hashes must differ when file bytes differ"
    );

    // fingerprint_image() hashes the decoded pixels — same pixels → same exact hash
    let fp_img_original = ImageFingerprinter::fingerprint_image(&decoded_original).unwrap();
    let fp_img_modified = ImageFingerprinter::fingerprint_image(&decoded_modified).unwrap();

    assert_eq!(
        fp_img_original.exact_hash(),
        fp_img_modified.exact_hash(),
        "fingerprint_image() exact hashes must match when pixels are identical"
    );

    // Perceptual hashes match in all cases (same pixel data after decode)
    assert_eq!(
        fp_original.phash().global_hash(),
        fp_modified.phash().global_hash(),
        "Perceptual hashes should match despite different file bytes"
    );
}
