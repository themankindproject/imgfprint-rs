//! Regression tests for the 2026-08 audit fixes.
//!
//! Each test pins one fixed behavior so it cannot silently regress:
//! - single-algorithm mode is bit-identical to the matching multi-hash layer
//! - `fingerprint_image` enforces dimension guards
//! - `MultiHashConfig` validation/sanitization rejects or repairs bad input
//! - `fingerprint_batch_chunked` preserves input order and matches batch output

use image::{DynamicImage, ImageBuffer, Rgb};
use imgfprint::{
    HashAlgorithm, ImageFingerprinter, ImgFprintError, MultiHashConfig, MultiHashFingerprint,
};

/// Structured test image (gradients + LCG noise) — deliberately NOT a solid
/// color, since solid-color images pass hash tests trivially.
fn make_png(size: u32, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(size, size, |x, y| {
        let r = ((x * 7 + y * 3) % 256) as u8 ^ next();
        let g = ((x * 3 + y * 7 + 128) % 256) as u8 ^ next();
        let b = ((x + y * 5 + 64) % 256) as u8 ^ next();
        Rgb([r, g, b])
    });
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

// ---------------------------------------------------------------------------
// Fix [1]: single-algorithm mode must be bit-identical to multi-hash layers
// ---------------------------------------------------------------------------

#[test]
fn single_mode_matches_multi_mode_all_algorithms() {
    for seed in 0..8u64 {
        let png = make_png(128 + seed as u32 * 17, seed + 1);
        let multi = ImageFingerprinter::fingerprint(&png).unwrap();

        let ahash = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::AHash).unwrap();
        let phash = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::PHash).unwrap();
        let dhash = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::DHash).unwrap();

        assert_eq!(
            ahash.global_hash(),
            multi.ahash().global_hash(),
            "AHash global diverged (seed {seed})"
        );
        assert_eq!(
            ahash.block_hashes(),
            multi.ahash().block_hashes(),
            "AHash blocks diverged (seed {seed})"
        );
        assert_eq!(
            phash.global_hash(),
            multi.phash().global_hash(),
            "PHash global diverged (seed {seed})"
        );
        assert_eq!(
            phash.block_hashes(),
            multi.phash().block_hashes(),
            "PHash blocks diverged (seed {seed})"
        );
        assert_eq!(
            dhash.global_hash(),
            multi.dhash().global_hash(),
            "DHash global diverged (seed {seed})"
        );
        assert_eq!(
            dhash.block_hashes(),
            multi.dhash().block_hashes(),
            "DHash blocks diverged (seed {seed})"
        );
    }
}

#[test]
fn fast_mode_still_produces_valid_fingerprints() {
    let png = make_png(200, 99);
    for alg in [HashAlgorithm::AHash, HashAlgorithm::DHash] {
        let fp = ImageFingerprinter::fingerprint_with_fast(&png, alg).unwrap();
        // Self-comparison must be perfect.
        let sim = ImageFingerprinter::compare(&fp, &fp);
        assert_eq!(sim.score, 1.0);
    }
    // PHash fast path falls back to Lanczos3, so it must match the standard path.
    let std_fp = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::PHash).unwrap();
    let fast_fp = ImageFingerprinter::fingerprint_with_fast(&png, HashAlgorithm::PHash).unwrap();
    assert_eq!(std_fp.global_hash(), fast_fp.global_hash());
    assert_eq!(std_fp.block_hashes(), fast_fp.block_hashes());
}

// ---------------------------------------------------------------------------
// Fix [2]: fingerprint_image enforces dimension guards
// ---------------------------------------------------------------------------

#[test]
fn fingerprint_image_rejects_tiny_images() {
    let tiny: DynamicImage = ImageBuffer::from_fn(4, 4, |_, _| Rgb([10u8, 20, 30])).into();
    let err = ImageFingerprinter::fingerprint_image(&tiny).unwrap_err();
    assert!(
        matches!(err, ImgFprintError::ImageTooSmall(_)),
        "expected ImageTooSmall, got {err:?}"
    );

    let one_px: DynamicImage = ImageBuffer::from_fn(1, 1, |_, _| Rgb([0u8, 0, 0])).into();
    assert!(matches!(
        ImageFingerprinter::fingerprint_image(&one_px),
        Err(ImgFprintError::ImageTooSmall(_))
    ));
}

#[test]
fn fingerprint_image_accepts_minimum_size() {
    let img: DynamicImage =
        ImageBuffer::from_fn(32, 32, |x, y| Rgb([x as u8, y as u8, 128])).into();
    assert!(ImageFingerprinter::fingerprint_image(&img).is_ok());
}

#[test]
fn fingerprint_image_matches_bytes_path_perceptual_hashes() {
    // Same pixels through both entry points must produce identical perceptual
    // layers (exact hash differs by design: file bytes vs pixel buffer).
    let png = make_png(96, 7);
    let from_bytes = ImageFingerprinter::fingerprint(&png).unwrap();
    let decoded = image::load_from_memory(&png).unwrap();
    let from_image = ImageFingerprinter::fingerprint_image(&decoded).unwrap();

    assert_eq!(
        from_bytes.phash().global_hash(),
        from_image.phash().global_hash()
    );
    assert_eq!(
        from_bytes.ahash().global_hash(),
        from_image.ahash().global_hash()
    );
    assert_eq!(
        from_bytes.dhash().global_hash(),
        from_image.dhash().global_hash()
    );
}

// ---------------------------------------------------------------------------
// Fix [3]: MultiHashConfig validation and sanitization
// ---------------------------------------------------------------------------

#[test]
fn multi_hash_config_validate_rejects_nan_negative_and_bad_threshold() {
    let base = MultiHashConfig::default();
    assert!(base.validate().is_ok());

    let nan = MultiHashConfig {
        phash_weight: f32::NAN,
        ..base
    };
    assert!(matches!(
        nan.validate(),
        Err(ImgFprintError::InvalidConfig(_))
    ));

    let negative = MultiHashConfig {
        ahash_weight: -1.0,
        ..base
    };
    assert!(matches!(
        negative.validate(),
        Err(ImgFprintError::InvalidConfig(_))
    ));

    let bad_threshold = MultiHashConfig {
        block_distance_threshold: 500,
        ..base
    };
    assert!(matches!(
        bad_threshold.validate(),
        Err(ImgFprintError::InvalidConfig(_))
    ));

    for inf in [f32::INFINITY, f32::NEG_INFINITY] {
        let infinite = MultiHashConfig {
            phash_weight: inf,
            ..base
        };
        assert!(
            matches!(infinite.validate(), Err(ImgFprintError::InvalidConfig(_))),
            "infinite weight {inf} must be rejected"
        );
        // Sanitized form excludes the algorithm instead of saturating at 1.0.
        assert_eq!(infinite.sanitized().phash_weight, 0.0);
    }
}

#[test]
fn multi_hash_config_sanitized_repairs_bad_values() {
    let bad = MultiHashConfig {
        ahash_weight: f32::NAN,
        phash_weight: -2.0,
        dhash_weight: 0.5,
        block_distance_threshold: 500,
        ..MultiHashConfig::default()
    };
    let fixed = bad.sanitized();
    assert_eq!(fixed.ahash_weight, 0.0);
    assert_eq!(fixed.phash_weight, 0.0);
    assert_eq!(fixed.dhash_weight, 0.5);
    assert_eq!(fixed.block_distance_threshold, 64);
    assert!(fixed.validate().is_ok());
}

#[test]
fn compare_with_config_never_produces_nan_score() {
    let png_a = make_png(96, 1);
    let png_b = make_png(96, 2);
    let fp_a = ImageFingerprinter::fingerprint(&png_a).unwrap();
    let fp_b = ImageFingerprinter::fingerprint(&png_b).unwrap();

    let poisoned = MultiHashConfig {
        phash_weight: f32::NAN,
        ..MultiHashConfig::default()
    };
    let sim = fp_a.compare_with_config(&fp_b, &poisoned);
    assert!(
        !sim.score.is_nan(),
        "NaN weight must not poison the score, got {:?}",
        sim
    );
    assert!(sim.score >= 0.0 && sim.score <= 1.0);
}

// ---------------------------------------------------------------------------
// Fix [7]: batch_chunked preserves order and matches batch output
// ---------------------------------------------------------------------------

#[test]
fn batch_chunked_matches_batch_and_preserves_order() {
    let imgs: Vec<(usize, Vec<u8>)> = (0..10).map(|i| (i, make_png(80, 200 + i as u64))).collect();

    let batch = ImageFingerprinter::fingerprint_batch(&imgs);

    let mut chunked: Vec<(usize, Result<MultiHashFingerprint, ImgFprintError>)> = Vec::new();
    ImageFingerprinter::fingerprint_batch_chunked(&imgs, 3, |id, result| {
        chunked.push((id, result));
    });

    assert_eq!(chunked.len(), batch.len());
    for ((cid, cresult), (bid, bresult)) in chunked.iter().zip(batch.iter()) {
        assert_eq!(cid, bid, "callback order diverged from input order");
        let cfp = cresult.as_ref().unwrap();
        let bfp = bresult.as_ref().unwrap();
        assert_eq!(cfp.exact_hash(), bfp.exact_hash());
        assert_eq!(cfp.phash().global_hash(), bfp.phash().global_hash());
    }
}

// ---------------------------------------------------------------------------
// Fix [8]: coarse_key clamps above 64 in all build modes
// ---------------------------------------------------------------------------

#[test]
fn coarse_key_clamps_above_64() {
    let png = make_png(96, 5);
    let fp = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::PHash).unwrap();
    assert_eq!(fp.coarse_key(64), fp.coarse_key(65));
    assert_eq!(fp.coarse_key(64), fp.coarse_key(u32::MAX));
    assert_eq!(fp.coarse_key(64), fp.global_hash());
    assert_eq!(fp.coarse_key(0), 0);
}
