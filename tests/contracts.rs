//! Public-API contracts: EXIF orientation across formats, the versioned byte
//! codec, threshold semantics, `Similarity` ordering, bounded file reads, and
//! the path batch API.

use image::metadata::Orientation;
use image::{DynamicImage, ImageEncoder, RgbImage};
use imgfprint::{
    decode_image, FingerprinterContext, HashAlgorithm, ImageFingerprint, ImageFingerprinter,
    ImgFprintError, MultiHashFingerprint, PreprocessConfig, Similarity, FORMAT_VERSION,
};

/// Asymmetric, non-square test image: every flip/rotation is distinguishable.
fn base_image() -> RgbImage {
    RgbImage::from_fn(96, 64, |x, y| {
        let marker = if x < 24 && y < 16 { 255 } else { 0 };
        image::Rgb([(x * 2) as u8, (y * 3) as u8, marker])
    })
}

/// Little-endian TIFF payload whose IFD0 holds a single Orientation entry.
fn exif_with_orientation(value: u16) -> Vec<u8> {
    let mut v = vec![b'I', b'I', 42, 0, 8, 0, 0, 0]; // header; IFD0 at offset 8
    v.extend_from_slice(&1u16.to_le_bytes()); // one entry
    v.extend_from_slice(&0x0112u16.to_le_bytes()); // Orientation
    v.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    v.extend_from_slice(&1u32.to_le_bytes()); // count
    v.extend_from_slice(&value.to_le_bytes());
    v.extend_from_slice(&[0, 0]); // value padding
    v.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
    v
}

fn encode_png(img: &RgbImage, exif: Option<Vec<u8>>) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::png::PngEncoder::new(&mut buf);
    if let Some(exif) = exif {
        enc.set_exif_metadata(exif).unwrap();
    }
    enc.write_image(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    buf
}

fn encode_jpeg(img: &RgbImage, exif: Option<Vec<u8>>) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
    if let Some(exif) = exif {
        enc.set_exif_metadata(exif).unwrap();
    }
    enc.write_image(
        img.as_raw(),
        img.width(),
        img.height(),
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    buf
}

fn oriented(img: &DynamicImage, exif_value: u8) -> DynamicImage {
    let mut out = img.clone();
    out.apply_orientation(Orientation::from_exif(exif_value).unwrap());
    out
}

fn png_bytes(size: u32, seed: u32) -> Vec<u8> {
    let img = RgbImage::from_fn(size, size, |x, y| {
        image::Rgb([
            (x.wrapping_mul(seed + 3) % 256) as u8,
            (y.wrapping_mul(seed + 5) % 256) as u8,
            ((x ^ y) % 256) as u8,
        ])
    });
    encode_png(&img, None)
}

// ---------------------------------------------------------------------------
// EXIF orientation
// ---------------------------------------------------------------------------

#[test]
fn png_exif_orientation_is_applied_for_all_eight_values() {
    let base = base_image();
    let base_dyn = DynamicImage::ImageRgb8(base.clone());
    for value in 1..=8u8 {
        let decoded = decode_image(&encode_png(
            &base,
            Some(exif_with_orientation(value.into())),
        ))
        .unwrap();
        assert_eq!(
            decoded.to_rgb8(),
            oriented(&base_dyn, value).to_rgb8(),
            "orientation {value}"
        );
    }
}

#[test]
fn jpeg_exif_orientation_is_applied_for_all_eight_values() {
    let base = base_image();
    // The EXIF segment does not change the entropy-coded data, so the plain
    // JPEG decodes to the exact pixels the oriented one must be built from.
    let plain = decode_image(&encode_jpeg(&base, None)).unwrap();
    for value in 1..=8u8 {
        let decoded = decode_image(&encode_jpeg(
            &base,
            Some(exif_with_orientation(value.into())),
        ))
        .unwrap();
        assert_eq!(
            decoded.to_rgb8(),
            oriented(&plain, value).to_rgb8(),
            "orientation {value}"
        );
    }
}

#[test]
fn rotated_png_fingerprints_like_its_upright_pixels() {
    let base = base_image();
    for value in [3u8, 6, 8] {
        let fp = ImageFingerprinter::fingerprint(&encode_png(
            &base,
            Some(exif_with_orientation(value.into())),
        ))
        .unwrap();
        let upright = ImageFingerprinter::fingerprint_image(&oriented(
            &DynamicImage::ImageRgb8(base.clone()),
            value,
        ))
        .unwrap();
        // Exact hashes differ by design (file bytes vs pixels); every
        // perceptual hash must match.
        for alg in [
            HashAlgorithm::AHash,
            HashAlgorithm::PHash,
            HashAlgorithm::DHash,
        ] {
            assert_eq!(
                fp.get(alg).global_hash(),
                upright.get(alg).global_hash(),
                "{alg} {value}"
            );
            assert_eq!(
                fp.get(alg).block_hashes(),
                upright.get(alg).block_hashes(),
                "{alg} {value}"
            );
        }
    }
}

#[test]
fn malformed_exif_never_fails_decoding() {
    let base = base_image();
    for exif in [
        vec![],
        b"II*\0".to_vec(),
        b"MM\0*\xff\xff\xff\xff".to_vec(),
        exif_with_orientation(0),
        exif_with_orientation(9),
        exif_with_orientation(u16::MAX),
    ] {
        let exif = (!exif.is_empty()).then_some(exif);
        let decoded = decode_image(&encode_png(&base, exif.clone())).unwrap();
        assert_eq!(decoded.to_rgb8(), base, "exif {exif:?}");
    }
}

// ---------------------------------------------------------------------------
// Versioned byte codec
// ---------------------------------------------------------------------------

#[test]
fn codec_round_trips_and_is_little_endian() {
    let png = png_bytes(80, 1);
    let multi = ImageFingerprinter::fingerprint(&png).unwrap();
    let single = ImageFingerprinter::fingerprint_with(&png, HashAlgorithm::DHash).unwrap();

    let mb = multi.to_bytes();
    assert_eq!(mb.len(), MultiHashFingerprint::ENCODED_LEN);
    assert_eq!(MultiHashFingerprint::from_bytes(&mb).unwrap(), multi);

    let sb = single.to_bytes();
    assert_eq!(sb.len(), ImageFingerprint::ENCODED_LEN);
    assert_eq!(ImageFingerprint::from_bytes(&sb).unwrap(), single);

    // Header and endianness are part of the format, not of the host.
    assert_eq!(&sb[..2], b"IF");
    assert_eq!(u32::from(sb[2]), FORMAT_VERSION);
    assert_eq!(&sb[4..36], single.exact_hash());
    assert_eq!(sb[36..44], single.global_hash().to_le_bytes());
    assert_eq!(sb[44..52], single.block_hashes()[0].to_le_bytes());
    assert_eq!(&mb[4..36], multi.exact_hash());
    let layers: Vec<u8> = [multi.ahash(), multi.phash(), multi.dhash()]
        .iter()
        .flat_map(|layer| layer.to_bytes()[4..].to_vec())
        .collect();
    assert_eq!(&mb[36..], &layers[..]);
}

#[test]
fn codec_rejects_malformed_input() {
    let multi = ImageFingerprinter::fingerprint(&png_bytes(64, 2)).unwrap();
    let good = multi.to_bytes();

    let reject = |bytes: &[u8]| {
        assert!(
            matches!(
                MultiHashFingerprint::from_bytes(bytes),
                Err(ImgFprintError::InvalidFingerprint(_))
            ),
            "accepted {} bytes",
            bytes.len()
        );
    };
    reject(&[]);
    reject(&good[..good.len() - 1]);
    reject(&[good.as_slice(), &[0]].concat());

    let mut bad_magic = good;
    bad_magic[0] = b'X';
    reject(&bad_magic);

    let mut bad_version = good;
    bad_version[2] = bad_version[2].wrapping_add(1);
    reject(&bad_version);

    let mut bad_kind = good;
    bad_kind[3] = 1;
    reject(&bad_kind);

    // A single-algorithm encoding is never mistaken for a multi one.
    let single = multi.phash().to_bytes();
    reject(&single);
    assert!(ImageFingerprint::from_bytes(&good).is_err());
}

// ---------------------------------------------------------------------------
// Threshold semantics and ordering
// ---------------------------------------------------------------------------

#[test]
fn out_of_range_thresholds_are_never_similar() {
    let a = ImageFingerprinter::fingerprint(&png_bytes(64, 3)).unwrap();
    let b = ImageFingerprinter::fingerprint(&png_bytes(64, 4)).unwrap();
    for t in [
        -5.0,
        -0.0001,
        1.0001,
        7.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert!(!a.is_similar(&a, t), "multi self at {t}");
        assert!(!a.is_similar(&b, t), "multi pair at {t}");
        assert!(!a.phash().is_similar(a.phash(), t), "single at {t}");
    }
    assert!(a.is_similar(&a, 1.0));
    assert!(a.is_similar(&b, 0.0));
    assert!(a.phash().is_similar(a.phash(), 1.0));
}

#[test]
fn single_algorithm_compare_matches_score_semantics() {
    let a = ImageFingerprinter::fingerprint_with(&png_bytes(64, 5), HashAlgorithm::PHash).unwrap();
    let b = ImageFingerprinter::fingerprint_with(&png_bytes(64, 6), HashAlgorithm::PHash).unwrap();
    assert_eq!(a.compare(&a), Similarity::perfect());
    let s = a.compare(&b);
    assert!((0.0..=1.0).contains(&s.score));
    assert_eq!(s.perceptual_distance, a.distance(&b));
    assert!(a.is_similar(&b, s.score));
}

#[test]
fn similarity_ordering_agrees_with_equality() {
    use std::cmp::Ordering;
    let s = |score, exact_match, perceptual_distance| Similarity {
        score,
        exact_match,
        perceptual_distance,
    };
    let cases = [
        s(0.9, false, 5),
        s(0.9, false, 7),
        s(0.9, true, 5),
        s(1.0, true, 0),
        s(0.2, false, 30),
    ];
    for a in &cases {
        for b in &cases {
            assert_eq!(
                a == b,
                a.partial_cmp(b) == Some(Ordering::Equal),
                "{a:?} vs {b:?}"
            );
        }
    }
    // More similar compares greater.
    assert!(s(0.9, false, 5) > s(0.9, false, 7));
    assert!(s(0.9, true, 5) > s(0.9, false, 5));
    assert!(s(0.91, false, 40) > s(0.9, true, 0));
    assert_eq!(s(f32::NAN, false, 0).partial_cmp(&s(0.5, false, 0)), None);
}

// ---------------------------------------------------------------------------
// File input
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn unbounded_special_files_are_rejected_without_exhausting_memory() {
    let mut ctx = FingerprinterContext::with_config(PreprocessConfig {
        max_input_bytes: 1 << 20,
        ..PreprocessConfig::default()
    });
    // /dev/zero reports a size of 0 but never ends.
    let err = ctx.fingerprint_path("/dev/zero").unwrap_err();
    assert!(matches!(err, ImgFprintError::IoError(_)), "{err:?}");
}

#[test]
fn context_config_applies_to_every_entry_point() {
    let tight = PreprocessConfig {
        max_dimension: 48,
        ..PreprocessConfig::default()
    };
    let mut ctx = FingerprinterContext::with_config(tight);
    assert_eq!(ctx.config(), &tight);

    let big = png_bytes(64, 7);
    let path = std::env::temp_dir().join("imgfprint_contracts_config.png");
    std::fs::write(&path, &big).unwrap();
    let decoded = image::load_from_memory(&big).unwrap();

    assert!(matches!(
        ctx.fingerprint(&big),
        Err(ImgFprintError::InvalidImage(_))
    ));
    assert!(matches!(
        ctx.fingerprint_with(&big, HashAlgorithm::AHash),
        Err(ImgFprintError::InvalidImage(_))
    ));
    assert!(matches!(
        ctx.fingerprint_path(&path),
        Err(ImgFprintError::InvalidImage(_))
    ));
    assert!(matches!(
        ctx.fingerprint_path_with(&path, HashAlgorithm::DHash),
        Err(ImgFprintError::InvalidImage(_))
    ));
    assert!(matches!(
        ctx.fingerprint_image(&decoded),
        Err(ImgFprintError::InvalidImage(_))
    ));

    ctx.set_config(PreprocessConfig::default());
    assert_eq!(
        ctx.fingerprint_path(&path).unwrap(),
        ImageFingerprinter::fingerprint(&big).unwrap()
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn fingerprint_paths_preserves_order_and_reports_per_path_errors() {
    let dir = std::env::temp_dir();
    let mut paths = Vec::new();
    for i in 0..9u32 {
        let p = dir.join(format!("imgfprint_contracts_paths_{i}.png"));
        std::fs::write(&p, png_bytes(48 + i * 7, i)).unwrap();
        paths.push(p);
    }
    paths.insert(4, dir.join("imgfprint_contracts_paths_missing.png"));

    let results = ImageFingerprinter::fingerprint_paths(paths.clone());
    assert_eq!(results.len(), paths.len());
    for ((path, result), expected) in results.iter().zip(&paths) {
        assert_eq!(path, expected);
        match ImageFingerprinter::fingerprint_path(path) {
            Ok(fp) => assert_eq!(result.as_ref().unwrap(), &fp),
            Err(_) => assert!(matches!(result, Err(ImgFprintError::IoError(_)))),
        }
    }
    for p in paths {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn pixel_exact_hash_ignores_container_slack() {
    // `ImageBuffer::from_raw` accepts containers longer than the image; the
    // exact hash must cover only the pixels, exactly like `to_rgb8()`.
    let tight = base_image();
    let mut padded = tight.as_raw().clone();
    padded.extend_from_slice(&[0xAB; 37]);
    let padded = RgbImage::from_raw(tight.width(), tight.height(), padded).unwrap();

    let a = ImageFingerprinter::fingerprint_image(&DynamicImage::ImageRgb8(tight)).unwrap();
    let b = ImageFingerprinter::fingerprint_image(&DynamicImage::ImageRgb8(padded)).unwrap();
    assert_eq!(a, b);
}
