// Stage-split timing: decode -> resize -> grayscale -> extract -> hash,
// all on the SAME image bytes, so the shares are directly comparable.
// Throwaway measurement tool, not part of the library.
use imgfprint::{HashAlgorithm, ImageFingerprinter};
use std::time::Instant;

fn make_png(w: u32, h: u32) -> Vec<u8> {
    let img = image::ImageBuffer::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

fn make_jpeg(w: u32, h: u32) -> Vec<u8> {
    // Same pixels as make_png, JPEG-encoded (quality 90): isolates the
    // decoder's share of the pipeline from the pixel content.
    let img = image::ImageBuffer::from_fn(w, h, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });
    let mut buf = Vec::new();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
    enc.encode_image(&img).unwrap();
    buf
}

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn time<F: FnMut()>(iters: u32, mut f: F) -> u128 {
    let mut ts = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        std::hint::black_box(t.elapsed().as_micros());
        ts.push(t.elapsed().as_micros());
    }
    median(ts)
}

fn main() {
    for (label, w, h) in [("300x300", 300, 300), ("1024x1024", 1024, 1024)] {
        for (enc, bytes) in [("png", make_png(w, h)), ("jpeg", make_jpeg(w, h))] {
            println!("=== {label} {enc} ({} KiB) ===", bytes.len() / 1024);

            // Full pipeline, multi + each single (steady-state, shared ctx).
            let full = time(20, || {
                std::hint::black_box(ImageFingerprinter::fingerprint(&bytes).unwrap());
            });
            let ahash = time(20, || {
                std::hint::black_box(
                    ImageFingerprinter::fingerprint_with(&bytes, HashAlgorithm::AHash).unwrap(),
                );
            });
            let phash = time(20, || {
                std::hint::black_box(
                    ImageFingerprinter::fingerprint_with(&bytes, HashAlgorithm::PHash).unwrap(),
                );
            });
            let dhash = time(20, || {
                std::hint::black_box(
                    ImageFingerprinter::fingerprint_with(&bytes, HashAlgorithm::DHash).unwrap(),
                );
            });
            println!("full multi : {full:8} us");
            println!("single ah  : {ahash:8} us");
            println!("single ph  : {phash:8} us");
            println!("single dh  : {dhash:8} us");

            // Pure compare cost (no image work): 1M iterations.
            let fp1 = ImageFingerprinter::fingerprint(&bytes).unwrap();
            let fp2 = ImageFingerprinter::fingerprint(&bytes).unwrap();
            let t = Instant::now();
            let n = 1_000_000u32;
            for _ in 0..n {
                std::hint::black_box(fp1.compare(std::hint::black_box(&fp2)));
            }
            println!("compare    : {:8} ns", t.elapsed().as_nanos() / n as u128);

            // Decode-only vs full: isolates decode share.
            let dec = time(50, || {
                std::hint::black_box(image::load_from_memory(&bytes).unwrap());
            });
            println!(
                "decode_only: {dec:8} us ({:.0}%)",
                100.0 * dec as f64 / full as f64
            );
            println!();
        }
    }
}
