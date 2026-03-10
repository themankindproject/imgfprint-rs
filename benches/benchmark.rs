use criterion::{black_box, criterion_group, criterion_main, Criterion};
use imgfprint_rs::ImageFingerprinter;

fn create_test_image() -> Vec<u8> {
    use image::{ImageBuffer, Rgb};

    let img = ImageBuffer::from_fn(300, 300, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    });

    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

fn benchmark_fingerprint(c: &mut Criterion) {
    let img = create_test_image();

    c.bench_function("fingerprint_single", |b| {
        b.iter(|| {
            let _ = ImageFingerprinter::fingerprint(black_box(&img));
        })
    });
}

fn benchmark_compare(c: &mut Criterion) {
    let img = create_test_image();
    let fp1 = ImageFingerprinter::fingerprint(&img).unwrap();
    let fp2 = ImageFingerprinter::fingerprint(&img).unwrap();

    c.bench_function("compare_fingerprints", |b| {
        b.iter(|| {
            let _ = ImageFingerprinter::compare(black_box(&fp1), black_box(&fp2));
        })
    });
}

fn benchmark_batch(c: &mut Criterion) {
    let images: Vec<_> = (0..10)
        .map(|i| (format!("img{}", i), create_test_image()))
        .collect();

    c.bench_function("fingerprint_batch_10", |b| {
        b.iter(|| {
            let _ = ImageFingerprinter::fingerprint_batch(black_box(&images));
        })
    });
}

criterion_group!(
    benches,
    benchmark_fingerprint,
    benchmark_compare,
    benchmark_batch
);
criterion_main!(benches);
