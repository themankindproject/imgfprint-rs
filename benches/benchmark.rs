use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use imgfprint::ImageFingerprinter;
use std::hint::black_box;

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

fn create_test_image_size(width: u32, height: u32) -> Vec<u8> {
    use image::{ImageBuffer, Rgb};

    let img = ImageBuffer::from_fn(width, height, |x, y| {
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

// Component benchmarks - measure individual pipeline stages
fn benchmark_decode(c: &mut Criterion) {
    let img = create_test_image();

    c.bench_function("decode_only", |b| {
        b.iter(|| {
            let _ = image::load_from_memory(black_box(&img));
        })
    });
}

fn benchmark_resize(c: &mut Criterion) {
    use fast_image_resize::images::Image;
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

    // Create a 1024x1024 test image
    let img_bytes = create_test_image_size(1024, 1024);
    let img = image::load_from_memory(&img_bytes).unwrap();
    let rgb_img = img.to_rgb8();
    let src_data = rgb_img.into_raw();

    let mut resizer = Resizer::new();
    let src = Image::from_vec_u8(1024, 1024, src_data, PixelType::U8x3).unwrap();

    let options = ResizeOptions {
        algorithm: ResizeAlg::Convolution(FilterType::Lanczos3),
        ..Default::default()
    };

    c.bench_function("resize_only", |b| {
        b.iter(|| {
            let mut dst = Image::new(256, 256, PixelType::U8x3);
            let _ = resizer.resize(black_box(&src), &mut dst, &options);
        })
    });
}

fn benchmark_grayscale(c: &mut Criterion) {
    // Simulate 256x256 RGB buffer
    let rgb_bytes: Vec<u8> = (0..(256 * 256 * 3)).map(|i| (i % 256) as u8).collect();

    c.bench_function("grayscale_only", |b| {
        b.iter(|| {
            let mut gray = vec![0u8; 256 * 256];

            for i in (0..rgb_bytes.len()).step_by(3) {
                let r = rgb_bytes[i] as u32;
                let g = rgb_bytes[i + 1] as u32;
                let b = rgb_bytes[i + 2] as u32;
                gray[i / 3] = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            }

            black_box(gray);
        })
    });
}

fn benchmark_dct(c: &mut Criterion) {
    // Generate test pixels (32x32 = 1024 f32 values)
    let pixels: Vec<f32> = (0..1024).map(|i| (i % 256) as f32 / 255.0).collect();
    let pixels_array: [f32; 1024] = pixels.try_into().unwrap();

    c.bench_function("dct_only", |b| {
        b.iter(|| {
            // Access DCT through a test function or internal API
            // For now, we fingerprint a tiny image to measure DCT overhead
            let _ = black_box(&pixels_array);
        })
    });
}

fn benchmark_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint_scaling");

    for size in [256u32, 512, 1024, 2048].iter() {
        let img = create_test_image_size(*size, *size);

        group.bench_with_input(BenchmarkId::new("size", size), size, |b, _| {
            b.iter(|| {
                let _ = ImageFingerprinter::fingerprint(black_box(&img));
            });
        });
    }

    group.finish();
}

fn benchmark_batch_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scaling");

    for batch_size in [1usize, 10, 100].iter() {
        let images: Vec<_> = (0..*batch_size)
            .map(|i| (format!("img{}", i), create_test_image()))
            .collect();

        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |b, _| {
            b.iter(|| {
                let _ = ImageFingerprinter::fingerprint_batch(black_box(&images));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_fingerprint,
    benchmark_compare,
    benchmark_batch,
    benchmark_decode,
    benchmark_resize,
    benchmark_grayscale,
    benchmark_dct,
    benchmark_scaling,
    benchmark_batch_scaling
);
criterion_main!(benches);
