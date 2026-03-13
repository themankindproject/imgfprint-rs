#![no_main]

use imgfprint::ImageFingerprinter;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test batch processing with multiple images
    // Split data into chunks to simulate multiple images
    let chunk_size = data.len() / 3 + 1;
    let images: Vec<(String, Vec<u8>)> = data
        .chunks(chunk_size)
        .enumerate()
        .map(|(i, chunk)| (format!("img{}", i), chunk.to_vec()))
        .collect();

    if !images.is_empty() {
        let _ = ImageFingerprinter::fingerprint_batch(&images);
    }
});
