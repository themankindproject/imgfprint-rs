#![no_main]

use imgfprint::ImageFingerprinter;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test multi-algorithm fingerprinting (default)
    let _ = ImageFingerprinter::fingerprint(data);
});
