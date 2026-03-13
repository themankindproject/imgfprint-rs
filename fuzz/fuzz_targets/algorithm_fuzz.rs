#![no_main]

use imgfprint::{HashAlgorithm, ImageFingerprinter};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test DHash specifically
    let _ = ImageFingerprinter::fingerprint_with(data, HashAlgorithm::DHash);

    // Test PHash specifically
    let _ = ImageFingerprinter::fingerprint_with(data, HashAlgorithm::PHash);
});
