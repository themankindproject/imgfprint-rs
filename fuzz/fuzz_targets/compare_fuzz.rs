#![no_main]

use imgfprint::{HashAlgorithm, ImageFingerprinter};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test multi-algorithm fingerprinting and comparison
    if let Ok(fp1) = ImageFingerprinter::fingerprint(data) {
        // Test MultiHashFingerprint methods
        let _ = fp1.compare(&fp1);
        let _ = fp1.is_similar(&fp1, 0.8);
        let _ = fp1.exact_hash();
        let _ = fp1.phash();
        let _ = fp1.dhash();

        // Test single algorithm mode
        if let Ok(fp_single) = ImageFingerprinter::fingerprint_with(data, HashAlgorithm::PHash) {
            let _ = ImageFingerprinter::compare(&fp_single, &fp_single);
            let _ = fp_single.is_similar(&fp_single, 0.8);
            let _ = fp_single.distance(&fp_single);
        }
    }
});
