#![no_main]

use imgfprint::ImageFingerprinter;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(fp1) = ImageFingerprinter::fingerprint(data) {
        let _ = ImageFingerprinter::compare(&fp1, &fp1);
        let _ = fp1.is_similar(&fp1, 0.8);
        let _ = fp1.distance(&fp1);
    }
});
