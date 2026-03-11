#![no_main]

use imgfprint_rs::ImageFingerprinter;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ImageFingerprinter::fingerprint(data);
});
