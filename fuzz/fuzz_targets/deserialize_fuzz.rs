#![no_main]

use imgfprint_rs::ImageFingerprint;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<ImageFingerprint>(data);
    let _ = bincode::deserialize::<ImageFingerprint>(data);
});
