#![no_main]

use imgfprint::{ImageFingerprint, MultiHashFingerprint};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test deserializing ImageFingerprint (single algorithm)
    let _ = serde_json::from_slice::<ImageFingerprint>(data);
    let _ = bincode::deserialize::<ImageFingerprint>(data);

    // Test deserializing MultiHashFingerprint (multi algorithm)
    let _ = serde_json::from_slice::<MultiHashFingerprint>(data);
    let _ = bincode::deserialize::<MultiHashFingerprint>(data);
});
