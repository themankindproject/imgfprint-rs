use imgfprint::{ImageFingerprinter, MultiHashFingerprint, FORMAT_VERSION};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("Usage: zero_copy_persistence <image> [image...]");
        std::process::exit(1);
    }

    let mut fingerprints = Vec::with_capacity(paths.len());
    for path in &paths {
        let bytes = std::fs::read(path)?;
        fingerprints.push(ImageFingerprinter::fingerprint(&bytes)?);
    }

    // Persist FORMAT_VERSION in your index manifest/header, then persist the
    // fingerprint slice as bytes. This is the zero-copy view UCFP's mmap index
    // can write directly.
    let stored_format_version = FORMAT_VERSION;
    let persisted: &[u8] = bytemuck::cast_slice(&fingerprints);

    println!(
        "prepared {} fingerprints as {} bytes with format version {}",
        fingerprints.len(),
        persisted.len(),
        stored_format_version
    );

    // On read, reject incompatible index files before casting bytes back.
    if stored_format_version != FORMAT_VERSION {
        return Err(format!(
            "fingerprint format mismatch: stored {}, crate {}",
            stored_format_version, FORMAT_VERSION
        )
        .into());
    }

    let restored: &[MultiHashFingerprint] = bytemuck::cast_slice(persisted);
    assert_eq!(restored, fingerprints.as_slice());

    println!(
        "round-tripped {} fingerprints without serde",
        restored.len()
    );
    Ok(())
}
