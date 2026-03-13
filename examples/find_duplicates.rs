use imgfprint::{HashAlgorithm, ImageFingerprinter};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image_dir>", args[0]);
        std::process::exit(1);
    }

    let dir = &args[1];
    let entries = std::fs::read_dir(dir)?;

    let mut fingerprints: HashMap<String, Vec<(String, u64)>> = HashMap::new();

    println!("Fingerprinting images...");

    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if !["jpg", "jpeg", "png", "gif", "webp", "bmp"].contains(&ext.as_str()) {
                continue;
            }

            let filename = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            match std::fs::read(&path) {
                Ok(bytes) => {
                    match ImageFingerprinter::fingerprint_with(&bytes, HashAlgorithm::PHash) {
                        Ok(fp) => {
                            let hash = fp.global_phash();
                            fingerprints
                                .entry(format!("{:016x}", hash))
                                .or_default()
                                .push((filename, hash));
                        }
                        Err(e) => eprintln!("Error fingerprinting {}: {}", filename, e),
                    }
                }
                Err(e) => eprintln!("Error reading {}: {}", filename, e),
            }
        }
    }

    println!("\nFound {} unique fingerprints\n", fingerprints.len());

    for (hash, files) in fingerprints {
        if files.len() > 1 {
            println!("Duplicate group (hash: {}):", &hash[..8]);
            for (filename, _) in &files {
                println!("  - {}", filename);
            }
            println!();
        }
    }

    Ok(())
}
