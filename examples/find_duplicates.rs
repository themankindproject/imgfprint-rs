//! Find duplicates in a directory of images.
//!
//! Two passes:
//!   1. Byte-identical groups via `HashMap<MultiHashFingerprint, Vec<PathBuf>>`
//!      (uses the `Hash` derive on `MultiHashFingerprint`).
//!   2. Perceptually-similar pairs via pairwise comparison among the unique
//!      fingerprints from pass 1.
//!
//! Usage:
//!   cargo run --release --example find_duplicates -- <image_dir> [threshold]
//!
//! The optional threshold is a similarity score in [0.0, 1.0] (default 0.95).

use imgfprint::{ImageFingerprinter, MultiHashFingerprint};
use std::collections::HashMap;
use std::path::PathBuf;

const SUPPORTED: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_dir> [threshold]", args[0]);
        std::process::exit(1);
    }

    let dir = &args[1];
    let threshold: f32 = args
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.95)
        .clamp(0.0, 1.0);

    let paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
                return false;
            };
            let lower = ext.to_ascii_lowercase();
            SUPPORTED.contains(&lower.as_str())
        })
        .collect();

    println!("Fingerprinting {} files...", paths.len());

    // Pass 1: byte-identical buckets keyed on the full fingerprint.
    let mut by_fingerprint: HashMap<MultiHashFingerprint, Vec<PathBuf>> = HashMap::new();
    for path in &paths {
        match ImageFingerprinter::fingerprint_path(path) {
            Ok(fp) => by_fingerprint.entry(fp).or_default().push(path.clone()),
            Err(e) => eprintln!("error on {}: {}", path.display(), e),
        }
    }

    println!("\n=== Byte-identical groups ===");
    let mut byte_groups = 0;
    for files in by_fingerprint.values() {
        if files.len() > 1 {
            byte_groups += 1;
            println!("Group of {}:", files.len());
            for p in files {
                println!("  - {}", p.display());
            }
        }
    }
    if byte_groups == 0 {
        println!("(none)");
    }

    // Pass 2: pairwise compare the unique fingerprints (those without an exact dupe).
    let unique: Vec<(&MultiHashFingerprint, &PathBuf)> = by_fingerprint
        .iter()
        .filter(|(_, files)| files.len() == 1)
        .map(|(fp, files)| (fp, &files[0]))
        .collect();

    println!(
        "\n=== Perceptually similar pairs (threshold {:.2}) ===",
        threshold
    );
    let mut pair_count = 0;
    for i in 0..unique.len() {
        for j in (i + 1)..unique.len() {
            let sim = unique[i].0.compare(unique[j].0);
            if sim.score >= threshold {
                pair_count += 1;
                println!(
                    "  {:.3}  {}  <->  {}",
                    sim.score,
                    unique[i].1.display(),
                    unique[j].1.display(),
                );
            }
        }
    }
    if pair_count == 0 {
        println!("(none)");
    }

    Ok(())
}
