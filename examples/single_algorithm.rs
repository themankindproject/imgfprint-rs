use imgfprint::{HashAlgorithm, ImageFingerprinter, ImgFprintError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <image.jpg>", args[0]);
        std::process::exit(1);
    }

    let image_bytes = std::fs::read(&args[1])?;

    println!("Computing fingerprints with different algorithms...\n");

    let ahash = ImageFingerprinter::fingerprint_with(&image_bytes, HashAlgorithm::AHash)?;
    let phash = ImageFingerprinter::fingerprint_with(&image_bytes, HashAlgorithm::PHash)?;
    let dhash = ImageFingerprinter::fingerprint_with(&image_bytes, HashAlgorithm::DHash)?;

    println!("AHash (fastest):     {:016x}", ahash.global_hash());
    println!("PHash (most robust): {:016x}", phash.global_hash());
    println!("DHash (gradients):   {:016x}", dhash.global_hash());

    println!("\nAll three algorithms produced valid fingerprints.");
    println!("Use fingerprint_with() when you need specific algorithm performance.");

    Ok(())
}
