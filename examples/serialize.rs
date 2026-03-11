use imgfprint::ImageFingerprinter;
use std::fs::File;
use std::io::{BufReader, BufWriter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image.jpg>", args[0]);
        std::process::exit(1);
    }

    let img_bytes = std::fs::read(&args[1])?;
    let fp = ImageFingerprinter::fingerprint(&img_bytes)?;

    // JSON serialization
    let json = serde_json::to_string_pretty(&fp)?;
    println!("JSON:\n{}\n", json);

    // Binary serialization
    let bin_path = "fingerprint.bin";
    let file = File::create(bin_path)?;
    let mut writer = BufWriter::new(file);
    bincode::serialize_into(&mut writer, &fp)?;
    println!("Binary: {} bytes\n", std::fs::metadata(bin_path)?.len());

    // Deserialize from JSON
    let fp_json: imgfprint::ImageFingerprint = serde_json::from_str(&json)?;
    println!("From JSON - global_phash: {:016x}", fp_json.global_phash());

    // Deserialize from binary
    let file = File::open(bin_path)?;
    let mut reader = BufReader::new(file);
    let fp_bin: imgfprint::ImageFingerprint = bincode::deserialize_from(&mut reader)?;
    println!("From binary - global_phash: {:016x}", fp_bin.global_phash());

    // Clean up
    std::fs::remove_file(bin_path)?;

    Ok(())
}
