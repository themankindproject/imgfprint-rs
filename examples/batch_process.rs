use imgfprint::ImageFingerprinter;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image_dir> [output.json]", args[0]);
        std::process::exit(1);
    }

    let dir = &args[1];
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "fingerprints.json".to_string());

    let paths: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let ext = e
                .path()
                .extension()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            ["jpg", "jpeg", "png", "gif", "webp", "bmp"].contains(&ext.as_str())
        })
        .collect();

    println!("Processing {} images...", paths.len());

    let images: Vec<_> = paths
        .iter()
        .filter_map(|p| {
            let name = p.file_name().to_string_lossy().to_string();
            fs::read(p.path()).ok().map(|bytes| (name, bytes))
        })
        .collect();

    let results = ImageFingerprinter::fingerprint_batch(&images);

    let mut output_data = Vec::new();
    for (name, fp_result) in results {
        if let Ok(fp) = fp_result {
            let exact_hash = fp.exact_hash().iter().fold(String::new(), |mut acc, &b| {
                use std::fmt::Write;
                write!(&mut acc, "{:02x}", b).unwrap();
                acc
            });
            output_data.push(serde_json::json!({
                "filename": name,
                "exact_hash": exact_hash,
                "global_phash": format!("{:016x}", fp.phash().global_phash()),
                "global_dhash": format!("{:016x}", fp.dhash().global_phash()),
            }));
        }
    }

    let json = serde_json::to_string_pretty(&output_data)?;
    fs::write(&output, json)?;

    println!("\nWrote {} fingerprints to {}", output_data.len(), output);

    Ok(())
}
