use imgfprint::ImageFingerprinter;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <query_image> <database_dir> [threshold]",
            args[0]
        );
        eprintln!("Example: {} query.jpg ./images 0.7", args[0]);
        std::process::exit(1);
    }

    let query_path = &args[1];
    let db_dir = &args[2];
    let threshold: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.7);

    let query_bytes = fs::read(query_path)?;
    let query_fp = ImageFingerprinter::fingerprint(&query_bytes)?;

    println!("Query: {}", query_path);
    println!("Threshold: {:.2}\n", threshold);
    println!("Searching database...\n");

    let db_paths: Vec<_> = fs::read_dir(db_dir)?
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

    let mut matches: Vec<(String, f32)> = Vec::new();

    for path in db_paths {
        let name = path.file_name().to_string_lossy().to_string();

        if let Ok(bytes) = fs::read(path.path()) {
            if let Ok(fp) = ImageFingerprinter::fingerprint(&bytes) {
                let sim = ImageFingerprinter::compare(&query_fp, &fp);

                if sim.score >= threshold {
                    matches.push((name, sim.score));
                }
            }
        }
    }

    matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Found {} similar images:\n", matches.len());

    for (name, score) in matches {
        println!("  {:.2} - {}", score, name);
    }

    Ok(())
}
