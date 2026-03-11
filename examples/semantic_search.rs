use imgfprint::{EmbeddingProvider, LocalProvider, LocalProviderConfig};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!(
            "Usage: {} <model.onnx> <query_image> <search_dir> [threshold]",
            args[0]
        );
        eprintln!("Example: {} clip.onnx query.jpg ./images 0.85", args[0]);
        eprintln!();
        eprintln!("This example requires a CLIP model in ONNX format.");
        eprintln!("You can download one from HuggingFace:");
        eprintln!("  https://huggingface.co/openai/clip-vit-base-patch32");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let query_path = &args[2];
    let search_dir = &args[3];
    let threshold: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.85);

    println!("Loading model from: {}", model_path);

    // Load the local CLIP model
    let provider = LocalProvider::from_file_with_config(
        model_path,
        LocalProviderConfig::clip_vit_base_patch32(),
    )?;

    println!("Model loaded successfully!");
    println!(
        "Input size: {}x{}",
        provider.config().input_size,
        provider.config().input_size
    );
    println!();

    // Generate embedding for query image
    println!("Processing query image: {}", query_path);
    let query_bytes = fs::read(query_path)?;
    let query_embedding = provider.embed(&query_bytes)?;
    println!("Query embedding: {} dimensions", query_embedding.len());
    println!();

    println!("Searching in: {}", search_dir);
    println!("Similarity threshold: {:.2}", threshold);
    println!();

    // Collect all image files
    let image_paths: Vec<_> = fs::read_dir(search_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let ext = e
                .path()
                .extension()
                .map(|s| s.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            ["jpg", "jpeg", "png", "gif", "webp", "bmp"].contains(&ext.as_str())
        })
        .map(|e| e.path())
        .collect();

    println!("Found {} images to compare", image_paths.len());
    println!("Processing...\n");

    let mut results: Vec<(String, f32, usize)> = Vec::new();

    for path in &image_paths {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        match fs::read(path) {
            Ok(bytes) => match provider.embed(&bytes) {
                Ok(embedding) => {
                    match imgfprint::semantic_similarity(&query_embedding, &embedding) {
                        Ok(similarity) => {
                            if similarity >= threshold {
                                results.push((filename.to_string(), similarity, embedding.len()));
                            }
                        }
                        Err(e) => eprintln!("Error comparing {}: {}", filename, e),
                    }
                }
                Err(e) => eprintln!("Error embedding {}: {}", filename, e),
            },
            Err(e) => eprintln!("Error reading {}: {}", filename, e),
        }
    }

    // Sort by similarity (highest first)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display results
    if results.is_empty() {
        println!("No similar images found above threshold {:.2}", threshold);
    } else {
        println!("Found {} semantically similar images:\n", results.len());
        println!("{:<12} {:<8} {}", "SIMILARITY", "DIM", "FILENAME");
        println!("{}", "-".repeat(60));

        for (filename, similarity, dims) in &results {
            let bar = similarity_bar(*similarity);
            println!(
                "{:.4} {:<4} {:<4} {} {}",
                similarity,
                bar,
                dims,
                if *similarity >= 0.95 { "★" } else { " " },
                filename
            );
        }
    }

    println!();
    println!("Legend:");
    println!("  ★ = Very high similarity (≥0.95)");
    println!("  ████ = Similarity bar (████ = 1.0, ░░░░ = 0.0)");

    Ok(())
}

fn similarity_bar(similarity: f32) -> &'static str {
    match (similarity * 10.0) as i32 {
        10 => "██████████",
        9 => "█████████░",
        8 => "████████░░",
        7 => "███████░░░",
        6 => "██████░░░░",
        5 => "█████░░░░░",
        4 => "████░░░░░░",
        3 => "███░░░░░░░",
        2 => "██░░░░░░░░",
        1 => "█░░░░░░░░░",
        _ => "░░░░░░░░░░",
    }
}
