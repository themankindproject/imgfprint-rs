use imgfprint::ImageFingerprinter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <image1.jpg> <image2.jpg>", args[0]);
        std::process::exit(1);
    }

    let img1 = std::fs::read(&args[1])?;
    let img2 = std::fs::read(&args[2])?;

    let fp1 = ImageFingerprinter::fingerprint(&img1)?;
    let fp2 = ImageFingerprinter::fingerprint(&img2)?;

    let sim = ImageFingerprinter::compare(&fp1, &fp2);

    println!("Image 1: {}", args[1]);
    println!("Image 2: {}", args[2]);
    println!();
    println!("Similarity Score: {:.2}", sim.score);
    println!("Exact Match: {}", sim.exact_match);
    println!("Perceptual Distance: {}", sim.perceptual_distance);

    if sim.score > 0.8 {
        println!("\n✓ Images are very similar!");
    } else if sim.score > 0.5 {
        println!("\n✓ Images are somewhat similar.");
    } else {
        println!("\n✗ Images are different.");
    }

    Ok(())
}
