use imgfprint::{ImageFingerprinter, ImgFprintError};

fn main() {
    let test_cases = vec![
        ("empty bytes", vec![]),
        ("too small", vec![0u8; 100]),
        ("invalid format", b"NOT_AN_IMAGE".to_vec()),
        ("corrupted jpeg", vec![0xff, 0xd8, 0xff, 0x00, 0x00, 0x00]),
    ];

    for (name, bytes) in test_cases {
        match ImageFingerprinter::fingerprint(&bytes) {
            Ok(fp) => println!("{}: unexpectedly succeeded", name),
            Err(e) => {
                let error_type = match &e {
                    ImgFprintError::DecodeError(_) => "DecodeError",
                    ImgFprintError::InvalidImage(_) => "InvalidImage",
                    ImgFprintError::UnsupportedFormat(_) => "UnsupportedFormat",
                    ImgFprintError::ImageTooSmall(_) => "ImageTooSmall",
                    ImgFprintError::ProcessingError(_) => "ProcessingError",
                    ImgFprintError::ProviderError(_) => "ProviderError",
                    ImgFprintError::InvalidEmbedding(_) => "InvalidEmbedding",
                    ImgFprintError::EmbeddingDimensionMismatch { .. } => {
                        "EmbeddingDimensionMismatch"
                    }
                };
                println!("{}: {} - {}", name, error_type, e);
            }
        }
    }

    println!("\nError handling tips:");
    println!("- Check specific variants with matches! macro");
    println!("- Use ImgFprintError::invalid_image() for custom messages");
    println!("- All variants implement std::error::Error");
}
