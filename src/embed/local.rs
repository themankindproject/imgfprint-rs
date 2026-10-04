//! Local embedding provider using ONNX inference.
//!
//! This module provides a local embedding provider that runs vision models
//! (such as CLIP) directly on the local machine using ONNX Runtime via tract.
//!
//! ## Features
//!
//! - Pure Rust implementation (no Python dependencies)
//! - Supports ONNX format models
//! - Configurable input image size
//! - Automatic image preprocessing (resize, normalize)
//!
//! ## Example
//!
//! ```rust,ignore
//! use imgfprint::LocalProvider;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a local CLIP model
//! let provider = LocalProvider::from_file("clip-vit-base-patch32.onnx")?;
//!
//! // Generate embedding for an image
//! let image_bytes = std::fs::read("image.jpg")?;
//! let embedding = provider.embed(&image_bytes)?;
//! # Ok(())
//! # }
//! ```

use crate::embed::{Embedding, EmbeddingProvider};
use crate::error::ImgFprintError;
use crate::imgproc::decode::{decode_image_with_config, PreprocessConfig};
use image::DynamicImage;
use std::path::Path;
use tract_onnx::prelude::*;

type RunnableOnnxModel =
    RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// CHW model input following the CLIP reference preprocessing: scale the
/// shorter side to `input_size` (bicubic), center-crop to a square, map to
/// `[0, 1]`, then normalize each channel with the configured mean/std.
///
/// Preserving the aspect ratio matters: stretching a non-square image to a
/// square (the previous behavior) shifts embeddings away from what the model
/// was trained on.
fn clip_input(image: &DynamicImage, config: &LocalProviderConfig) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation)] // model input sizes are tiny
    let size = config.input_size as u32;
    let rgb = image
        .resize_to_fill(size, size, image::imageops::FilterType::CatmullRom)
        .to_rgb8();
    let raw = rgb.as_raw();
    let pixels = config.input_size * config.input_size;
    debug_assert_eq!(raw.len(), 3 * pixels);

    let mut data = Vec::with_capacity(3 * pixels);
    for c in 0..3 {
        let mean = config.normalize_mean[c];
        let std = config.normalize_std[c];
        data.extend(
            raw[c..]
                .iter()
                .step_by(3)
                .map(|&v| (f32::from(v) / 255.0 - mean) / std),
        );
    }
    data
}

/// Configuration for the local embedding provider.
#[derive(Debug, Clone)]
pub struct LocalProviderConfig {
    /// Input image size (width and height) expected by the model.
    /// Common values: 224 (CLIP ViT-B/32), 336 (CLIP ViT-L/14)
    pub input_size: usize,

    /// Mean values for normalization (RGB order).
    /// CLIP uses [0.48145466, 0.4578275, 0.40821073]
    pub normalize_mean: [f32; 3],

    /// Standard deviation values for normalization (RGB order).
    /// CLIP uses [0.26862954, 0.26130258, 0.27577711]
    pub normalize_std: [f32; 3],

    /// Whether to normalize the output embedding (L2 normalization).
    pub normalize_output: bool,
}

impl Default for LocalProviderConfig {
    fn default() -> Self {
        Self::clip_vit_base_patch32()
    }
}

impl LocalProviderConfig {
    /// Creates a configuration for CLIP ViT-B/32 models.
    #[must_use]
    pub fn clip_vit_base_patch32() -> Self {
        Self {
            input_size: 224,
            normalize_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            normalize_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            normalize_output: true,
        }
    }

    /// Creates a configuration for CLIP ViT-L/14 models.
    #[must_use]
    pub fn clip_vit_large_patch14() -> Self {
        Self {
            input_size: 336,
            normalize_mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            normalize_std: [0.268_629_54, 0.261_302_6, 0.275_777_1],
            normalize_output: true,
        }
    }
}

/// A local embedding provider that runs ONNX models.
///
/// This provider loads a vision model in ONNX format and uses it to
/// generate semantic embeddings for images. The model runs locally
/// without requiring external API calls.
///
/// # Thread Safety
///
/// `LocalProvider` is thread-safe and can be shared across threads.
/// The underlying ONNX model is reference-counted internally.
///
/// # Example
///
/// ```rust,ignore
/// use imgfprint::LocalProvider;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let provider = LocalProvider::from_file("model.onnx")?;
///
/// let image = std::fs::read("image.jpg")?;
/// let embedding = provider.embed(&image)?;
///
/// println!("Generated embedding with {} dimensions", embedding.len());
/// # Ok(())
/// # }
/// ```
pub struct LocalProvider {
    model: RunnableOnnxModel,
    config: LocalProviderConfig,
}

impl std::fmt::Debug for LocalProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalProvider")
            .field("config", &self.config)
            .field("model", &"<RunnableModel>")
            .finish()
    }
}

impl LocalProvider {
    /// Creates a new LocalProvider from an ONNX model file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError::ProviderError`] if:
    /// - The file cannot be read
    /// - The model cannot be parsed
    /// - The model is not a valid vision model
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use imgfprint::LocalProvider;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let provider = LocalProvider::from_file("clip.onnx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ImgFprintError> {
        let config = LocalProviderConfig::default();
        Self::from_file_with_config(path, config)
    }

    /// Creates a new LocalProvider from an ONNX model file with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the ONNX model file
    /// * `config` - Configuration for image preprocessing
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError::ProviderError`] if the model cannot be loaded.
    pub fn from_file_with_config<P: AsRef<Path>>(
        path: P,
        config: LocalProviderConfig,
    ) -> Result<Self, ImgFprintError> {
        let model = tract_onnx::onnx()
            .model_for_path(&path)
            .map_err(|e| {
                ImgFprintError::ProviderError(format!(
                    "Failed to load ONNX model from {}: {}",
                    path.as_ref().display(),
                    e
                ))
            })?
            .into_optimized()
            .map_err(|e| ImgFprintError::ProviderError(format!("Failed to optimize model: {}", e)))?
            .into_runnable()
            .map_err(|e| {
                ImgFprintError::ProviderError(format!("Failed to make model runnable: {}", e))
            })?;

        Ok(Self { model, config })
    }

    /// Creates a new LocalProvider from ONNX model bytes.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Raw bytes of the ONNX model
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError::ProviderError`] if the model cannot be parsed.
    pub fn from_bytes(model_bytes: &[u8]) -> Result<Self, ImgFprintError> {
        let config = LocalProviderConfig::default();
        Self::from_bytes_with_config(model_bytes, config)
    }

    /// Creates a new LocalProvider from ONNX model bytes with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Raw bytes of the ONNX model
    /// * `config` - Configuration for image preprocessing
    ///
    /// # Errors
    ///
    /// Returns [`ImgFprintError::ProviderError`] if the model cannot be parsed.
    pub fn from_bytes_with_config(
        model_bytes: &[u8],
        config: LocalProviderConfig,
    ) -> Result<Self, ImgFprintError> {
        let mut cursor = std::io::Cursor::new(model_bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| {
                ImgFprintError::ProviderError(format!("Failed to parse ONNX model: {}", e))
            })?
            .into_optimized()
            .map_err(|e| ImgFprintError::ProviderError(format!("Failed to optimize model: {}", e)))?
            .into_runnable()
            .map_err(|e| {
                ImgFprintError::ProviderError(format!("Failed to make model runnable: {}", e))
            })?;

        Ok(Self { model, config })
    }

    /// Returns the configuration of this provider.
    pub fn config(&self) -> &LocalProviderConfig {
        &self.config
    }

    /// Decodes `image_bytes` and builds the `[1, 3, size, size]` input tensor.
    ///
    /// Decoding goes through the crate's guarded decoder (size and
    /// dimension caps, decompression-bomb limit, EXIF orientation), so an
    /// untrusted upload cannot exhaust memory here either.
    fn preprocess_image(&self, image_bytes: &[u8]) -> Result<Tensor, ImgFprintError> {
        let size = self.config.input_size;
        if size == 0 {
            return Err(ImgFprintError::invalid_config(
                "LocalProviderConfig::input_size must be > 0",
            ));
        }
        let guards = PreprocessConfig {
            // Any size is fine: the image is resized to `input_size` anyway.
            min_dimension: 1,
            ..PreprocessConfig::default()
        };
        let img = decode_image_with_config(image_bytes, &guards)?;
        let data = clip_input(&img, &self.config);

        Tensor::from_shape(&[1, 3, size, size], &data)
            .map_err(|e| ImgFprintError::ProcessingError(format!("Failed to create tensor: {}", e)))
    }

    /// L2 normalizes a vector.
    fn l2_normalize(vector: &mut [f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

impl EmbeddingProvider for LocalProvider {
    fn embed(&self, image: &[u8]) -> Result<Embedding, ImgFprintError> {
        // Preprocess the image
        let input_tensor = self.preprocess_image(image)?;

        // Run inference using cached model (no clone needed)
        let output = self
            .model
            .run(tvec!(input_tensor.into()))
            .map_err(|e| ImgFprintError::ProviderError(format!("Inference failed: {}", e)))?;

        // Extract the embedding vector
        let output_tensor = output
            .first()
            .ok_or_else(|| ImgFprintError::ProviderError("Empty model output".to_string()))?;

        let embedding_vec: Vec<f32> = output_tensor
            .as_slice::<f32>()
            .map_err(|e| ImgFprintError::ProviderError(format!("Failed to extract output: {}", e)))?
            .to_vec();

        // Apply L2 normalization if configured
        let mut embedding_vec = embedding_vec;
        if self.config.normalize_output {
            Self::l2_normalize(&mut embedding_vec);
        }

        // Create and return the embedding
        Embedding::new(embedding_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = LocalProviderConfig::default();
        assert_eq!(config.input_size, 224);
        assert!(config.normalize_output);
    }

    #[test]
    fn test_config_clip_vit_base() {
        let config = LocalProviderConfig::clip_vit_base_patch32();
        assert_eq!(config.input_size, 224);
    }

    #[test]
    fn test_config_clip_vit_large() {
        let config = LocalProviderConfig::clip_vit_large_patch14();
        assert_eq!(config.input_size, 336);
    }

    #[test]
    fn test_l2_normalize() {
        let mut vec = vec![3.0, 4.0];
        LocalProvider::l2_normalize(&mut vec);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0, 0.0];
        LocalProvider::l2_normalize(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn clip_input_is_chw_normalized_and_center_cropped() {
        // 300x100: left third red, middle third green, right third blue.
        // Shorter side -> 224 gives 672x224; the centered 224x224 crop is
        // exactly the green third, so every pixel must normalize to green.
        let img = image::RgbImage::from_fn(300, 100, |x, _| match x / 100 {
            0 => image::Rgb([255, 0, 0]),
            1 => image::Rgb([0, 255, 0]),
            _ => image::Rgb([0, 0, 255]),
        });
        let config = LocalProviderConfig::default();
        let data = clip_input(&DynamicImage::ImageRgb8(img), &config);
        let plane = config.input_size * config.input_size;
        assert_eq!(data.len(), 3 * plane);

        let expect = |c: usize, v: f32| (v - config.normalize_mean[c]) / config.normalize_std[c];
        // Interior pixels only: the bicubic kernel blends ~2px at the edges.
        let side = config.input_size;
        for y in 8..side - 8 {
            for x in 8..side - 8 {
                let i = y * side + x;
                assert!((data[i] - expect(0, 0.0)).abs() < 0.02, "R at ({x},{y})");
                assert!(
                    (data[plane + i] - expect(1, 1.0)).abs() < 0.02,
                    "G at ({x},{y})"
                );
                assert!(
                    (data[2 * plane + i] - expect(2, 0.0)).abs() < 0.02,
                    "B at ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn clip_input_handles_tiny_and_portrait_images() {
        let config = LocalProviderConfig::default();
        let plane = config.input_size * config.input_size;
        for (w, h) in [(1, 1), (3, 500), (500, 3), (224, 224)] {
            let img = DynamicImage::ImageRgb8(image::RgbImage::new(w, h));
            assert_eq!(clip_input(&img, &config).len(), 3 * plane, "{w}x{h}");
        }
    }
}
