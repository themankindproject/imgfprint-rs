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
use std::path::Path;
use tract_onnx::prelude::*;

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
        Self {
            input_size: 224,
            // CLIP normalization values
            normalize_mean: [0.48145466, 0.4578275, 0.40821073],
            normalize_std: [0.26862954, 0.26130258, 0.27577711],
            normalize_output: true,
        }
    }
}

impl LocalProviderConfig {
    /// Creates a configuration for CLIP ViT-B/32 models.
    pub fn clip_vit_base_patch32() -> Self {
        Self {
            input_size: 224,
            normalize_mean: [0.48145466, 0.4578275, 0.40821073],
            normalize_std: [0.26862954, 0.26130258, 0.27577711],
            normalize_output: true,
        }
    }

    /// Creates a configuration for CLIP ViT-L/14 models.
    pub fn clip_vit_large_patch14() -> Self {
        Self {
            input_size: 336,
            normalize_mean: [0.48145466, 0.4578275, 0.40821073],
            normalize_std: [0.26862954, 0.26130258, 0.27577711],
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
    model: TypedModel,
    config: LocalProviderConfig,
}

impl std::fmt::Debug for LocalProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalProvider")
            .field("config", &self.config)
            .field("model", &"<TypedModel>")
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

        // Get the model type
        let typed_model: TypedModel = model.model().clone();

        Ok(Self {
            model: typed_model,
            config,
        })
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
        let model = tract_onnx::onnx()
            .model_for_read(&mut model_bytes.as_ref())
            .map_err(|e| {
                ImgFprintError::ProviderError(format!("Failed to parse ONNX model: {}", e))
            })?
            .into_optimized()
            .map_err(|e| ImgFprintError::ProviderError(format!("Failed to optimize model: {}", e)))?
            .into_runnable()
            .map_err(|e| {
                ImgFprintError::ProviderError(format!("Failed to make model runnable: {}", e))
            })?;

        // Get the model type
        let typed_model: TypedModel = model.model().clone();

        Ok(Self {
            model: typed_model,
            config,
        })
    }

    /// Returns the configuration of this provider.
    pub fn config(&self) -> &LocalProviderConfig {
        &self.config
    }

    /// Preprocesses an image for the model.
    ///
    /// This function:
    /// 1. Decodes the image
    /// 2. Resizes to the configured input size
    /// 3. Converts to RGB float tensor
    /// 4. Applies normalization
    fn preprocess_image(&self, image_bytes: &[u8]) -> Result<Tensor, ImgFprintError> {
        // Decode the image
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| ImgFprintError::DecodeError(format!("Failed to decode image: {}", e)))?;

        // Resize to input size
        let resized = img.resize_exact(
            self.config.input_size as u32,
            self.config.input_size as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB
        let rgb_img = resized.to_rgb8();

        // Create tensor with shape [1, 3, H, W] (batch, channels, height, width)
        let size = self.config.input_size;
        let mut tensor_data: Vec<f32> = Vec::with_capacity(3 * size * size);

        // Fill in CHW format (channels first)
        for c in 0..3 {
            for y in 0..size {
                for x in 0..size {
                    let pixel = rgb_img.get_pixel(x as u32, y as u32);
                    let value = pixel[c] as f32 / 255.0;
                    // Normalize
                    let normalized =
                        (value - self.config.normalize_mean[c]) / self.config.normalize_std[c];
                    tensor_data.push(normalized);
                }
            }
        }

        // Create tensor
        let tensor = Tensor::from_shape(&[1, 3, size, size], &tensor_data).map_err(|e| {
            ImgFprintError::ProcessingError(format!("Failed to create tensor: {}", e))
        })?;

        Ok(tensor)
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

        // Create a runnable model for this inference
        let runnable = self
            .model
            .clone()
            .into_runnable()
            .map_err(|e| ImgFprintError::ProviderError(format!("Model error: {}", e)))?;

        // Run inference
        let output = runnable
            .run(tvec!(input_tensor.into()))
            .map_err(|e| ImgFprintError::ProviderError(format!("Inference failed: {}", e)))?;

        // Extract the embedding vector
        let output_tensor = output
            .get(0)
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
}
