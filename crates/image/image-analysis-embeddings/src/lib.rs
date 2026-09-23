#![doc = include_str!("../README.md")]

pub mod surface;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use image_analysis_core::{ImageView, OwnedImage};
use image_analysis_detection::{FaceDetection, FaceLandmarks};
use image_analysis_processing::{
    crop_image_rect, image_model_preprocessing_from_config, image_to_model_tensor,
    preprocess_image_for_model,
    validate_image_model_preprocessing, ChannelOrder, ImageModelPreprocessing, ImageModelTensor,
};
use math_geometry_2d::RectU32;
use model_runtime::{HuggingFaceModelSpec, ModelRuntimeBackend, ModelTask};
use serde::{Deserialize, Serialize};
use video_analysis_core::{BoundingBox, DetectError, Result};
use vision_core::{VisualDetectionKind, VisualEmbedding, VisualRegion};

/// Embedding task family for still images and faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEmbeddingTask {
    /// Dense image embedding.
    ImageEmbedding,
    /// Dense face embedding.
    FaceEmbedding,
}

impl ImageEmbeddingTask {
    /// Returns all embedding task variants.
    pub const ALL: &'static [Self] = &[Self::ImageEmbedding, Self::FaceEmbedding];

    /// Returns the stable API path segment for this task.
    pub fn path_segment(self) -> &'static str {
        match self {
            Self::ImageEmbedding => "embed",
            Self::FaceEmbedding => "face-embed",
        }
    }
}

/// Runtime families for image embedding execution and postprocessing.
pub type ImageEmbeddingRuntime = ModelRuntimeBackend;

/// Image embedding model metadata for UI and CLI discovery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageEmbeddingCatalogEntry {
    /// Stable local entry id.
    pub id: String,
    /// Upstream model id or strategy id.
    pub model_id: String,
    /// Task family.
    pub task: ImageEmbeddingTask,
    /// Preferred runtime.
    pub runtime: ImageEmbeddingRuntime,
    /// Whether the runtime is available in the default contributor build.
    pub supported: bool,
    /// Optional fallback entry id.
    pub fallback: Option<String>,
    /// Human-readable note for unsupported or fallback-only paths.
    pub note: Option<String>,
}

/// Variants describing image embedding preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageEmbeddingPreset {
    /// The CLIP ViT-B/32 variant.
    #[default]
    ClipVitBasePatch32,
    /// The Xenova CLIP ViT-B/32 ONNX variant.
    XenovaClipVitBasePatch32Onnx,
}

impl ImageEmbeddingPreset {
    /// Returns the model spec for this preset.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        match self {
            Self::ClipVitBasePatch32 => {
                HuggingFaceModelSpec::new("openai/clip-vit-base-patch32", ModelTask::ImageEmbedding)
                    .name("clip-vit-base-patch32")
                    .file("config.json")
                    .file("preprocessor_config.json")
                    .first_available_file(["model.safetensors", "pytorch_model.bin"])
            }
            Self::XenovaClipVitBasePatch32Onnx => {
                HuggingFaceModelSpec::new("Xenova/clip-vit-base-patch32", ModelTask::ImageEmbedding)
                    .name("xenova-clip-vit-base-patch32-onnx")
                    .file("config.json")
                    .file("preprocessor_config.json")
                    .first_available_file([
                        "onnx/vision_model_quantized.onnx",
                        "onnx/vision_model.onnx",
                    ])
            }
        }
    }
}

/// Variants describing face embedding preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FaceEmbeddingPreset {
    /// The OpenCV SFace ONNX variant.
    #[default]
    OpenCvSFace,
}

impl FaceEmbeddingPreset {
    /// Returns the model spec for this preset.
    pub fn model_spec(self) -> HuggingFaceModelSpec {
        match self {
            Self::OpenCvSFace => {
                HuggingFaceModelSpec::new("opencv/face_recognition_sface", ModelTask::FaceEmbedding)
                    .name("opencv-sface-onnx")
                    .file("face_recognition_sface_2021dec.onnx")
            }
        }
    }
}

/// Dense image embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageEmbedding {
    /// Embedding values.
    pub vector: Vec<f32>,
    /// Adapter-specific attributes.
    pub attributes: BTreeMap<String, String>,
}

impl ImageEmbedding {
    /// Creates a new image embedding.
    pub fn new(vector: Vec<f32>) -> Result<Self> {
        validate_vector(&vector)?;
        Ok(Self {
            vector,
            attributes: BTreeMap::new(),
        })
    }

    /// Adds an adapter-specific attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this image embedding into a shared visual embedding.
    pub fn to_visual_embedding(&self) -> Result<VisualEmbedding> {
        let mut visual = VisualEmbedding::new(self.vector.clone()).map_err(vision_error)?;
        visual.attributes = self.attributes.clone();
        Ok(visual)
    }
}

/// Dense face embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEmbedding {
    /// Optional source face region.
    pub region: Option<BoundingBox>,
    /// Embedding values.
    pub vector: Vec<f32>,
    /// Adapter-specific attributes.
    pub attributes: BTreeMap<String, String>,
}

impl FaceEmbedding {
    /// Creates a new face embedding.
    pub fn new(vector: Vec<f32>) -> Result<Self> {
        validate_vector(&vector)?;
        Ok(Self {
            region: None,
            vector,
            attributes: BTreeMap::new(),
        })
    }

    /// Sets the source face region.
    pub fn region(mut self, value: BoundingBox) -> Self {
        self.region = Some(value);
        self
    }

    /// Adds an adapter-specific attribute.
    pub fn attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Converts this face embedding into a shared visual embedding.
    pub fn to_visual_embedding(&self) -> Result<VisualEmbedding> {
        let mut visual = VisualEmbedding::new(self.vector.clone())
            .map_err(vision_error)?
            .kind(VisualDetectionKind::Face)
            .map_err(vision_error)?;
        if let Some(region) = self.region {
            visual = visual
                .region(VisualRegion::from(region))
                .map_err(vision_error)?;
        }
        visual.attributes = self.attributes.clone();
        Ok(visual)
    }
}

impl TryFrom<ImageEmbedding> for VisualEmbedding {
    type Error = DetectError;

    fn try_from(value: ImageEmbedding) -> std::result::Result<Self, Self::Error> {
        value.to_visual_embedding()
    }
}

impl TryFrom<&ImageEmbedding> for VisualEmbedding {
    type Error = DetectError;

    fn try_from(value: &ImageEmbedding) -> std::result::Result<Self, Self::Error> {
        value.to_visual_embedding()
    }
}

impl TryFrom<FaceEmbedding> for VisualEmbedding {
    type Error = DetectError;

    fn try_from(value: FaceEmbedding) -> std::result::Result<Self, Self::Error> {
        value.to_visual_embedding()
    }
}

impl TryFrom<&FaceEmbedding> for VisualEmbedding {
    type Error = DetectError;

    fn try_from(value: &FaceEmbedding) -> std::result::Result<Self, Self::Error> {
        value.to_visual_embedding()
    }
}

/// Backend trait for dense image embedding.
pub trait ImageEmbedderBackend {
    /// Embeds one image.
    fn embed_image(&mut self, image: &ImageView<'_>) -> Result<ImageEmbedding>;
}

/// Backend trait for dense face embedding.
pub trait FaceEmbedderBackend {
    /// Embeds a face region from one image.
    fn embed_face(
        &mut self,
        image: &ImageView<'_>,
        detection: Option<&FaceDetection>,
    ) -> Result<FaceEmbedding>;
}

/// Returns embedding catalog entries.
pub fn image_embedding_catalog(
    task: Option<ImageEmbeddingTask>,
) -> Vec<ImageEmbeddingCatalogEntry> {
    let entries = vec![
        catalog_entry(
            "clip-vit-base-patch32",
            ImageEmbeddingTask::ImageEmbedding,
            ImageEmbeddingPreset::ClipVitBasePatch32.model_spec(),
            ModelRuntimeBackend::Onnx,
            false,
            Some("hashed-image-fallback"),
            Some("Download the model with model-runtime and run it through an image backend."),
        ),
        catalog_entry(
            "xenova-clip-vit-base-patch32-onnx",
            ImageEmbeddingTask::ImageEmbedding,
            ImageEmbeddingPreset::XenovaClipVitBasePatch32Onnx.model_spec(),
            ModelRuntimeBackend::Onnx,
            false,
            Some("hashed-image-fallback"),
            Some("ONNX model variant for browser/server embedding backends."),
        ),
        catalog_entry(
            "opencv-sface-onnx",
            ImageEmbeddingTask::FaceEmbedding,
            FaceEmbeddingPreset::OpenCvSFace.model_spec(),
            ModelRuntimeBackend::Onnx,
            false,
            Some("imported-face-embedding"),
            Some("Download the model with model-runtime and run it through an image backend."),
        ),
    ];

    entries
        .into_iter()
        .filter(|entry| task.is_none_or(|task| entry.task == task))
        .collect()
}

/// Compatibility alias for callers migrating off aggregate task catalog names.
pub fn model_catalog(task: Option<ImageEmbeddingTask>) -> Vec<ImageEmbeddingCatalogEntry> {
    image_embedding_catalog(task)
}

/// Parses an embedding task string.
pub fn parse_task(input: &str) -> Option<ImageEmbeddingTask> {
    ImageEmbeddingTask::ALL
        .iter()
        .copied()
        .find(|task| task.path_segment() == input)
        .or(match input {
            "image_embedding" | "embedding" => Some(ImageEmbeddingTask::ImageEmbedding),
            "face_embedding" => Some(ImageEmbeddingTask::FaceEmbedding),
            _ => None,
        })
}

/// Returns preset ids registered for image and face embeddings.
pub fn registered_image_embedding_presets() -> Vec<String> {
    image_embedding_catalog(None)
        .into_iter()
        .map(|entry| entry.id)
        .collect()
}

/// Returns a schema-oriented summary.
pub fn schema_summary() -> serde_json::Value {
    serde_json::json!({
        "tasks": ImageEmbeddingTask::ALL
            .iter()
            .map(|task| task.path_segment())
            .collect::<Vec<_>>(),
        "models": image_embedding_catalog(None),
        "registeredPresets": registered_image_embedding_presets()
    })
}

fn catalog_entry(
    id: &str,
    task: ImageEmbeddingTask,
    spec: HuggingFaceModelSpec,
    runtime: ModelRuntimeBackend,
    supported: bool,
    fallback: Option<&str>,
    note: Option<&str>,
) -> ImageEmbeddingCatalogEntry {
    ImageEmbeddingCatalogEntry {
        id: id.to_string(),
        model_id: spec.repo_id_value().unwrap_or(id).to_string(),
        task,
        runtime,
        supported,
        fallback: fallback.map(str::to_string),
        note: note.map(str::to_string),
    }
}

fn validate_vector(vector: &[f32]) -> Result<()> {
    if vector.is_empty() {
        return Err(DetectError::InvalidArgument(
            "embedding vector must not be empty".to_string(),
        ));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "embedding vector values must be finite".to_string(),
        ));
    }
    Ok(())
}

fn vision_error(error: vision_core::VisionError) -> DetectError {
    DetectError::InvalidArgument(error.to_string())
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX image embedding options.
pub struct OnnxImageEmbeddingOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// Expected vector size.
    pub expected_vector_size: Option<usize>,
    /// Whether to L2 normalize output.
    pub normalize: bool,
}

impl Default for OnnxImageEmbeddingOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing {
                input_width: 224,
                input_height: 224,
                rescale_factor: 1.0 / 255.0,
                mean: [0.48145466, 0.4578275, 0.40821073],
                std: [0.26862954, 0.261_302_6, 0.275_777_1],
                channel_order: ChannelOrder::Rgb,
            },
            expected_vector_size: None,
            normalize: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX face embedding options.
pub struct OnnxFaceEmbeddingOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// Expected vector size.
    pub expected_vector_size: Option<usize>,
    /// Whether to L2 normalize output.
    pub normalize: bool,
}

impl Default for OnnxFaceEmbeddingOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing {
                input_width: 112,
                input_height: 112,
                rescale_factor: 1.0,
                mean: [0.0, 0.0, 0.0],
                std: [1.0, 1.0, 1.0],
                channel_order: ChannelOrder::Rgb,
            },
            expected_vector_size: None,
            normalize: true,
        }
    }
}

const SFACE_REFERENCE_SIZE: f64 = 112.0;
const SFACE_REFERENCE_LANDMARKS: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

#[derive(Debug, Clone, Copy, PartialEq)]
struct SimilarityTransform {
    a: f64,
    b: f64,
    tx: f64,
    ty: f64,
}

/// Aligns a YuNet-style five-landmark face to the canonical OpenCV SFace frame.
///
/// OpenCV's SFace reference path aligns the five normalized facial landmarks
/// before feature extraction. This keeps the model input stable under modest
/// translation, scale, and in-plane rotation instead of stretching a raw
/// bounding-box crop into the 112x112 recognition input.
pub fn align_face_for_sface(
    image: &ImageView<'_>,
    landmarks: &FaceLandmarks,
    output_width: u32,
    output_height: u32,
) -> Result<OwnedImage> {
    image.validate()?;
    if output_width == 0 || output_height == 0 {
        return Err(DetectError::InvalidDimensions {
            width: output_width,
            height: output_height,
        });
    }
    let transform =
        sface_similarity_transform(landmarks, image.width, image.height, output_width, output_height)?;
    warp_similarity_rgb(image, transform, output_width, output_height)
}

fn sface_similarity_transform(
    landmarks: &FaceLandmarks,
    image_width: u32,
    image_height: u32,
    output_width: u32,
    output_height: u32,
) -> Result<SimilarityTransform> {
    if landmarks.points.len() != 5 {
        return Err(DetectError::InvalidArgument(format!(
            "SFace alignment requires exactly five landmarks, got {}",
            landmarks.points.len()
        )));
    }
    let source = landmarks
        .points
        .iter()
        .map(|point| {
            [
                f64::from(point[0]) * f64::from(image_width),
                f64::from(point[1]) * f64::from(image_height),
            ]
        })
        .collect::<Vec<_>>();
    if source.iter().flatten().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "SFace alignment landmarks must be finite".to_string(),
        ));
    }

    let scale_x = f64::from(output_width) / SFACE_REFERENCE_SIZE;
    let scale_y = f64::from(output_height) / SFACE_REFERENCE_SIZE;
    let destination = SFACE_REFERENCE_LANDMARKS.map(|point| {
        [point[0] * scale_x, point[1] * scale_y]
    });
    let source_mean = [
        source.iter().map(|point| point[0]).sum::<f64>() / 5.0,
        source.iter().map(|point| point[1]).sum::<f64>() / 5.0,
    ];
    let destination_mean = [
        destination.iter().map(|point| point[0]).sum::<f64>() / 5.0,
        destination.iter().map(|point| point[1]).sum::<f64>() / 5.0,
    ];

    let mut denominator = 0.0;
    let mut a_numerator = 0.0;
    let mut b_numerator = 0.0;
    for (source, destination) in source.iter().zip(destination.iter()) {
        let sx = source[0] - source_mean[0];
        let sy = source[1] - source_mean[1];
        let dx = destination[0] - destination_mean[0];
        let dy = destination[1] - destination_mean[1];
        denominator += sx * sx + sy * sy;
        a_numerator += sx * dx + sy * dy;
        b_numerator += sx * dy - sy * dx;
    }
    if !denominator.is_finite() || denominator <= f64::EPSILON {
        return Err(DetectError::InvalidArgument(
            "SFace alignment landmarks are degenerate".to_string(),
        ));
    }
    let a = a_numerator / denominator;
    let b = b_numerator / denominator;
    let tx = destination_mean[0] - a * source_mean[0] + b * source_mean[1];
    let ty = destination_mean[1] - b * source_mean[0] - a * source_mean[1];
    if [a, b, tx, ty].iter().any(|value| !value.is_finite()) || a * a + b * b <= f64::EPSILON {
        return Err(DetectError::InvalidArgument(
            "SFace alignment transform is not invertible".to_string(),
        ));
    }
    Ok(SimilarityTransform { a, b, tx, ty })
}

fn warp_similarity_rgb(
    image: &ImageView<'_>,
    transform: SimilarityTransform,
    output_width: u32,
    output_height: u32,
) -> Result<OwnedImage> {
    let denominator = transform.a * transform.a + transform.b * transform.b;
    if !denominator.is_finite() || denominator <= f64::EPSILON {
        return Err(DetectError::InvalidArgument(
            "SFace alignment transform is not invertible".to_string(),
        ));
    }

    let mut data = vec![0_u8; output_width as usize * output_height as usize * 3];
    for y in 0..output_height {
        for x in 0..output_width {
            let dx = f64::from(x) - transform.tx;
            let dy = f64::from(y) - transform.ty;
            let source_x = (transform.a * dx + transform.b * dy) / denominator;
            let source_y = (-transform.b * dx + transform.a * dy) / denominator;
            let rgb = bilinear_rgb(image, source_x, source_y);
            let offset = (y as usize * output_width as usize + x as usize) * 3;
            data[offset..offset + 3].copy_from_slice(&rgb);
        }
    }
    OwnedImage::new_rgb(output_width, output_height, data)
}

fn bilinear_rgb(image: &ImageView<'_>, x: f64, y: f64) -> [u8; 3] {
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx = x - x0 as f64;
    let wy = y - y0 as f64;
    let samples = [
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x1, y0, wx * (1.0 - wy)),
        (x0, y1, (1.0 - wx) * wy),
        (x1, y1, wx * wy),
    ];
    let mut result = [0.0_f64; 3];
    for (sample_x, sample_y, weight) in samples {
        if sample_x < 0
            || sample_y < 0
            || sample_x >= i64::from(image.width)
            || sample_y >= i64::from(image.height)
        {
            continue;
        }
        let pixel = image.pixel_rgb(sample_x as u32, sample_y as u32);
        for channel in 0..3 {
            result[channel] += f64::from(pixel[channel]) * weight;
        }
    }
    result.map(|value| value.round().clamp(0.0, 255.0) as u8)
}

#[derive(Debug, Clone, PartialEq)]
struct OnnxEmbeddingBundleInfo {
    config_path: Option<PathBuf>,
    preprocessor_config_path: Option<PathBuf>,
    model_path: PathBuf,
}

#[cfg(any(feature = "onnx", test))]
const EMBEDDING_OUTPUT_NAME_HINTS: &[&str] =
    &["image_embeds", "embeds", "embedding", "pooler_output"];

/// Trait for ONNX image embedding runner implementations.
pub trait OnnxImageEmbeddingRunner {
    /// Runs image embedding.
    fn run_image_embedding(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>>;
}

/// Trait for ONNX face embedding runner implementations.
pub trait OnnxFaceEmbeddingRunner {
    /// Runs face embedding.
    fn run_face_embedding(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>>;
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable ONNX image embedding runner.
pub struct UnavailableOnnxImageEmbeddingRunner;

impl OnnxImageEmbeddingRunner for UnavailableOnnxImageEmbeddingRunner {
    fn run_image_embedding(
        &mut self,
        _input: &ImageModelTensor,
    ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
        Err(DetectError::Source(
            "native ONNX image embedding execution is unavailable; construct with a runner or enable `onnx`"
                .to_string(),
        ))
    }
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable ONNX face embedding runner.
pub struct UnavailableOnnxFaceEmbeddingRunner;

impl OnnxFaceEmbeddingRunner for UnavailableOnnxFaceEmbeddingRunner {
    fn run_face_embedding(
        &mut self,
        _input: &ImageModelTensor,
    ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
        Err(DetectError::Source(
            "native ONNX face embedding execution is unavailable; construct with a runner or enable `onnx`"
                .to_string(),
        ))
    }
}

#[cfg(feature = "onnx")]
impl OnnxImageEmbeddingRunner for runtime_onnx::OnnxSession {
    fn run_image_embedding(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
        run_session_f32_outputs(self, input)
    }
}

#[cfg(feature = "onnx")]
impl OnnxFaceEmbeddingRunner for runtime_onnx::OnnxSession {
    fn run_face_embedding(
        &mut self,
        input: &ImageModelTensor,
    ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
        run_session_f32_outputs(self, input)
    }
}

#[derive(Debug)]
/// Data type for ONNX image embedder.
pub struct OnnxImageEmbedder<R = UnavailableOnnxImageEmbeddingRunner> {
    options: OnnxImageEmbeddingOptions,
    runner: R,
}

#[cfg(not(feature = "onnx"))]
impl OnnxImageEmbedder<UnavailableOnnxImageEmbeddingRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        Ok(Self {
            options: image_embedding_options_from_bundle(&bundle)?,
            runner: UnavailableOnnxImageEmbeddingRunner,
        })
    }
}

#[cfg(feature = "onnx")]
impl OnnxImageEmbedder<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_embedding_bundle(&bundle, EmbeddingBundleKind::Image)?;
        Ok(Self {
            options: image_embedding_options_from_bundle(&bundle)?,
            runner: runtime_onnx::OnnxSession::from_file(info.model_path)
                .map_err(runtime_onnx_error)?,
        })
    }
}

impl<R: OnnxImageEmbeddingRunner> OnnxImageEmbedder<R> {
    /// Returns this value with options.
    pub fn with_options(options: OnnxImageEmbeddingOptions, runner: R) -> Result<Self> {
        validate_image_model_preprocessing(&options.preprocessing)?;
        validate_expected_size(options.expected_vector_size, "image")?;
        Ok(Self { options, runner })
    }
}

impl<R: OnnxImageEmbeddingRunner> ImageEmbedderBackend for OnnxImageEmbedder<R> {
    fn embed_image(&mut self, image: &ImageView<'_>) -> Result<ImageEmbedding> {
        let input = preprocess_image_for_model(image, &self.options.preprocessing)?;
        let output = self.runner.run_image_embedding(&input)?;
        let mut vector = select_embedding_vector(&output, self.options.expected_vector_size)?;
        if self.options.normalize {
            normalize_vector(&mut vector);
        }
        ImageEmbedding::new(vector)
    }
}

#[derive(Debug)]
/// Data type for ONNX face embedder.
pub struct OnnxFaceEmbedder<R = UnavailableOnnxFaceEmbeddingRunner> {
    options: OnnxFaceEmbeddingOptions,
    runner: R,
}

#[cfg(not(feature = "onnx"))]
impl OnnxFaceEmbedder<UnavailableOnnxFaceEmbeddingRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        Ok(Self {
            options: face_embedding_options_from_bundle(&bundle)?,
            runner: UnavailableOnnxFaceEmbeddingRunner,
        })
    }
}

#[cfg(feature = "onnx")]
impl OnnxFaceEmbedder<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let info = validate_onnx_embedding_bundle(&bundle, EmbeddingBundleKind::Face)?;
        Ok(Self {
            options: face_embedding_options_from_bundle(&bundle)?,
            runner: runtime_onnx::OnnxSession::from_file(info.model_path)
                .map_err(runtime_onnx_error)?,
        })
    }
}

impl<R: OnnxFaceEmbeddingRunner> OnnxFaceEmbedder<R> {
    /// Returns this value with options.
    pub fn with_options(options: OnnxFaceEmbeddingOptions, runner: R) -> Result<Self> {
        validate_image_model_preprocessing(&options.preprocessing)?;
        validate_expected_size(options.expected_vector_size, "face")?;
        Ok(Self { options, runner })
    }
}

impl<R: OnnxFaceEmbeddingRunner> FaceEmbedderBackend for OnnxFaceEmbedder<R> {
    fn embed_face(
        &mut self,
        image: &ImageView<'_>,
        detection: Option<&FaceDetection>,
    ) -> Result<FaceEmbedding> {
        image.validate()?;
        let cropped;
        let aligned;
        let (input, preprocessing_kind) = if let Some(detection) = detection {
            if let Some(landmarks) = detection.landmarks.as_ref() {
                aligned = align_face_for_sface(
                    image,
                    landmarks,
                    self.options.preprocessing.input_width,
                    self.options.preprocessing.input_height,
                )?;
                (
                    image_to_model_tensor(&aligned.as_view(), &self.options.preprocessing)?,
                    "sface-five-point-alignment",
                )
            } else {
                let region = face_detection_rect(detection, image)?;
                cropped = crop_image_rect(image, region)?;
                (
                    preprocess_image_for_model(&cropped.as_view(), &self.options.preprocessing)?,
                    "bounding-box-crop",
                )
            }
        } else {
            (
                preprocess_image_for_model(image, &self.options.preprocessing)?,
                "full-image",
            )
        };
        let output = self.runner.run_face_embedding(&input)?;
        let mut vector = select_embedding_vector(&output, self.options.expected_vector_size)?;
        if self.options.normalize {
            normalize_vector(&mut vector);
        }
        let mut embedding = FaceEmbedding::new(vector)?
            .attribute("facePreprocessing", preprocessing_kind);
        if let Some(detection) = detection {
            embedding = embedding.region(face_detection_box(detection, image)?);
        }
        Ok(embedding)
    }
}

/// Returns image embedding options from a model-runtime bundle.
pub fn image_embedding_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxImageEmbeddingOptions> {
    let info = validate_onnx_embedding_bundle(bundle, EmbeddingBundleKind::Image)?;
    let config = read_optional_json(info.config_path.as_deref())?;
    let preprocessing = preprocessing_from_bundle(
        info.preprocessor_config_path.as_deref(),
        OnnxImageEmbeddingOptions::default().preprocessing,
    )?;
    Ok(OnnxImageEmbeddingOptions {
        preprocessing,
        expected_vector_size: config.as_ref().and_then(embedding_size_from_config),
        normalize: true,
    })
}

/// Returns face embedding options from a model-runtime bundle.
pub fn face_embedding_options_from_bundle(
    bundle: &model_runtime::ModelBundle,
) -> Result<OnnxFaceEmbeddingOptions> {
    let info = validate_onnx_embedding_bundle(bundle, EmbeddingBundleKind::Face)?;
    let config = read_optional_json(info.config_path.as_deref())?;
    let preprocessing = preprocessing_from_bundle(
        info.preprocessor_config_path.as_deref(),
        OnnxFaceEmbeddingOptions::default().preprocessing,
    )?;
    Ok(OnnxFaceEmbeddingOptions {
        preprocessing,
        expected_vector_size: config.as_ref().and_then(embedding_size_from_config),
        normalize: true,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbeddingBundleKind {
    Image,
    Face,
}

fn validate_onnx_embedding_bundle(
    bundle: &model_runtime::ModelBundle,
    kind: EmbeddingBundleKind,
) -> Result<OnnxEmbeddingBundleInfo> {
    match (kind, &bundle.manifest.task) {
        (EmbeddingBundleKind::Image, ModelTask::ImageEmbedding) => {}
        (EmbeddingBundleKind::Face, ModelTask::FaceEmbedding) => {}
        (EmbeddingBundleKind::Face, ModelTask::Custom(task)) if task == "face_embedding" => {}
        (EmbeddingBundleKind::Image, task) => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX image embedding bundle {} task must be image_embedding, got {:?}",
                bundle_descriptor(bundle),
                task
            )))
        }
        (EmbeddingBundleKind::Face, task) => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX face embedding bundle {} task must be face_embedding, got {:?}",
                bundle_descriptor(bundle),
                task
            )))
        }
    }
    let onnx_files = bundle_files_with_extension(bundle, "onnx");
    let model_path = match onnx_files.as_slice() {
        [path] => path.clone(),
        [] => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX embedding bundle {} must contain exactly one `.onnx` model file selected by extension, found 0",
                bundle_descriptor(bundle)
            )))
        }
        files => {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX embedding bundle {} must contain exactly one `.onnx` model file selected by extension, found {}",
                bundle_descriptor(bundle),
                files.len()
            )))
        }
    };
    Ok(OnnxEmbeddingBundleInfo {
        config_path: bundle.file_path("config.json"),
        preprocessor_config_path: bundle.file_path("preprocessor_config.json"),
        model_path,
    })
}

fn bundle_descriptor(bundle: &model_runtime::ModelBundle) -> String {
    format!(
        "`{}` (task {:?})",
        bundle.manifest.name, bundle.manifest.task
    )
}

fn preprocessing_from_bundle(
    path: Option<&Path>,
    default: ImageModelPreprocessing,
) -> Result<ImageModelPreprocessing> {
    let preprocessing = if let Some(path) = path {
        image_model_preprocessing_from_config(&read_json(path)?)?
    } else {
        default
    };
    validate_image_model_preprocessing(&preprocessing)?;
    Ok(preprocessing)
}

fn embedding_size_from_config(config: &serde_json::Value) -> Option<usize> {
    ["projection_dim", "hidden_size", "embedding_size"]
        .iter()
        .find_map(|key| {
            config
                .get(*key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
        })
}

fn bundle_files_with_extension(
    bundle: &model_runtime::ModelBundle,
    extension: &str,
) -> Vec<PathBuf> {
    bundle
        .manifest
        .files
        .values()
        .filter(|file| {
            Path::new(&file.remote_path)
                .extension()
                .and_then(|value| value.to_str())
                == Some(extension)
        })
        .map(|file| bundle.root.join(&file.local_path))
        .collect()
}

fn read_optional_json(path: Option<&Path>) -> Result<Option<serde_json::Value>> {
    path.map(read_json).transpose()
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let data = std::fs::read(path)?;
    serde_json::from_slice(&data).map_err(|err| {
        DetectError::Source(format!("failed to parse JSON `{}`: {err}", path.display()))
    })
}

fn face_detection_rect(detection: &FaceDetection, image: &ImageView<'_>) -> Result<RectU32> {
    let bbox = face_detection_box(detection, image)?;
    Ok(RectU32::new(bbox.x, bbox.y, bbox.width, bbox.height)?)
}

fn face_detection_box(detection: &FaceDetection, image: &ImageView<'_>) -> Result<BoundingBox> {
    let width = image.width as f32;
    let height = image.height as f32;
    let x = (detection.bbox.x * width).floor().clamp(0.0, width) as u32;
    let y = (detection.bbox.y * height).floor().clamp(0.0, height) as u32;
    let max_x = ((detection.bbox.x + detection.bbox.width) * width)
        .ceil()
        .clamp(0.0, width) as u32;
    let max_y = ((detection.bbox.y + detection.bbox.height) * height)
        .ceil()
        .clamp(0.0, height) as u32;
    if max_x <= x || max_y <= y {
        return Err(DetectError::InvalidArgument(
            "face detection crop region is empty".to_string(),
        ));
    }
    BoundingBox::new(x, y, max_x - x, max_y - y)
}

fn validate_expected_size(size: Option<usize>, kind: &str) -> Result<()> {
    if let Some(size) = size {
        if size == 0 {
            return Err(DetectError::InvalidArgument(format!(
                "ONNX {kind} embedding expected vector size must be greater than zero"
            )));
        }
    }
    Ok(())
}

fn select_embedding_vector(
    outputs: &[runtime_onnx::OnnxF32Tensor],
    expected_size: Option<usize>,
) -> Result<Vec<f32>> {
    let vector = outputs
        .iter()
        .find(|tensor| expected_size.is_none_or(|size| tensor.values.len() == size))
        .ok_or_else(|| {
            let sizes = outputs
                .iter()
                .map(|tensor| tensor.values.len().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let expected = expected_size
                .map(|size| size.to_string())
                .unwrap_or_else(|| "any".to_string());
            DetectError::InvalidArgument(format!(
                "ONNX embedding output was missing expected vector size {expected}; available vector sizes: [{}]",
                sizes
            ))
        })?
        .values
        .clone();
    validate_embedding_vector(&vector)?;
    Ok(vector)
}

#[cfg(test)]
fn select_embedding_vector_from_named_outputs(
    outputs: &[runtime_onnx::OnnxNamedTensor],
    expected_size: Option<usize>,
) -> Result<Vec<f32>> {
    let ordered = ordered_embedding_f32_outputs(outputs);
    select_embedding_vector(&ordered, expected_size)
}

#[cfg(any(feature = "onnx", test))]
fn ordered_embedding_f32_outputs(
    outputs: &[runtime_onnx::OnnxNamedTensor],
) -> Vec<runtime_onnx::OnnxF32Tensor> {
    let mut selected = Vec::new();
    if let Some((preferred_index, tensor)) =
        outputs
            .iter()
            .enumerate()
            .find_map(|(index, output)| match &output.tensor {
                runtime_onnx::OnnxTensorValue::F32(tensor)
                    if embedding_output_name_is_preferred(&output.name) =>
                {
                    Some((index, tensor.clone()))
                }
                _ => None,
            })
    {
        selected.push(tensor);
        selected.extend(outputs.iter().enumerate().filter_map(|(index, output)| {
            if index == preferred_index {
                return None;
            }
            match &output.tensor {
                runtime_onnx::OnnxTensorValue::F32(tensor) => Some(tensor.clone()),
                _ => None,
            }
        }));
        return selected;
    }
    selected.extend(outputs.iter().filter_map(|output| match &output.tensor {
        runtime_onnx::OnnxTensorValue::F32(tensor) => Some(tensor.clone()),
        _ => None,
    }));
    selected
}

#[cfg(any(feature = "onnx", test))]
fn embedding_output_name_is_preferred(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    EMBEDDING_OUTPUT_NAME_HINTS
        .iter()
        .any(|hint| name.contains(hint))
}

fn validate_embedding_vector(vector: &[f32]) -> Result<()> {
    if vector.is_empty() {
        return Err(DetectError::InvalidArgument(
            "ONNX embedding output vector must not be empty".to_string(),
        ));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "ONNX embedding output vector values must be finite".to_string(),
        ));
    }
    Ok(())
}

fn normalize_vector(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON && norm.is_finite() {
        for value in vector {
            *value /= norm;
        }
    }
}

#[cfg(feature = "onnx")]
fn run_session_f32_outputs(
    session: &mut runtime_onnx::OnnxSession,
    input: &ImageModelTensor,
) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
    use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

    let metadata = session.metadata().map_err(runtime_onnx_error)?;
    let input_name = runtime_onnx::input_name(&metadata, 0)
        .map_err(runtime_onnx_error)?
        .to_string();
    let outputs = session
        .run(vec![runtime_onnx::OnnxNamedTensor {
            name: input_name,
            tensor: OnnxTensorValue::F32(
                OnnxTensor::new(
                    vec![
                        1,
                        input.channels,
                        input.height as usize,
                        input.width as usize,
                    ],
                    input.values.clone(),
                )
                .map_err(runtime_onnx_error)?,
            ),
        }])
        .map_err(runtime_onnx_error)?;
    let selected = ordered_embedding_f32_outputs(&outputs);
    if selected.is_empty() {
        return Err(DetectError::InvalidArgument(
            "ONNX embedding session returned no f32 outputs".to_string(),
        ));
    }
    Ok(selected)
}

#[cfg(feature = "onnx")]
fn runtime_onnx_error(error: runtime_onnx::OnnxRuntimeError) -> DetectError {
    match error {
        runtime_onnx::OnnxRuntimeError::InvalidArgument(message)
        | runtime_onnx::OnnxRuntimeError::InvalidTensorShape(message) => {
            DetectError::InvalidArgument(message)
        }
        runtime_onnx::OnnxRuntimeError::Io(error) => DetectError::Io(error),
        other => DetectError::Source(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image_analysis_core::{ImagePixelFormat, OwnedImage};
    use image_analysis_detection::{FaceBox, FaceLandmarks};
    use model_runtime::{ModelBundle, ModelBundleFile, ModelBundleManifest};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Debug, Clone)]
    struct FakeImageRunner {
        outputs: Vec<runtime_onnx::OnnxF32Tensor>,
    }

    impl OnnxImageEmbeddingRunner for FakeImageRunner {
        fn run_image_embedding(
            &mut self,
            _input: &ImageModelTensor,
        ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
            Ok(self.outputs.clone())
        }
    }

    #[derive(Debug, Clone)]
    struct FakeFaceRunner {
        outputs: Vec<runtime_onnx::OnnxF32Tensor>,
        last_input_size: Rc<RefCell<Option<(u32, u32)>>>,
    }

    impl OnnxFaceEmbeddingRunner for FakeFaceRunner {
        fn run_face_embedding(
            &mut self,
            input: &ImageModelTensor,
        ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
            *self.last_input_size.borrow_mut() = Some((input.width, input.height));
            Ok(self.outputs.clone())
        }
    }

    fn tensor(values: Vec<f32>) -> runtime_onnx::OnnxF32Tensor {
        runtime_onnx::OnnxF32Tensor::new(vec![1, values.len()], values).unwrap()
    }

    fn unchecked_tensor(values: Vec<f32>) -> runtime_onnx::OnnxF32Tensor {
        runtime_onnx::OnnxF32Tensor {
            shape: vec![1, values.len()],
            values,
        }
    }

    fn named_tensor(name: &str, values: Vec<f32>) -> runtime_onnx::OnnxNamedTensor {
        runtime_onnx::OnnxNamedTensor {
            name: name.to_string(),
            tensor: runtime_onnx::OnnxTensorValue::F32(tensor(values)),
        }
    }

    fn image() -> OwnedImage {
        OwnedImage::new(4, 4, ImagePixelFormat::Rgb24, vec![128; 4 * 4 * 3], 4 * 3).unwrap()
    }

    fn coordinate_image(width: u32, height: u32) -> OwnedImage {
        let mut data = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[
                    x.min(255) as u8,
                    y.min(255) as u8,
                    ((x + y) / 2).min(255) as u8,
                ]);
            }
        }
        OwnedImage::new_rgb(width, height, data).unwrap()
    }

    fn sface_landmarks_for_transform(
        image_width: u32,
        image_height: u32,
        scale: f64,
        translate_x: f64,
        translate_y: f64,
    ) -> FaceLandmarks {
        FaceLandmarks::new(
            SFACE_REFERENCE_LANDMARKS
                .iter()
                .map(|point| {
                    [
                        ((point[0] * scale + translate_x) / f64::from(image_width)) as f32,
                        ((point[1] * scale + translate_y) / f64::from(image_height)) as f32,
                    ]
                })
                .collect(),
        )
        .unwrap()
    }

    fn test_bundle(
        task: ModelTask,
        files: &[(&str, &str, &str)],
    ) -> (tempfile::TempDir, ModelBundle) {
        let temp = tempfile::tempdir().unwrap();
        let mut manifest_files = BTreeMap::new();
        for (remote_path, local_path, contents) in files {
            let path = temp.path().join(local_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, contents).unwrap();
            manifest_files.insert(
                (*remote_path).to_string(),
                ModelBundleFile {
                    remote_path: (*remote_path).to_string(),
                    local_path: (*local_path).to_string(),
                    size_bytes: contents.len() as u64,
                },
            );
        }
        let bundle = ModelBundle {
            root: temp.path().to_path_buf(),
            manifest: ModelBundleManifest {
                schema_version: 1,
                name: "test".to_string(),
                repo_id: "owner/model".to_string(),
                revision: "main".to_string(),
                task,
                files: manifest_files,
            },
        };
        (temp, bundle)
    }

    #[test]
    fn embedding_catalog_reports_presets() {
        let catalog = image_embedding_catalog(None);
        assert!(catalog
            .iter()
            .any(|entry| entry.id == "xenova-clip-vit-base-patch32-onnx"));
        assert!(catalog.iter().any(|entry| entry.id == "opencv-sface-onnx"));
    }

    #[test]
    fn embedding_vectors_must_be_finite() {
        assert!(ImageEmbedding::new(vec![0.0, 1.0]).is_ok());
        assert!(ImageEmbedding::new(vec![f32::NAN]).is_err());
        assert!(FaceEmbedding::new(vec![f32::INFINITY]).is_err());
    }

    #[test]
    fn image_embedder_selects_expected_vector_and_normalizes() {
        let options = OnnxImageEmbeddingOptions {
            expected_vector_size: Some(2),
            normalize: true,
            ..OnnxImageEmbeddingOptions::default()
        };
        let runner = FakeImageRunner {
            outputs: vec![tensor(vec![9.0, 9.0, 9.0]), tensor(vec![3.0, 4.0])],
        };
        let mut embedder = OnnxImageEmbedder::with_options(options, runner).unwrap();
        let embedding = embedder.embed_image(&image().as_view()).unwrap();
        assert_eq!(embedding.vector, vec![0.6, 0.8]);
    }

    #[test]
    fn named_embedding_output_wins_over_fallback() {
        let outputs = vec![
            named_tensor("last_hidden_state", vec![9.0, 9.0]),
            named_tensor("image_embeds", vec![3.0, 4.0]),
        ];
        let vector = select_embedding_vector_from_named_outputs(&outputs, Some(2)).unwrap();
        assert_eq!(vector, vec![3.0, 4.0]);
    }

    #[test]
    fn named_pooler_output_wins_over_fallback() {
        let outputs = vec![
            named_tensor("last_hidden_state", vec![9.0, 9.0]),
            named_tensor("pooler_output", vec![5.0, 12.0]),
        ];
        let vector = select_embedding_vector_from_named_outputs(&outputs, Some(2)).unwrap();
        assert_eq!(vector, vec![5.0, 12.0]);
    }

    #[test]
    fn embedding_output_falls_back_to_first_f32_tensor() {
        let outputs = vec![named_tensor("last_hidden_state", vec![1.0, 2.0])];
        let vector = select_embedding_vector_from_named_outputs(&outputs, None).unwrap();
        assert_eq!(vector, vec![1.0, 2.0]);
    }

    #[test]
    fn image_embedder_can_skip_normalization() {
        let options = OnnxImageEmbeddingOptions {
            normalize: false,
            ..OnnxImageEmbeddingOptions::default()
        };
        let runner = FakeImageRunner {
            outputs: vec![tensor(vec![3.0, 4.0])],
        };
        let mut embedder = OnnxImageEmbedder::with_options(options, runner).unwrap();
        let embedding = embedder.embed_image(&image().as_view()).unwrap();
        assert_eq!(embedding.vector, vec![3.0, 4.0]);
    }

    #[test]
    fn image_embedder_rejects_non_finite_output_before_normalization() {
        let runner = FakeImageRunner {
            outputs: vec![unchecked_tensor(vec![1.0, f32::NAN])],
        };
        let mut embedder =
            OnnxImageEmbedder::with_options(OnnxImageEmbeddingOptions::default(), runner).unwrap();
        let err = embedder.embed_image(&image().as_view()).unwrap_err();
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn image_embedder_rejects_expected_size_mismatch() {
        let options = OnnxImageEmbeddingOptions {
            expected_vector_size: Some(3),
            ..OnnxImageEmbeddingOptions::default()
        };
        let runner = FakeImageRunner {
            outputs: vec![tensor(vec![1.0, 2.0])],
        };
        let mut embedder = OnnxImageEmbedder::with_options(options, runner).unwrap();
        let err = embedder.embed_image(&image().as_view()).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("expected vector size 3"));
        assert!(message.contains("2"));
    }

    #[test]
    fn image_embedder_rejects_empty_output_vector() {
        let runner = FakeImageRunner {
            outputs: vec![runtime_onnx::OnnxF32Tensor::new(vec![1, 0], vec![]).unwrap()],
        };
        let mut embedder =
            OnnxImageEmbedder::with_options(OnnxImageEmbeddingOptions::default(), runner).unwrap();
        let err = embedder.embed_image(&image().as_view()).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn sface_alignment_preserves_canonical_landmark_frame() {
        let image = coordinate_image(112, 112);
        let landmarks = sface_landmarks_for_transform(112, 112, 1.0, 0.0, 0.0);
        let aligned = align_face_for_sface(&image.as_view(), &landmarks, 112, 112).unwrap();
        assert_eq!(aligned, image);
    }

    #[test]
    fn sface_alignment_inverts_scale_and_translation() {
        let image = coordinate_image(224, 224);
        let landmarks = sface_landmarks_for_transform(224, 224, 2.0, 10.0, 12.0);
        let aligned = align_face_for_sface(&image.as_view(), &landmarks, 112, 112).unwrap();
        assert_eq!(aligned.as_view().pixel_rgb(0, 0), [10, 12, 11]);
        assert_eq!(aligned.as_view().pixel_rgb(56, 72), [122, 156, 139]);
    }

    #[test]
    fn sface_alignment_requires_exactly_five_landmarks() {
        let image = coordinate_image(112, 112);
        let landmarks = FaceLandmarks::new(vec![[0.25, 0.25], [0.75, 0.25]]).unwrap();
        let error = align_face_for_sface(&image.as_view(), &landmarks, 112, 112).unwrap_err();
        assert!(error.to_string().contains("exactly five landmarks"));
    }

    #[test]
    fn face_embedder_crops_detection_region_before_preprocessing() {
        let options = OnnxFaceEmbeddingOptions {
            preprocessing: ImageModelPreprocessing {
                input_width: 2,
                input_height: 2,
                ..OnnxFaceEmbeddingOptions::default().preprocessing
            },
            normalize: false,
            ..OnnxFaceEmbeddingOptions::default()
        };
        let last_input_size = Rc::new(RefCell::new(None));
        let runner = FakeFaceRunner {
            outputs: vec![tensor(vec![1.0, 2.0])],
            last_input_size: Rc::clone(&last_input_size),
        };
        let mut embedder = OnnxFaceEmbedder::with_options(options, runner).unwrap();
        let detection =
            FaceDetection::new(FaceBox::new(0.25, 0.25, 0.5, 0.5).unwrap(), 0.9).unwrap();
        let embedding = embedder
            .embed_face(&image().as_view(), Some(&detection))
            .unwrap();
        assert_eq!(*last_input_size.borrow(), Some((2, 2)));
        assert_eq!(
            embedding.region,
            Some(BoundingBox::new(1, 1, 2, 2).unwrap())
        );
        assert_eq!(embedding.vector, vec![1.0, 2.0]);
    }

    #[test]
    fn face_embedder_uses_five_point_alignment_when_landmarks_are_present() {
        #[derive(Debug)]
        struct InspectRunner {
            first_rgb: Rc<RefCell<Option<[f32; 3]>>>,
        }
        impl OnnxFaceEmbeddingRunner for InspectRunner {
            fn run_face_embedding(
                &mut self,
                input: &ImageModelTensor,
            ) -> Result<Vec<runtime_onnx::OnnxF32Tensor>> {
                let plane = input.width as usize * input.height as usize;
                *self.first_rgb.borrow_mut() = Some([
                    input.values[0],
                    input.values[plane],
                    input.values[2 * plane],
                ]);
                Ok(vec![tensor(vec![1.0, 2.0])])
            }
        }

        let first_rgb = Rc::new(RefCell::new(None));
        let runner = InspectRunner {
            first_rgb: Rc::clone(&first_rgb),
        };
        let options = OnnxFaceEmbeddingOptions {
            normalize: false,
            ..OnnxFaceEmbeddingOptions::default()
        };
        let mut embedder = OnnxFaceEmbedder::with_options(options, runner).unwrap();
        let detection = FaceDetection::new(
            FaceBox::new(0.05, 0.05, 0.85, 0.9).unwrap(),
            0.99,
        )
        .unwrap()
        .landmarks(sface_landmarks_for_transform(224, 224, 2.0, 10.0, 12.0));

        let embedding = embedder
            .embed_face(&coordinate_image(224, 224).as_view(), Some(&detection))
            .unwrap();

        assert_eq!(*first_rgb.borrow(), Some([10.0, 12.0, 11.0]));
        assert_eq!(
            embedding.attributes["facePreprocessing"],
            "sface-five-point-alignment"
        );
    }

    #[test]
    fn face_embedder_clamps_partially_out_of_bounds_detection_region() {
        let options = OnnxFaceEmbeddingOptions {
            preprocessing: ImageModelPreprocessing {
                input_width: 4,
                input_height: 4,
                ..OnnxFaceEmbeddingOptions::default().preprocessing
            },
            normalize: false,
            ..OnnxFaceEmbeddingOptions::default()
        };
        let last_input_size = Rc::new(RefCell::new(None));
        let runner = FakeFaceRunner {
            outputs: vec![tensor(vec![1.0, 2.0])],
            last_input_size: Rc::clone(&last_input_size),
        };
        let mut embedder = OnnxFaceEmbedder::with_options(options, runner).unwrap();
        let detection =
            FaceDetection::new(FaceBox::new(-0.25, -0.25, 0.75, 0.75).unwrap(), 0.9).unwrap();
        let embedding = embedder
            .embed_face(&image().as_view(), Some(&detection))
            .unwrap();
        assert_eq!(*last_input_size.borrow(), Some((4, 4)));
        assert_eq!(
            embedding.region,
            Some(BoundingBox::new(0, 0, 2, 2).unwrap())
        );
    }

    #[test]
    fn image_embedding_bundle_reads_config_and_preprocessing() {
        let (_temp, bundle) = test_bundle(
            ModelTask::ImageEmbedding,
            &[
                ("onnx/model.onnx", "onnx/model.onnx", "model"),
                ("config.json", "config.json", r#"{"projection_dim": 512}"#),
                (
                    "preprocessor_config.json",
                    "preprocessor_config.json",
                    r#"{"size":{"width":128,"height":96},"rescale_factor":0.5,"image_mean":[0.1,0.2,0.3],"image_std":[1.0,2.0,3.0],"channel_order":"bgr"}"#,
                ),
            ],
        );
        let options = image_embedding_options_from_bundle(&bundle).unwrap();
        assert_eq!(options.expected_vector_size, Some(512));
        assert_eq!(options.preprocessing.input_width, 128);
        assert_eq!(options.preprocessing.input_height, 96);
        assert_eq!(options.preprocessing.channel_order, ChannelOrder::Bgr);
        assert!(options.normalize);
    }

    #[test]
    fn face_embedding_bundle_accepts_custom_task_and_hidden_size() {
        let (_temp, bundle) = test_bundle(
            ModelTask::Custom("face_embedding".to_string()),
            &[
                ("model.onnx", "model.onnx", "model"),
                ("config.json", "config.json", r#"{"hidden_size": 128}"#),
            ],
        );
        let options = face_embedding_options_from_bundle(&bundle).unwrap();
        assert_eq!(options.expected_vector_size, Some(128));
    }

    #[test]
    fn embedding_bundle_rejects_missing_onnx_and_invalid_task() {
        let (_temp, missing) = test_bundle(
            ModelTask::ImageEmbedding,
            &[("config.json", "config.json", "{}")],
        );
        let err = image_embedding_options_from_bundle(&missing).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("test"));
        assert!(message.contains("ImageEmbedding"));
        assert!(message.contains("found 0"));

        let (_temp, invalid) = test_bundle(
            ModelTask::FaceDetection,
            &[("model.onnx", "model.onnx", "model")],
        );
        let err = image_embedding_options_from_bundle(&invalid).unwrap_err();
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn embedding_bundle_rejects_multiple_onnx_files_with_count() {
        let (_temp, bundle) = test_bundle(
            ModelTask::ImageEmbedding,
            &[
                ("model-a.onnx", "model-a.onnx", "a"),
                ("model-b.onnx", "model-b.onnx", "b"),
            ],
        );
        let err = image_embedding_options_from_bundle(&bundle).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("found 2"));
        assert!(message.contains("selected by extension"));
    }
}
