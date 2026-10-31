//! ONNX face detection: options, runners, and the detector.

use std::collections::BTreeMap;

use image_analysis_core::ImageView;
use image_analysis_processing::{
    preprocess_image_for_model, validate_image_model_preprocessing, ChannelOrder,
    ImageModelPreprocessing, ImageModelTensor,
};
use video_analysis_core::{DetectError, Result};

use crate::onnx_bundle::validate_threshold;
#[cfg(feature = "onnx")]
use crate::onnx_bundle::{runtime_onnx_error, validate_onnx_face_detection_bundle};
use crate::{
    decode_yunet_face_detections, face_detection_options_from_bundle, FaceDetection,
    FaceDetectorBackend,
};

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX face detection options.
pub struct OnnxFaceDetectionOptions {
    /// The preprocessing value.
    pub preprocessing: ImageModelPreprocessing,
    /// The score threshold value.
    pub score_threshold: f32,
    /// The NMS threshold value.
    pub nms_threshold: f32,
    /// The top k value.
    pub top_k: usize,
}

impl Default for OnnxFaceDetectionOptions {
    fn default() -> Self {
        Self {
            preprocessing: ImageModelPreprocessing {
                input_width: 320,
                input_height: 320,
                rescale_factor: 1.0,
                mean: [0.0, 0.0, 0.0],
                std: [1.0, 1.0, 1.0],
                channel_order: ChannelOrder::Bgr,
            },
            score_threshold: 0.9,
            nms_threshold: 0.3,
            top_k: 5000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Data type for ONNX face detection output.
pub struct OnnxFaceDetectionOutput {
    /// Named output tensors.
    pub tensors: BTreeMap<String, runtime_onnx::OnnxF32Tensor>,
}

/// Trait for ONNX face detection runner implementations.
pub trait OnnxFaceDetectionRunner {
    /// Runs face detection.
    fn run_face_detection(&mut self, input: &ImageModelTensor) -> Result<OnnxFaceDetectionOutput>;
}

#[derive(Debug, Clone, Default)]
/// Data type for unavailable ONNX face detection runner.
pub struct UnavailableOnnxFaceDetectionRunner;

impl OnnxFaceDetectionRunner for UnavailableOnnxFaceDetectionRunner {
    fn run_face_detection(&mut self, _input: &ImageModelTensor) -> Result<OnnxFaceDetectionOutput> {
        Err(DetectError::Source(
            "native ONNX face detection execution is unavailable; construct with a runner or enable `onnx`"
                .to_string(),
        ))
    }
}

#[cfg(feature = "onnx")]
impl OnnxFaceDetectionRunner for runtime_onnx::OnnxSession {
    fn run_face_detection(&mut self, input: &ImageModelTensor) -> Result<OnnxFaceDetectionOutput> {
        use runtime_onnx::{OnnxRunner, OnnxTensor, OnnxTensorValue};

        let metadata = self.metadata().map_err(runtime_onnx_error)?;
        let input_name = runtime_onnx::input_name(&metadata, 0)
            .map_err(runtime_onnx_error)?
            .to_string();
        let outputs = self
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
        let mut tensors = BTreeMap::new();
        for (index, output) in outputs.iter().enumerate() {
            if let Ok(tensor) =
                runtime_onnx::f32_output_by_name_or_index(&outputs, &output.name, index)
            {
                tensors.insert(output.name.clone(), tensor.clone());
            }
        }
        Ok(OnnxFaceDetectionOutput { tensors })
    }
}

#[derive(Debug)]
/// Data type for ONNX face detector.
pub struct OnnxFaceDetector<R = UnavailableOnnxFaceDetectionRunner> {
    options: OnnxFaceDetectionOptions,
    runner: R,
}

#[cfg(not(feature = "onnx"))]
impl OnnxFaceDetector<UnavailableOnnxFaceDetectionRunner> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        let options = face_detection_options_from_bundle(&bundle)?;
        Ok(Self {
            options,
            runner: UnavailableOnnxFaceDetectionRunner,
        })
    }
}

#[cfg(feature = "onnx")]
impl OnnxFaceDetector<runtime_onnx::OnnxSession> {
    /// Builds this value from bundle.
    pub fn from_bundle(bundle: model_runtime::ModelBundle) -> Result<Self> {
        use runtime_onnx::OnnxRunner;

        let info = validate_onnx_face_detection_bundle(&bundle)?;
        let runner = runtime_onnx::OnnxSession::from_file(info.model_path)
            .map_err(runtime_onnx_error)?;
        let mut options = face_detection_options_from_bundle(&bundle)?;
        let metadata = runner.metadata().map_err(runtime_onnx_error)?;
        let input = metadata.inputs.first().ok_or_else(|| {
            DetectError::InvalidArgument("YuNet model has no image input".to_string())
        })?;
        configure_face_model_input(&mut options.preprocessing, input)?;
        Ok(Self { options, runner })
    }
}

#[cfg(feature = "onnx")]
pub(crate) fn configure_face_model_input(
    preprocessing: &mut ImageModelPreprocessing,
    input: &runtime_onnx::OnnxIoInfo,
) -> Result<()> {
    use runtime_onnx::OnnxDimension;

    let [batch, channels, height, width] = input.dimensions.as_slice() else {
        return Err(DetectError::InvalidArgument(
            "YuNet image input must have NCHW dimensions".to_string(),
        ));
    };
    if matches!(batch, OnnxDimension::Fixed(value) if *value != 1)
        || matches!(channels, OnnxDimension::Fixed(value) if *value != 3)
    {
        return Err(DetectError::InvalidArgument(
            "YuNet image input requires one three-channel image".to_string(),
        ));
    }
    // Unlike OpenCV DNN, ONNX Runtime enforces the model's declared fixed shape.
    for (dimension, target) in [
        (width, &mut preprocessing.input_width),
        (height, &mut preprocessing.input_height),
    ] {
        if let OnnxDimension::Fixed(value) = dimension {
            *target = u32::try_from(*value).map_err(|_| {
                DetectError::InvalidArgument("YuNet input dimension exceeds u32".to_string())
            })?;
        }
    }
    validate_image_model_preprocessing(preprocessing)
}

impl<R: OnnxFaceDetectionRunner> OnnxFaceDetector<R> {
    /// Builds this value from runner.
    pub fn from_runner(bundle: model_runtime::ModelBundle, runner: R) -> Result<Self> {
        Ok(Self {
            options: face_detection_options_from_bundle(&bundle)?,
            runner,
        })
    }

    /// Returns this value with options.
    pub fn with_options(options: OnnxFaceDetectionOptions, runner: R) -> Result<Self> {
        validate_image_model_preprocessing(&options.preprocessing)?;
        validate_threshold(options.score_threshold)?;
        validate_threshold(options.nms_threshold)?;
        if options.top_k == 0 {
            return Err(DetectError::InvalidArgument(
                "ONNX face detection top_k must be greater than zero".to_string(),
            ));
        }
        Ok(Self { options, runner })
    }
}

impl<R: OnnxFaceDetectionRunner> FaceDetectorBackend for OnnxFaceDetector<R> {
    fn detect_faces(&mut self, image: &ImageView<'_>) -> Result<Vec<FaceDetection>> {
        let input = preprocess_image_for_model(image, &self.options.preprocessing)?;
        let output = self.runner.run_face_detection(&input)?;
        decode_yunet_face_detections(&output, &self.options, (image.width, image.height))
    }
}
