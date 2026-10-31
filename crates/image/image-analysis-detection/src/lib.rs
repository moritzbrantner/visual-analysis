#![doc = include_str!("../README.md")]

mod color_blob;
mod mask_proposal;
mod onnx_bundle;
mod onnx_face;
mod onnx_object;
pub mod surface;
#[cfg(test)]
mod tests;
mod types;
mod yunet;

/// Compatibility re-export for callers that previously imported this from the ONNX crate.
pub use image_analysis_processing::{ChannelOrder, OnnxImagePreprocessing, OnnxImageTensor};

pub use color_blob::{ColorBlobDetectionOptions, ColorBlobDetector, TargetColor};
pub use mask_proposal::{
    segment_to_detection, segments_to_detections, ImageDetectionRequest, MaskProposalDetector,
};
pub use onnx_bundle::{
    face_detection_options_from_bundle, object_detection_options_from_bundle, options_from_bundle,
    validate_onnx_vision_bundle, OnnxVisionBundleInfo,
};
pub use onnx_face::{
    OnnxFaceDetectionOptions, OnnxFaceDetectionOutput, OnnxFaceDetectionRunner, OnnxFaceDetector,
    UnavailableOnnxFaceDetectionRunner,
};
pub use onnx_object::{
    decode_object_detections, BoxFormat, OnnxObjectDetectionOptions, OnnxObjectDetectionOutput,
    OnnxObjectDetectionRunner, OnnxObjectDetector, UnavailableOnnxRunner,
};
pub use types::{
    FaceBox, FaceDetection, FaceDetectionPreset, FaceDetectorBackend, FaceLandmarks,
    FrameDetections, ImageDetection,
};
pub use yunet::decode_yunet_face_detections;
