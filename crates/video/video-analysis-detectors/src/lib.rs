#![doc = include_str!("../README.md")]

pub mod canonical;
pub mod surface;

pub use canonical::{analyze_content_source, detect_source, CanonicalContentAnalysis};
pub use scenedetect_core::{
    AdaptiveDetectorConfig as AdaptiveDetector, BoundaryReviewOptions, ContentDetectorConfig,
    DetectionOptions, DetectorConfig, HashDetectorConfig as HashDetector,
    HistogramDetectorConfig as HistogramDetector, MinSceneLenPolicy,
    ThresholdDetectorConfig as ThresholdDetector,
};
pub use video_analysis_core::{ContentDetector, ContentWeights, FlashFilterMode};

/// Marks the registry-owned canonical detector seam used by this compatibility package.
pub const CANONICAL_SCENE_OWNER: &str = "scenedetect-core";
