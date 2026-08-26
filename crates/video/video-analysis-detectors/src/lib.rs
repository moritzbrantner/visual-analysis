#![doc = include_str!("../README.md")]

pub mod canonical;
pub mod surface;

pub use canonical::{analyze_content_source, CanonicalContentAnalysis};
pub use scenedetect_core::{
    AdaptiveDetectorConfig as AdaptiveDetector, DetectorConfig, HashDetectorConfig as HashDetector,
    HistogramDetectorConfig as HistogramDetector, ThresholdDetectorConfig as ThresholdDetector,
};
pub use video_analysis_core::{ContentDetector, ContentWeights, FlashFilterMode};

/// Marks the registry-owned canonical detector seam used by this compatibility package.
pub const CANONICAL_SCENE_OWNER: &str = "scenedetect-core";
