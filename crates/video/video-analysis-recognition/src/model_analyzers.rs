use video_analysis_core::{
    AnalysisEvent, Observation, Result, TextAnalyzer, TextSegment, VideoAnalyzer, VideoFrame,
};

use model_runtime::{ModelRuntimeBackend, ModelTask};

use crate::{
    normalize_predictions, PredictionRepairOptions, RawPose2dPrediction, RawPose3dPrediction,
    RawPrediction,
};

/// Trait for vision model backend implementations.
pub trait VisionModelBackend {
    /// Returns task.
    fn task(&self) -> ModelTask;
    /// Returns runtime backend.
    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::Custom("vision".to_string())
    }
    /// Returns predict frame.
    fn predict_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<RawPrediction>>;
}

/// Trait for pose model backend implementations.
pub trait PoseModelBackend {
    /// Returns task.
    fn task(&self) -> ModelTask {
        ModelTask::PoseEstimation2d
    }

    /// Returns runtime backend.
    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::Custom("pose".to_string())
    }

    /// Returns predict frame.
    fn predict_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<RawPose2dPrediction>>;
}

/// Trait for pose lift model backend implementations.
pub trait PoseLiftModelBackend {
    /// Returns task.
    fn task(&self) -> ModelTask {
        ModelTask::PoseLifting3d
    }

    /// Returns runtime backend.
    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::Custom("pose_lift".to_string())
    }

    /// Returns lift poses.
    fn lift_poses(&mut self, sequence: &[RawPose2dPrediction]) -> Result<Vec<RawPose3dPrediction>>;
}

/// Trait for text model backend implementations.
pub trait TextModelBackend {
    /// Returns task.
    fn task(&self) -> ModelTask;
    /// Returns runtime backend.
    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::Custom("text".to_string())
    }
    /// Returns predict text.
    fn predict_text(&mut self, segment: &TextSegment<'_>) -> Result<Vec<RawPrediction>>;
}

/// Data type for model video analyzer.
pub struct ModelVideoAnalyzer<B> {
    name: String,
    backend: B,
    repair: PredictionRepairOptions,
}

impl<B> ModelVideoAnalyzer<B> {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, backend: B) -> Self {
        Self {
            name: name.into(),
            backend,
            repair: PredictionRepairOptions::default(),
        }
    }

    /// Returns repair options.
    pub fn repair_options(mut self, value: PredictionRepairOptions) -> Self {
        self.repair = value;
        self
    }

    /// Returns backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns backend mut.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: VisionModelBackend> VideoAnalyzer for ModelVideoAnalyzer<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        let task = self.backend.task();
        let raw = self.backend.predict_frame(frame)?;
        Ok(
            normalize_predictions(raw, &task, Some((frame.width, frame.height)), self.repair)
                .into_iter()
                .map(|prediction| prediction.to_observation(self.name()))
                .collect(),
        )
    }
}

/// Data type for model text analyzer.
pub struct ModelTextAnalyzer<B> {
    name: String,
    backend: B,
    repair: PredictionRepairOptions,
}

impl<B> ModelTextAnalyzer<B> {
    /// Creates a new value.
    pub fn new(name: impl Into<String>, backend: B) -> Self {
        Self {
            name: name.into(),
            backend,
            repair: PredictionRepairOptions::default(),
        }
    }

    /// Returns repair options.
    pub fn repair_options(mut self, value: PredictionRepairOptions) -> Self {
        self.repair = value;
        self
    }

    /// Returns backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns backend mut.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

impl<B: TextModelBackend> TextAnalyzer for ModelTextAnalyzer<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_segment(&mut self, segment: &TextSegment<'_>) -> Result<Vec<AnalysisEvent>> {
        let task = self.backend.task();
        let raw = self.backend.predict_text(segment)?;
        Ok(normalize_predictions(raw, &task, None, self.repair)
            .into_iter()
            .map(|prediction| {
                let mut event = prediction.to_event(self.name());
                if let Some(timestamp) = segment.timestamp {
                    event = event.at_timestamp(timestamp);
                }
                event
            })
            .collect())
    }
}
