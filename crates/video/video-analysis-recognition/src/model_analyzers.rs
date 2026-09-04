use video_analysis_core::{
    AnalysisEvent, Observation, Result, TextAnalyzer, TextSegment, VideoAnalyzer, VideoFrame,
};

use model_runtime::{ModelRuntimeBackend, ModelTask};

use crate::{
    normalize_predictions, PredictionRepairOptions, RawPose2dPrediction, RawPose3dPrediction,
    RawPrediction,
};

#[cfg(feature = "ocr")]
use std::collections::BTreeSet;

#[cfg(feature = "ocr")]
use image_analysis_core::ImageView;
#[cfg(feature = "ocr")]
use image_analysis_ocr::{OcrBackend, OcrConfidence, OcrDocument, OcrRequest};
#[cfg(feature = "ocr")]
use video_analysis_core::{ObservationKind, Scene};

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

/// Adapts an image-analysis OCR backend to the shared video-analysis pipeline.
///
/// OCR algorithms and model/runtime policy remain owned by `image-analysis-ocr`.
/// This adapter only projects structured OCR output into timestampable video
/// observations so downstream video tracking and semantic analysis can reuse it.
#[cfg(feature = "ocr")]
pub struct OcrVideoAnalyzer<B> {
    name: String,
    backend: B,
    request: OcrRequest,
}

#[cfg(feature = "ocr")]
impl<B> OcrVideoAnalyzer<B> {
    /// Creates an OCR analyzer using the backend and the default OCR request.
    pub fn new(name: impl Into<String>, backend: B) -> Self {
        Self {
            name: name.into(),
            backend,
            request: OcrRequest::default(),
        }
    }

    /// Replaces the OCR request used for every sampled frame.
    pub fn request(mut self, request: OcrRequest) -> Self {
        self.request = request;
        self
    }

    /// Returns the OCR request.
    pub fn ocr_request(&self) -> &OcrRequest {
        &self.request
    }

    /// Returns the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Returns the backend mutably.
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Consumes the analyzer and returns its backend.
    pub fn into_backend(self) -> B {
        self.backend
    }
}

#[cfg(feature = "ocr")]
impl<B: OcrBackend> VideoAnalyzer for OcrVideoAnalyzer<B> {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        let image = ImageView::from_video_frame(frame)?;
        let document = self.backend.recognize_image(&image, &self.request)?;
        Ok(ocr_document_observations(self.name(), &document))
    }
}

/// Returns the deterministic representative-frame plan for a scene list.
///
/// Each scene contributes its first, midpoint, and final frame. Duplicate frame
/// indices are removed while preserving deterministic ascending order.
#[cfg(feature = "ocr")]
pub fn representative_scene_frames(scenes: &[Scene]) -> Result<BTreeSet<u64>> {
    if scenes.is_empty() {
        return Err(video_analysis_core::DetectError::InvalidArgument(
            "scene-aware OCR requires at least one scene".to_string(),
        ));
    }

    let mut frames = BTreeSet::new();
    for scene in scenes {
        let start = scene.start.frame_index;
        let end = scene.end.frame_index;
        if end < start {
            return Err(video_analysis_core::DetectError::InvalidArgument(format!(
                "scene end frame {end} precedes start frame {start}"
            )));
        }
        frames.insert(start);
        frames.insert(start + (end - start) / 2);
        frames.insert(end);
    }
    Ok(frames)
}

/// OCR analyzer that delegates to `OcrVideoAnalyzer` only for representative
/// frames from an existing canonical scene list.
///
/// The caller can run this analyzer through the ordinary `VideoAnalysisPipeline`.
/// Non-representative frames are skipped without invoking the OCR backend, so a
/// decoded source can still make one sequential pass while expensive recognition
/// is bounded by the scene sampling plan.
#[cfg(feature = "ocr")]
pub struct SceneAwareOcrVideoAnalyzer<B> {
    inner: OcrVideoAnalyzer<B>,
    representative_frames: BTreeSet<u64>,
}

#[cfg(feature = "ocr")]
impl<B> SceneAwareOcrVideoAnalyzer<B> {
    /// Creates a scene-aware OCR analyzer from a canonical scene list.
    pub fn new(name: impl Into<String>, backend: B, scenes: &[Scene]) -> Result<Self> {
        Ok(Self {
            inner: OcrVideoAnalyzer::new(name, backend),
            representative_frames: representative_scene_frames(scenes)?,
        })
    }

    /// Replaces the OCR request used for representative frames.
    pub fn request(mut self, request: OcrRequest) -> Self {
        self.inner = self.inner.request(request);
        self
    }

    /// Returns the exact frame indices selected for OCR.
    pub fn representative_frames(&self) -> &BTreeSet<u64> {
        &self.representative_frames
    }

    /// Returns the OCR backend.
    pub fn backend(&self) -> &B {
        self.inner.backend()
    }

    /// Returns the OCR backend mutably.
    pub fn backend_mut(&mut self) -> &mut B {
        self.inner.backend_mut()
    }

    /// Consumes the analyzer and returns the OCR backend.
    pub fn into_backend(self) -> B {
        self.inner.into_backend()
    }
}

#[cfg(feature = "ocr")]
impl<B: OcrBackend> VideoAnalyzer for SceneAwareOcrVideoAnalyzer<B> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn process_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<Observation>> {
        if !self
            .representative_frames
            .contains(&frame.position.frame_index)
        {
            return Ok(Vec::new());
        }
        self.inner.process_frame(frame)
    }
}

/// Projects one OCR document into deterministic video text observations.
///
/// Line-level output is preferred because it preserves useful layout geometry.
/// Blocks without lines are emitted at block granularity, and a backend that only
/// returns document text still produces one observation. Frame/timestamp metadata
/// is intentionally left unset so `VideoAnalysisPipeline` can stamp it uniformly.
#[cfg(feature = "ocr")]
pub fn ocr_document_observations(analyzer: &str, document: &OcrDocument) -> Vec<Observation> {
    let mut observations = Vec::new();

    for (block_index, block) in document.blocks.iter().enumerate() {
        if block.lines.is_empty() {
            if let Some(observation) = ocr_text_observation(
                analyzer,
                block.text.trim(),
                block.region,
                block.confidence,
                document,
                "block",
                block_index,
                None,
            ) {
                observations.push(observation);
            }
            continue;
        }

        for (line_index, line) in block.lines.iter().enumerate() {
            if let Some(observation) = ocr_text_observation(
                analyzer,
                line.text.trim(),
                line.region.or(block.region),
                line.confidence.or(block.confidence),
                document,
                "line",
                block_index,
                Some(line_index),
            ) {
                observations.push(observation);
            }
        }
    }

    if observations.is_empty() {
        if let Some(observation) = ocr_text_observation(
            analyzer,
            document.text.trim(),
            None,
            document.confidence,
            document,
            "document",
            0,
            None,
        ) {
            observations.push(observation);
        }
    }

    observations
}

#[cfg(feature = "ocr")]
fn ocr_text_observation(
    analyzer: &str,
    text: &str,
    region: Option<video_analysis_core::BoundingBox>,
    confidence: Option<OcrConfidence>,
    document: &OcrDocument,
    granularity: &str,
    block_index: usize,
    line_index: Option<usize>,
) -> Option<Observation> {
    if text.is_empty() {
        return None;
    }

    let mut observation = Observation::new(analyzer, ObservationKind::Text)
        .text(text)
        .attribute("ocr.granularity", granularity)
        .attribute("ocr.blockIndex", block_index.to_string());
    if let Some(line_index) = line_index {
        observation = observation.attribute("ocr.lineIndex", line_index.to_string());
    }
    if let Some(language) = document.language.as_deref() {
        observation = observation.attribute("language", language);
    }
    if let Some(region) = region {
        observation = observation.region(region);
    }
    if let Some(confidence) = confidence {
        observation = observation.score(confidence.value() as f32 / 100.0);
    }
    Some(observation)
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

#[cfg(all(test, feature = "ocr"))]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use image_analysis_ocr::{OcrTextBlock, OcrTextLine};
    use num_rational::Rational64;
    use video_analysis_core::{
        BoundingBox, FramePosition, ObservationKind, Scene, VideoAnalysisPipeline, VideoFrame,
    };

    use super::*;
    use crate::{analyze_video_text_semantics, VideoTextSemanticContext};

    struct FixtureOcr;

    impl OcrBackend for FixtureOcr {
        fn recognize_image(
            &mut self,
            image: &ImageView<'_>,
            _request: &OcrRequest,
        ) -> Result<OcrDocument> {
            let region = BoundingBox::new(1, 1, 2, 1)?;
            let line = OcrTextLine::new("Five Ways")?
                .region(region)
                .confidence(92);
            let block = OcrTextBlock::paragraph("Five Ways")?.line(line);
            Ok(OcrDocument::new("Five Ways", image.width, image.height)?
                .language("en")
                .block(block))
        }
    }

    struct CountingOcr {
        calls: Rc<Cell<usize>>,
    }

    impl OcrBackend for CountingOcr {
        fn recognize_image(
            &mut self,
            image: &ImageView<'_>,
            _request: &OcrRequest,
        ) -> Result<OcrDocument> {
            self.calls.set(self.calls.get() + 1);
            OcrDocument::new("Scene Text", image.width, image.height)
        }
    }

    fn position(frame_index: u64) -> FramePosition {
        FramePosition::from_frame_index(frame_index, Rational64::new(30, 1))
    }

    fn scene(start: u64, end: u64) -> Scene {
        Scene {
            start: position(start),
            end: position(end),
        }
    }

    #[test]
    fn video_pipeline_stamps_structured_ocr_observations() -> Result<()> {
        let analyzer = OcrVideoAnalyzer::new("ocr", FixtureOcr);
        let mut pipeline = VideoAnalysisPipeline::builder().analyzer(analyzer).build()?;
        let position = FramePosition::from_frame_index(30, Rational64::new(30, 1));
        let pixels = vec![0_u8; 4 * 2 * 3];
        let frame = VideoFrame::rgb24(position, 4, 2, &pixels)?;

        let result = pipeline.process_frame_ref(&frame)?;
        assert_eq!(result.observations.len(), 1);
        let observation = &result.observations[0];
        assert_eq!(observation.kind, ObservationKind::Text);
        assert_eq!(observation.text.as_deref(), Some("Five Ways"));
        assert_eq!(observation.frame, Some(position));
        assert_eq!(observation.timestamp, Some(position.timestamp));
        assert_eq!(observation.score, Some(0.92));
        assert_eq!(observation.region, Some(BoundingBox::new(1, 1, 2, 1)?));
        assert_eq!(
            observation.attributes.get("ocr.granularity").map(String::as_str),
            Some("line")
        );
        assert_eq!(
            observation.attributes.get("language").map(String::as_str),
            Some("en")
        );
        Ok(())
    }

    #[test]
    fn document_only_backends_still_emit_searchable_text() -> Result<()> {
        let document = OcrDocument::new("Title Card", 8, 8)?.confidence(80);
        let observations = ocr_document_observations("ocr", &document);
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].text.as_deref(), Some("Title Card"));
        assert_eq!(observations[0].score, Some(0.8));
        assert_eq!(
            observations[0]
                .attributes
                .get("ocr.granularity")
                .map(String::as_str),
            Some("document")
        );
        Ok(())
    }

    #[test]
    fn scene_aware_ocr_samples_edges_and_midpoints_then_tracks_per_scene() -> Result<()> {
        let scenes = vec![scene(0, 4), scene(5, 8)];
        assert_eq!(
            representative_scene_frames(&scenes)?
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 2, 4, 5, 6, 8]
        );

        let calls = Rc::new(Cell::new(0));
        let analyzer = SceneAwareOcrVideoAnalyzer::new(
            "ocr",
            CountingOcr {
                calls: calls.clone(),
            },
            &scenes,
        )?;
        let mut pipeline = VideoAnalysisPipeline::builder().analyzer(analyzer).build()?;
        let pixels = vec![0_u8; 2 * 2 * 3];
        for frame_index in 0..=8 {
            let frame = VideoFrame::rgb24(position(frame_index), 2, 2, &pixels)?;
            pipeline.process_frame_ref(&frame)?;
        }
        let analysis = pipeline.finish_analysis()?;
        assert_eq!(calls.get(), 6);
        assert_eq!(analysis.observations.len(), 6);

        let semantics = analyze_video_text_semantics(
            VideoTextSemanticContext {
                video_id: "video-1",
                width: 2,
                height: 2,
                duration_seconds: Some(8.0 / 30.0),
                scenes: &scenes,
            },
            &analysis.observations,
        )?;
        assert_eq!(semantics.tracks.len(), 2);
        assert!(semantics
            .tracks
            .iter()
            .all(|track| track.sample_count == 3 && track.scene_indices.len() == 1));
        Ok(())
    }

    #[test]
    fn scene_aware_ocr_rejects_missing_scene_plan() {
        let error = SceneAwareOcrVideoAnalyzer::new("ocr", FixtureOcr, &[]).unwrap_err();
        assert!(error.to_string().contains("at least one scene"));
    }
}
