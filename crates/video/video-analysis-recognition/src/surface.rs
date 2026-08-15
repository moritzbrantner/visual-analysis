//! Library-owned runtime surface for `video-analysis-recognition`.

use std::collections::BTreeMap;

use runtime_core::{
    OperationId, PackageSurface, RuntimeCapabilities, SurfaceOperation, SurfaceRequest,
    SurfaceResponse,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use video_analysis_core::{BoundingBox, ObservationKind};

use crate::{Embedding, MatchOptions, ReferenceIdentity, ReferenceLibrary};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Describe package",
                "Recognition result contracts for labels, tracks, and confidence summaries.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "video.recognition.labelPlan",
                "Plan labels",
                "Builds a reference library and searches candidate embeddings against it.",
                serde_json::json!({
                    "identities": [{"id": "alice", "label": "Alice", "kind": "face", "embeddings": [[1.0, 0.0]]}],
                    "candidates": [{"id": "candidate-1", "kind": "face", "embedding": [1.0, 0.0]}],
                    "options": {"minScore": 0.5, "maxResults": 2}
                }),
            ),
            operation(
                "video.recognition.trackSummary",
                "Summarize recognition tracks",
                "Summarizes recognized labels over tracks and confidence acceptance thresholds.",
                serde_json::json!({
                    "threshold": 0.75,
                    "matches": [{"trackId": "track-1", "label": "Alice", "kind": "face", "score": 0.92}]
                }),
            ),
            operation(
                "video.recognition.confidence",
                "Summarize confidence",
                "Summarizes confidence score ranges and threshold acceptance counts.",
                serde_json::json!({"threshold": 0.75, "scores": [0.9, 0.5, 0.8]}),
            ),
        ],
    }
}

fn operation(
    id: &str,
    name: &str,
    description: &str,
    example_request: serde_json::Value,
) -> SurfaceOperation {
    SurfaceOperation {
        id: OperationId::new(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        curation: runtime_core::SurfaceOperationCuration::from_operation_id(id),
        input_schema: serde_json::json!({"type": "object", "additionalProperties": true, "xOperationCategory": runtime_core::operation_category(id)}),
        output_schema: serde_json::json!({"type": "object", "xOperationCategory": runtime_core::operation_category(id)}),
        example_request,
        wasm_supported: true,
        server_supported: true,
    }
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => describe_value(request.input),
        "video.recognition.labelPlan" => label_plan_value(parse_input(request.input)?)?,
        "video.recognition.trackSummary" => track_summary_value(parse_input(request.input)?)?,
        "video.recognition.confidence" => confidence_value(parse_input(request.input)?)?,
        operation => {
            return Err(format!(
                "unsupported operation `{operation}` for {}",
                env!("CARGO_PKG_NAME")
            ))
        }
    };
    Ok(surface_response(operation, value))
}

fn describe_value(input: serde_json::Value) -> serde_json::Value {
    let surface = package_surface();
    serde_json::json!({
        "library": surface.library,
        "version": surface.version,
        "operationCount": surface.operations.len(),
        "operations": surface.operations.iter().map(|operation| operation.id.as_str()).collect::<Vec<_>>(),
        "input": input
    })
}

fn surface_response(operation: OperationId, value: serde_json::Value) -> SurfaceResponse {
    SurfaceResponse {
        operation,
        value,
        diagnostics: Vec::new(),
        artifacts: Vec::new(),
    }
}

fn parse_input<T: DeserializeOwned>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| format!("invalid request: {error}"))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelPlanRequest {
    #[serde(default)]
    identities: Vec<IdentityRequest>,
    #[serde(default)]
    candidates: Vec<CandidateRequest>,
    #[serde(default)]
    options: MatchOptionsRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IdentityRequest {
    id: String,
    label: String,
    #[serde(default = "default_kind")]
    kind: String,
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CandidateRequest {
    id: Option<String>,
    #[serde(default = "default_kind")]
    kind: String,
    embedding: Vec<f32>,
    region: Option<BoundingBoxRequest>,
    detector_label: Option<String>,
    detection_score: Option<f32>,
    track_id: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
struct BoundingBoxRequest {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MatchOptionsRequest {
    #[serde(default = "default_min_score")]
    min_score: f32,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

impl Default for MatchOptionsRequest {
    fn default() -> Self {
        Self {
            min_score: default_min_score(),
            max_results: default_max_results(),
        }
    }
}

fn default_kind() -> String {
    "object".to_string()
}

fn default_min_score() -> f32 {
    0.75
}

fn default_max_results() -> usize {
    1
}

fn label_plan_value(request: LabelPlanRequest) -> Result<serde_json::Value, String> {
    let mut library = ReferenceLibrary::new();
    for identity in request.identities {
        let mut reference =
            ReferenceIdentity::new(identity.id, identity.label, kind(&identity.kind));
        for (key, value) in identity.attributes {
            reference = reference.attribute(key, value);
        }
        for embedding in identity.embeddings {
            reference
                .add_embedding(embedding)
                .map_err(|error| error.to_string())?;
        }
        library
            .add_identity(reference)
            .map_err(|error| error.to_string())?;
    }

    let options = MatchOptions::new(request.options.min_score)
        .map_err(|error| error.to_string())?
        .max_results(request.options.max_results);
    let mut candidates = Vec::new();
    for (index, candidate) in request.candidates.into_iter().enumerate() {
        if let Some(region) = candidate.region {
            let _ = bbox(region)?;
        }
        if let Some(score) = candidate.detection_score {
            if !score.is_finite() {
                return Err("detectionScore must be finite".to_string());
            }
        }
        let embedding = Embedding::new(candidate.embedding).map_err(|error| error.to_string())?;
        let matches = library
            .search(&embedding, Some(&kind(&candidate.kind)), &options)
            .map_err(|error| error.to_string())?;
        candidates.push(serde_json::json!({
            "id": candidate.id.unwrap_or_else(|| format!("candidate-{index}")),
            "kind": candidate.kind,
            "detectorLabel": candidate.detector_label,
            "detectionScore": candidate.detection_score,
            "trackId": candidate.track_id,
            "matchCount": matches.len(),
            "matches": matches.into_iter().map(match_json).collect::<Vec<_>>()
        }));
    }

    Ok(serde_json::json!({
        "referenceCount": library.len(),
        "dimensions": library.dimensions(),
        "candidateCount": candidates.len(),
        "candidates": candidates,
        "options": {"minScore": options.min_score, "maxResults": options.max_results},
        "externalToolsRequired": false
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrackSummaryRequest {
    #[serde(default = "default_min_score")]
    threshold: f32,
    #[serde(default)]
    matches: Vec<MatchedCandidateRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MatchedCandidateRequest {
    track_id: Option<String>,
    label: String,
    #[serde(default = "default_kind")]
    kind: String,
    score: f32,
}

fn track_summary_value(request: TrackSummaryRequest) -> Result<serde_json::Value, String> {
    validate_threshold(request.threshold)?;
    let mut by_track = BTreeMap::<String, usize>::new();
    let mut by_label = BTreeMap::<String, usize>::new();
    let mut by_kind = BTreeMap::<String, usize>::new();
    let mut accepted = 0usize;
    let mut rejected = 0usize;

    for (index, matched) in request.matches.into_iter().enumerate() {
        if !matched.score.is_finite() {
            return Err("match score must be finite".to_string());
        }
        let track = matched
            .track_id
            .unwrap_or_else(|| format!("untracked-{index}"));
        *by_track.entry(track).or_default() += 1;
        *by_label.entry(matched.label).or_default() += 1;
        *by_kind.entry(matched.kind).or_default() += 1;
        if matched.score >= request.threshold {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    Ok(serde_json::json!({
        "threshold": request.threshold,
        "matchCount": accepted + rejected,
        "acceptedCount": accepted,
        "rejectedCount": rejected,
        "tracks": by_track,
        "labels": by_label,
        "kinds": by_kind
    }))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConfidenceRequest {
    #[serde(default = "default_min_score")]
    threshold: f32,
    #[serde(default)]
    scores: Vec<f32>,
}

fn confidence_value(request: ConfidenceRequest) -> Result<serde_json::Value, String> {
    validate_threshold(request.threshold)?;
    if request.scores.iter().any(|score| !score.is_finite()) {
        return Err("scores must be finite".to_string());
    }
    let accepted = request
        .scores
        .iter()
        .filter(|score| **score >= request.threshold)
        .count();
    let rejected = request.scores.len().saturating_sub(accepted);
    let min = request.scores.iter().copied().reduce(f32::min);
    let max = request.scores.iter().copied().reduce(f32::max);
    let mean = if request.scores.is_empty() {
        None
    } else {
        Some(request.scores.iter().sum::<f32>() / request.scores.len() as f32)
    };

    Ok(serde_json::json!({
        "threshold": request.threshold,
        "scoreCount": request.scores.len(),
        "acceptedCount": accepted,
        "rejectedCount": rejected,
        "min": min,
        "max": max,
        "mean": mean
    }))
}

fn match_json(value: crate::RecognitionMatch) -> serde_json::Value {
    serde_json::json!({
        "referenceId": value.reference_id,
        "label": value.label,
        "kind": kind_name(&value.kind),
        "score": value.score,
        "attributes": value.attributes
    })
}

fn validate_threshold(value: f32) -> Result<(), String> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err("threshold must be finite and between -1.0 and 1.0".to_string());
    }
    Ok(())
}

fn bbox(request: BoundingBoxRequest) -> Result<BoundingBox, String> {
    BoundingBox::new(request.x, request.y, request.width, request.height)
        .map_err(|error| error.to_string())
}

fn kind(value: &str) -> ObservationKind {
    match value {
        "text" => ObservationKind::Text,
        "face" => ObservationKind::Face,
        "scene" => ObservationKind::Scene,
        "object" => ObservationKind::Object,
        other => ObservationKind::Custom(other.to_string()),
    }
}

fn kind_name(value: &ObservationKind) -> String {
    match value {
        ObservationKind::Text => "text".to_string(),
        ObservationKind::Face => "face".to_string(),
        ObservationKind::Object => "object".to_string(),
        ObservationKind::Scene => "scene".to_string(),
        ObservationKind::Custom(value) => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_plan_searches_reference_library() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.recognition.labelPlan"),
            input: serde_json::json!({
                "identities": [{"id": "alice", "label": "Alice", "kind": "face", "embeddings": [[1.0, 0.0]]}],
                "candidates": [{"kind": "face", "embedding": [1.0, 0.0]}],
                "options": {"minScore": 0.5}
            }),
        })
        .expect("label plan");
        assert_eq!(response.value["referenceCount"], 1);
        assert_eq!(response.value["candidates"][0]["matchCount"], 1);
        assert!(response.value.get("plan").is_none());
    }

    #[test]
    fn confidence_rejects_non_finite_scores() {
        let error = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.recognition.confidence"),
            input: serde_json::json!({"scores": [serde_json::Value::String("nan".to_string())]}),
        })
        .expect_err("invalid score");
        assert!(error.contains("invalid request"));
    }

    #[test]
    fn track_summary_counts_accepted_matches() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("video.recognition.trackSummary"),
            input: serde_json::json!({
                "threshold": 0.75,
                "matches": [
                    {"trackId": "track-1", "label": "Alice", "kind": "face", "score": 0.9},
                    {"trackId": "track-1", "label": "Alice", "kind": "face", "score": 0.5}
                ]
            }),
        })
        .expect("track summary");
        assert_eq!(response.value["acceptedCount"], 1);
        assert_eq!(response.value["rejectedCount"], 1);
    }
}
