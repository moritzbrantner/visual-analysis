//! Library-owned runtime surface for `vision-core`.

use runtime_core::{
    describe_surface_response, structured_surface_response, surface_operation, PackageSurface,
    RuntimeCapabilities, SurfaceOperation, SurfaceRequest, SurfaceResponse,
};
use serde::Deserialize;

use crate::{IdentityMatch, VisualDetection, VisualEmbedding};

/// Returns the package surface exposed by every transport wrapper.
pub fn package_surface() -> PackageSurface {
    PackageSurface {
        library: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        capabilities: RuntimeCapabilities::pure_rust(),
        operations: vec![
            operation(
                "describe",
                "Inspect package metadata",
                "Shared visual detection, keypoint, embedding, and identity-match contracts.",
                serde_json::json!({"includeOperations": true}),
            ),
            operation(
                "vision.validateDetection",
                "Validate visual detection",
                "Validates one visual detection contract.",
                serde_json::json!({
                    "detection": {
                        "id": "face-1",
                        "kind": "face",
                        "label": "face",
                        "score": 0.92,
                        "region": {"x": 1, "y": 2, "width": 16, "height": 20},
                        "keypoints": [],
                        "attributes": {}
                    }
                }),
            ),
            operation(
                "vision.validateEmbedding",
                "Validate visual embedding",
                "Validates one source-aware visual embedding contract.",
                serde_json::json!({
                    "embedding": {
                        "id": "embedding-1",
                        "sourceDetectionId": "face-1",
                        "kind": "face",
                        "region": {"x": 1, "y": 2, "width": 16, "height": 20},
                        "vector": [1.0, 0.0],
                        "attributes": {}
                    }
                }),
            ),
            operation(
                "vision.validateIdentityMatch",
                "Validate identity match",
                "Validates one visual identity match summary.",
                serde_json::json!({
                    "identityMatch": {
                        "sourceDetectionId": "face-1",
                        "sourceEmbeddingId": "embedding-1",
                        "referenceId": "alice",
                        "label": "Alice",
                        "score": 0.98,
                        "kind": "face",
                        "attributes": {}
                    }
                }),
            ),
            operation(
                "vision.convertDetectionSummary",
                "Summarize visual detections",
                "Returns a deterministic summary of visual detection contracts.",
                serde_json::json!({
                    "detections": [
                        {
                            "id": "face-1",
                            "kind": "face",
                            "label": "face",
                            "score": 0.92,
                            "region": {"x": 1, "y": 2, "width": 16, "height": 20},
                            "keypoints": [],
                            "attributes": {}
                        }
                    ]
                }),
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
    let mut operation = surface_operation(id, name, description, example_request);
    let contract = match id {
        "vision.validateDetection" => {
            Some(runtime_core::landscape::LandscapeOperationContract::new(
                runtime_core::landscape::LandscapeFunction::new(
                    "vision.validateDetection",
                    env!("CARGO_PKG_NAME"),
                )
                .input(runtime_core::landscape::LandscapePort::new(
                    "detection",
                    runtime_core::landscape::well_known::vision_detection(),
                ))
                .output(runtime_core::landscape::LandscapePort::new(
                    "detection",
                    runtime_core::landscape::well_known::vision_detection(),
                )),
            ))
        }
        "vision.validateEmbedding" => {
            Some(runtime_core::landscape::LandscapeOperationContract::new(
                runtime_core::landscape::LandscapeFunction::new(
                    "vision.validateEmbedding",
                    env!("CARGO_PKG_NAME"),
                )
                .input(runtime_core::landscape::LandscapePort::new(
                    "embedding",
                    runtime_core::landscape::well_known::vision_embedding(),
                ))
                .output(runtime_core::landscape::LandscapePort::new(
                    "embedding",
                    runtime_core::landscape::well_known::vision_embedding(),
                )),
            ))
        }
        _ => None,
    };
    if let Some(contract) = contract {
        runtime_core::attach_landscape_contract(&mut operation, contract);
    }
    operation
}

/// Runs one library-owned operation.
pub fn run_surface_operation(request: SurfaceRequest) -> Result<SurfaceResponse, String> {
    let operation = request.operation.clone();
    let value = match request.operation.as_str() {
        "describe" => return Ok(describe_surface_response(&package_surface(), request)),
        "vision.validateDetection" => validate_detection_value(parse_input(request.input)?)?,
        "vision.validateEmbedding" => validate_embedding_value(parse_input(request.input)?)?,
        "vision.validateIdentityMatch" => {
            validate_identity_match_value(parse_input(request.input)?)?
        }
        "vision.convertDetectionSummary" => detection_summary_value(parse_input(request.input)?)?,
        operation => {
            return Err(runtime_core::SurfaceError::unsupported_operation(
                operation,
                env!("CARGO_PKG_NAME"),
            )
            .to_error_string());
        }
    };
    Ok(structured_surface_response(
        operation.clone(),
        workflow_title(operation.as_str()),
        workflow_message(operation.as_str()),
        workflow_summary(operation.as_str(), &value),
        value,
    ))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DetectionRequest {
    detection: VisualDetection,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddingRequest {
    embedding: VisualEmbedding,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IdentityMatchRequest {
    identity_match: IdentityMatch,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DetectionSummaryRequest {
    detections: Vec<VisualDetection>,
}

fn validate_detection_value(request: DetectionRequest) -> Result<serde_json::Value, String> {
    request
        .detection
        .validate()
        .map_err(|error| validation_error("vision.validateDetection", error))?;
    Ok(serde_json::json!({
        "valid": true,
        "detection": request.detection,
    }))
}

fn validate_embedding_value(request: EmbeddingRequest) -> Result<serde_json::Value, String> {
    request
        .embedding
        .validate()
        .map_err(|error| validation_error("vision.validateEmbedding", error))?;
    Ok(serde_json::json!({
        "valid": true,
        "embedding": request.embedding,
    }))
}

fn validate_identity_match_value(
    request: IdentityMatchRequest,
) -> Result<serde_json::Value, String> {
    request
        .identity_match
        .validate()
        .map_err(|error| validation_error("vision.validateIdentityMatch", error))?;
    Ok(serde_json::json!({
        "valid": true,
        "identityMatch": request.identity_match,
    }))
}

fn detection_summary_value(request: DetectionSummaryRequest) -> Result<serde_json::Value, String> {
    for detection in &request.detections {
        detection
            .validate()
            .map_err(|error| validation_error("vision.convertDetectionSummary", error))?;
    }
    let mut by_kind = std::collections::BTreeMap::<String, usize>::new();
    let mut with_ids = 0_usize;
    let mut keypoints = 0_usize;
    for detection in &request.detections {
        *by_kind
            .entry(kind_label(&detection.kind).to_string())
            .or_default() += 1;
        if detection.id.is_some() {
            with_ids += 1;
        }
        keypoints += detection.keypoints.len();
    }
    Ok(serde_json::json!({
        "detectionCount": request.detections.len(),
        "withIds": with_ids,
        "keypointCount": keypoints,
        "byKind": by_kind,
        "detections": request.detections,
    }))
}

fn kind_label(kind: &crate::VisualDetectionKind) -> &str {
    match kind {
        crate::VisualDetectionKind::Face => "face",
        crate::VisualDetectionKind::Person => "person",
        crate::VisualDetectionKind::Object => "object",
        crate::VisualDetectionKind::TextRegion => "textRegion",
        crate::VisualDetectionKind::Custom(value) => value.as_str(),
    }
}

fn validation_error(operation: &str, error: impl std::fmt::Display) -> String {
    runtime_core::SurfaceError::invalid_request(
        Some(runtime_core::OperationId::new(operation)),
        format!("invalid visual contract: {error}"),
    )
    .to_error_string()
}

fn workflow_title(operation: &str) -> &'static str {
    match operation {
        "vision.validateDetection" => "Validated visual detection",
        "vision.validateEmbedding" => "Validated visual embedding",
        "vision.validateIdentityMatch" => "Validated identity match",
        "vision.convertDetectionSummary" => "Visual detection summary",
        _ => "Vision core result",
    }
}

fn workflow_message(operation: &str) -> &'static str {
    match operation {
        "vision.validateDetection" => "Validated a localized visual detection contract.",
        "vision.validateEmbedding" => "Validated a source-aware visual embedding contract.",
        "vision.validateIdentityMatch" => "Validated a scored identity match summary.",
        "vision.convertDetectionSummary" => {
            "Summarized visual detections by kind and source metadata."
        }
        _ => "Ran a vision-core package operation.",
    }
}

fn workflow_summary(operation: &str, value: &serde_json::Value) -> serde_json::Value {
    match operation {
        "vision.convertDetectionSummary" => serde_json::json!({
            "status": "ok",
            "detectionCount": value["detectionCount"],
            "withIds": value["withIds"],
            "keypointCount": value["keypointCount"],
        }),
        _ => serde_json::json!({"status": "ok", "valid": value["valid"]}),
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(input: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(input).map_err(|error| {
        runtime_core::SurfaceError::invalid_request(
            None::<runtime_core::OperationId>,
            format!("invalid request: {error}"),
        )
        .to_error_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_core::{OperationId, SurfaceRequest};

    #[test]
    fn validates_detection_operation() {
        let response = run_surface_operation(SurfaceRequest {
            operation: OperationId::new("vision.validateDetection"),
            input: serde_json::json!({
                "detection": {
                    "id": "face-1",
                    "kind": "face",
                    "label": "face",
                    "score": 0.92,
                    "region": {"x": 1, "y": 2, "width": 16, "height": 20},
                    "keypoints": [],
                    "attributes": {}
                }
            }),
        })
        .unwrap();
        assert_eq!(response.value["result"]["valid"], true);
    }
}
