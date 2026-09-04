use std::collections::{BTreeMap, BTreeSet};

use semantic_core::{
    Annotation, ConceptId, EntityId, Evidence, EvidenceKey, EvidenceValue, FiniteNumber, Producer,
    ProducerId, Relation, SemanticContractError, SemanticNodeRef,
};
use serde::{Deserialize, Serialize};
use video_analysis_core::{BoundingBox, DetectError, Observation, ObservationKind, Result, Scene};

const CLASSIFIER_VERSION: &str = "1";
const MAX_FALLBACK_GAP_SECONDS: f64 = 8.0;
const MAX_FALLBACK_GAP_FRAMES: u64 = 240;
const MIN_FALLBACK_IOU: f64 = 0.30;
const MAX_CENTER_DELTA_RATIO: f64 = 0.08;

/// Conservative semantic roles for OCR text observed across video frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoTextRole {
    /// Short-lived bottom-centered text consistent with subtitles.
    Subtitle,
    /// Dense text in the closing portion of a video.
    EndCredit,
    /// Stable multi-region text consistent with a presentation slide.
    PresentationSlide,
    /// Text belonging to or embedded in the filmed scene.
    SceneText,
}

/// One temporal text occurrence assembled from frame-level OCR observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTextTrack {
    /// Stable semantic entity identity for this text occurrence.
    pub id: EntityId,
    /// Deterministic consensus text for the track.
    pub text: String,
    /// First conservative semantic role assigned by the heuristic classifier.
    pub role: VideoTextRole,
    /// Number of OCR observations associated with this track.
    pub sample_count: u32,
    /// Earliest observed frame index.
    pub start_frame: Option<u64>,
    /// Latest observed frame index.
    pub end_frame: Option<u64>,
    /// Earliest observed timestamp in seconds.
    pub start_seconds: Option<f64>,
    /// Latest observed timestamp in seconds.
    pub end_seconds: Option<f64>,
    /// Sorted unique scene indices touched by this track.
    pub scene_indices: Vec<u64>,
    /// Mean OCR bounding box across observations that supplied geometry.
    pub representative_region: Option<BoundingBox>,
}

/// Shared semantic claims and relations produced for video text tracks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoTextSemanticAnalysis {
    /// Deterministically ordered temporal text tracks.
    pub tracks: Vec<VideoTextTrack>,
    /// Generic video-text plus role annotations for each track.
    pub annotations: Vec<Annotation>,
    /// Explicit track-to-scene relations.
    pub relations: Vec<Relation>,
}

/// Context required to classify OCR text in a video without owning scene detection.
#[derive(Debug, Clone, Copy)]
pub struct VideoTextSemanticContext<'a> {
    /// Stable consumer-provided video identity.
    pub video_id: &'a str,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Known video duration when available.
    pub duration_seconds: Option<f64>,
    /// Existing scene boundaries, normally produced through the canonical detector seam.
    pub scenes: &'a [Scene],
}

#[derive(Debug, Clone)]
struct TextSample {
    text: String,
    normalized_text: String,
    frame: Option<u64>,
    seconds: Option<f64>,
    scene_index: Option<u64>,
    region: Option<BoundingBox>,
    score: Option<f32>,
    upstream_track_id: Option<String>,
}

#[derive(Debug, Clone)]
struct TrackBuilder {
    samples: Vec<TextSample>,
    upstream_track_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SceneTextStats {
    track_count: usize,
    stable_track_count: usize,
    upper_or_middle_track_count: usize,
}

#[derive(Debug, Clone)]
struct TrackSummary {
    id: EntityId,
    text: String,
    sample_count: u32,
    start_frame: Option<u64>,
    end_frame: Option<u64>,
    start_seconds: Option<f64>,
    end_seconds: Option<f64>,
    scene_indices: Vec<u64>,
    representative_region: Option<BoundingBox>,
    average_ocr_score: Option<f64>,
    upstream_track_id: Option<String>,
}

/// Builds temporal OCR tracks and projects conservative video-text semantics into
/// the shared Foundation annotation contract.
///
/// Source observations are never modified. Semantic confidence intentionally
/// remains absent: OCR/model scores are source evidence, not calibrated role
/// probabilities.
pub fn analyze_video_text_semantics(
    context: VideoTextSemanticContext<'_>,
    observations: &[Observation],
) -> Result<VideoTextSemanticAnalysis> {
    if context.video_id.trim().is_empty() {
        return Err(DetectError::InvalidArgument(
            "video text semantic context requires a non-empty video_id".to_string(),
        ));
    }
    if context.width == 0 || context.height == 0 {
        return Err(DetectError::InvalidArgument(
            "video text semantic context requires non-zero dimensions".to_string(),
        ));
    }
    if context
        .duration_seconds
        .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
    {
        return Err(DetectError::InvalidArgument(
            "video duration must be finite and non-negative".to_string(),
        ));
    }

    let samples = collect_text_samples(context, observations);
    let builders = associate_samples(context, samples);
    let mut summaries = builders
        .into_iter()
        .map(|builder| summarize_track(context, builder))
        .collect::<Result<Vec<_>>>()?;
    summaries.sort_by(|left, right| left.id.as_str().cmp(right.id.as_str()));

    let scene_stats = scene_text_stats(context, &summaries);
    let duration = effective_video_duration(context, &summaries);
    let producer = semantic_producer()?;
    let mut tracks = Vec::with_capacity(summaries.len());
    let mut annotations = Vec::with_capacity(summaries.len().saturating_mul(2));
    let mut relations = Vec::new();

    for summary in summaries {
        let role = classify_track(context, &summary, &scene_stats, duration);
        let evidence = semantic_evidence(context, &summary, &scene_stats, duration)?;
        annotations.push(Annotation {
            subject: summary.id.clone(),
            concept: concept("visual-analysis:video-text")?,
            confidence: None,
            producer: Some(producer.clone()),
            evidence: evidence.clone(),
        });
        annotations.push(Annotation {
            subject: summary.id.clone(),
            concept: video_text_role_concept(role)?,
            confidence: None,
            producer: Some(producer.clone()),
            evidence,
        });

        for scene_index in &summary.scene_indices {
            relations.push(Relation {
                subject: SemanticNodeRef::Entity(summary.id.clone()),
                predicate: concept("visual-analysis:relation:appears-in-scene")?,
                object: SemanticNodeRef::Entity(scene_entity_id(context, *scene_index)),
                confidence: None,
                producer: Some(producer.clone()),
                evidence: vec![integer_evidence(
                    "visual-analysis:video-text-evidence:scene-index",
                    i64_from_u64(*scene_index),
                )?],
            });
        }

        tracks.push(VideoTextTrack {
            id: summary.id,
            text: summary.text,
            role,
            sample_count: summary.sample_count,
            start_frame: summary.start_frame,
            end_frame: summary.end_frame,
            start_seconds: summary.start_seconds,
            end_seconds: summary.end_seconds,
            scene_indices: summary.scene_indices,
            representative_region: summary.representative_region,
        });
    }

    relations.sort_by(|left, right| relation_sort_key(left).cmp(&relation_sort_key(right)));
    Ok(VideoTextSemanticAnalysis {
        tracks,
        annotations,
        relations,
    })
}

/// Returns the visual-analysis-owned concept identifier for a video text role.
pub fn video_text_role_concept(role: VideoTextRole) -> Result<ConceptId> {
    concept(match role {
        VideoTextRole::Subtitle => "visual-analysis:video-text-role:subtitle",
        VideoTextRole::EndCredit => "visual-analysis:video-text-role:end-credit",
        VideoTextRole::PresentationSlide => {
            "visual-analysis:video-text-role:presentation-slide"
        }
        VideoTextRole::SceneText => "visual-analysis:video-text-role:scene-text",
    })
}

fn collect_text_samples(
    context: VideoTextSemanticContext<'_>,
    observations: &[Observation],
) -> Vec<TextSample> {
    let mut samples = observations
        .iter()
        .filter_map(|observation| {
            let text = observation.text.as_deref()?.trim();
            if text.is_empty() || !is_visual_text_observation(observation) {
                return None;
            }
            let normalized_text = normalize_text(text);
            if normalized_text.is_empty() {
                return None;
            }
            let frame = observation.frame.map(|position| position.frame_index);
            let seconds = observation
                .timestamp
                .or_else(|| observation.frame.map(|position| position.timestamp))
                .map(|timestamp| timestamp.seconds())
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0);
            let scene_index = observation
                .scene_index
                .or_else(|| frame.and_then(|frame| infer_scene_index(context.scenes, frame)));
            Some(TextSample {
                text: text.to_string(),
                normalized_text,
                frame,
                seconds,
                scene_index,
                region: observation.region,
                score: observation.score.filter(|score| score.is_finite()),
                upstream_track_id: observation
                    .track_id
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string),
            })
        })
        .collect::<Vec<_>>();

    samples.sort_by(|left, right| sample_sort_key(left).cmp(&sample_sort_key(right)));
    samples
}

fn is_visual_text_observation(observation: &Observation) -> bool {
    matches!(observation.kind, ObservationKind::Text)
        || observation.analyzer.to_ascii_lowercase().contains("ocr")
}

fn sample_sort_key(
    sample: &TextSample,
) -> (
    Option<u64>,
    Option<i64>,
    Option<u64>,
    String,
    Option<(u32, u32, u32, u32)>,
    Option<String>,
) {
    (
        sample.frame,
        sample.seconds.map(seconds_sort_key),
        sample.scene_index,
        sample.normalized_text.clone(),
        sample
            .region
            .map(|region| (region.x, region.y, region.width, region.height)),
        sample.upstream_track_id.clone(),
    )
}

fn seconds_sort_key(seconds: f64) -> i64 {
    (seconds * 1_000_000.0).round() as i64
}

fn associate_samples(
    context: VideoTextSemanticContext<'_>,
    samples: Vec<TextSample>,
) -> Vec<TrackBuilder> {
    let mut upstream = BTreeMap::<String, TrackBuilder>::new();
    let mut fallback = Vec::<TrackBuilder>::new();

    for sample in samples {
        if let Some(track_id) = sample.upstream_track_id.clone() {
            upstream
                .entry(track_id.clone())
                .or_insert_with(|| TrackBuilder {
                    samples: Vec::new(),
                    upstream_track_id: Some(track_id),
                })
                .samples
                .push(sample);
            continue;
        }

        let candidate = fallback
            .iter()
            .enumerate()
            .filter_map(|(index, track)| {
                fallback_match_score(context, track, &sample).map(|score| (index, score))
            })
            .max_by(|(left_index, left_score), (right_index, right_score)| {
                left_score
                    .total_cmp(right_score)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index);

        if let Some(index) = candidate {
            fallback[index].samples.push(sample);
        } else {
            fallback.push(TrackBuilder {
                samples: vec![sample],
                upstream_track_id: None,
            });
        }
    }

    let mut result = upstream.into_values().collect::<Vec<_>>();
    result.extend(fallback);
    result
}

fn fallback_match_score(
    context: VideoTextSemanticContext<'_>,
    track: &TrackBuilder,
    sample: &TextSample,
) -> Option<f64> {
    let last = track.samples.last()?;
    if last.normalized_text != sample.normalized_text {
        return None;
    }
    if last.scene_index.is_some()
        && sample.scene_index.is_some()
        && last.scene_index != sample.scene_index
    {
        return None;
    }
    if !within_temporal_gap(last, sample) {
        return None;
    }
    spatial_match_score(context, last.region, sample.region)
}

fn within_temporal_gap(left: &TextSample, right: &TextSample) -> bool {
    if let (Some(left_seconds), Some(right_seconds)) = (left.seconds, right.seconds) {
        return right_seconds >= left_seconds
            && right_seconds - left_seconds <= MAX_FALLBACK_GAP_SECONDS;
    }
    if let (Some(left_frame), Some(right_frame)) = (left.frame, right.frame) {
        return right_frame >= left_frame
            && right_frame - left_frame <= MAX_FALLBACK_GAP_FRAMES;
    }
    true
}

fn spatial_match_score(
    context: VideoTextSemanticContext<'_>,
    left: Option<BoundingBox>,
    right: Option<BoundingBox>,
) -> Option<f64> {
    let (Some(left), Some(right)) = (left, right) else {
        return Some(0.0);
    };
    let iou = bounding_box_iou(left, right);
    if iou >= MIN_FALLBACK_IOU {
        return Some(1.0 + iou);
    }
    let (left_x, left_y) = box_center(left);
    let (right_x, right_y) = box_center(right);
    let dx = (left_x - right_x).abs() / f64::from(context.width);
    let dy = (left_y - right_y).abs() / f64::from(context.height);
    if dx <= MAX_CENTER_DELTA_RATIO && dy <= MAX_CENTER_DELTA_RATIO {
        return Some(1.0 - (dx + dy) / 2.0);
    }
    None
}

fn summarize_track(
    context: VideoTextSemanticContext<'_>,
    mut builder: TrackBuilder,
) -> Result<TrackSummary> {
    builder
        .samples
        .sort_by(|left, right| sample_sort_key(left).cmp(&sample_sort_key(right)));
    let text = consensus_text(&builder.samples);
    let start_frame = builder.samples.iter().filter_map(|sample| sample.frame).min();
    let end_frame = builder.samples.iter().filter_map(|sample| sample.frame).max();
    let start_seconds = builder
        .samples
        .iter()
        .filter_map(|sample| sample.seconds)
        .min_by(f64::total_cmp);
    let end_seconds = builder
        .samples
        .iter()
        .filter_map(|sample| sample.seconds)
        .max_by(f64::total_cmp);
    let scene_indices = builder
        .samples
        .iter()
        .filter_map(|sample| sample.scene_index)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let representative_region = mean_region(&builder.samples);
    let scores = builder
        .samples
        .iter()
        .filter_map(|sample| sample.score)
        .map(f64::from)
        .collect::<Vec<_>>();
    let average_ocr_score = (!scores.is_empty())
        .then(|| scores.iter().sum::<f64>() / scores.len() as f64);
    let first = builder.samples.first().ok_or_else(|| {
        DetectError::InvalidArgument("video text track cannot be empty".to_string())
    })?;
    let id = if let Some(upstream_id) = builder.upstream_track_id.as_deref() {
        EntityId::derive(&[
            "visual-analysis",
            "video-text-track-v1",
            context.video_id,
            "upstream",
            upstream_id,
        ])
    } else {
        let first_frame = first
            .frame
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        let first_seconds = first
            .seconds
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "none".to_string());
        let first_scene = first
            .scene_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        let region_signature = first
            .region
            .map(|region| format!("{}:{}:{}:{}", region.x, region.y, region.width, region.height))
            .unwrap_or_else(|| "none".to_string());
        EntityId::derive(&[
            "visual-analysis",
            "video-text-track-v1",
            context.video_id,
            "fallback",
            &first.normalized_text,
            &first_frame,
            &first_seconds,
            &first_scene,
            &region_signature,
        ])
    };

    Ok(TrackSummary {
        id,
        text,
        sample_count: builder.samples.len().min(u32::MAX as usize) as u32,
        start_frame,
        end_frame,
        start_seconds,
        end_seconds,
        scene_indices,
        representative_region,
        average_ocr_score,
        upstream_track_id: builder.upstream_track_id,
    })
}

fn consensus_text(samples: &[TextSample]) -> String {
    let mut counts = BTreeMap::<String, (usize, String)>::new();
    for sample in samples {
        let entry = counts
            .entry(sample.normalized_text.clone())
            .or_insert_with(|| (0, sample.text.clone()));
        entry.0 += 1;
        if sample.text < entry.1 {
            entry.1 = sample.text.clone();
        }
    }
    counts
        .into_iter()
        .max_by(|(left_key, (left_count, _)), (right_key, (right_count, _))| {
            left_count
                .cmp(right_count)
                .then_with(|| right_key.cmp(left_key))
        })
        .map(|(_, (_, text))| text)
        .unwrap_or_default()
}

fn mean_region(samples: &[TextSample]) -> Option<BoundingBox> {
    let regions = samples
        .iter()
        .filter_map(|sample| sample.region)
        .collect::<Vec<_>>();
    if regions.is_empty() {
        return None;
    }
    let count = regions.len() as u64;
    Some(BoundingBox {
        x: (regions.iter().map(|region| u64::from(region.x)).sum::<u64>() / count) as u32,
        y: (regions.iter().map(|region| u64::from(region.y)).sum::<u64>() / count) as u32,
        width: (regions
            .iter()
            .map(|region| u64::from(region.width))
            .sum::<u64>()
            / count) as u32,
        height: (regions
            .iter()
            .map(|region| u64::from(region.height))
            .sum::<u64>()
            / count) as u32,
    })
}

fn scene_text_stats(
    context: VideoTextSemanticContext<'_>,
    summaries: &[TrackSummary],
) -> BTreeMap<u64, SceneTextStats> {
    let mut stats = BTreeMap::<u64, SceneTextStats>::new();
    for summary in summaries {
        let stable = is_temporally_stable(summary);
        let upper_or_middle = summary
            .representative_region
            .map(|region| box_center(region).1 / f64::from(context.height) < 0.72)
            .unwrap_or(false);
        for scene_index in &summary.scene_indices {
            let entry = stats.entry(*scene_index).or_default();
            entry.track_count += 1;
            if stable {
                entry.stable_track_count += 1;
            }
            if upper_or_middle {
                entry.upper_or_middle_track_count += 1;
            }
        }
    }
    stats
}

fn classify_track(
    context: VideoTextSemanticContext<'_>,
    summary: &TrackSummary,
    scene_stats: &BTreeMap<u64, SceneTextStats>,
    duration: Option<f64>,
) -> VideoTextRole {
    let stats = summary
        .scene_indices
        .iter()
        .filter_map(|scene_index| scene_stats.get(scene_index))
        .copied()
        .max_by_key(|stats| (stats.track_count, stats.stable_track_count))
        .unwrap_or_default();
    let late = is_late_video(summary, duration);
    let center = summary.representative_region.map(box_center);
    let horizontally_centered = center
        .map(|(x, _)| {
            let ratio = x / f64::from(context.width);
            (0.10..=0.90).contains(&ratio)
        })
        .unwrap_or(true);

    if late && stats.track_count >= 4 && horizontally_centered {
        return VideoTextRole::EndCredit;
    }

    if is_subtitle_like(context, summary) {
        return VideoTextRole::Subtitle;
    }

    if stats.track_count >= 3
        && stats.stable_track_count >= 2
        && stats.upper_or_middle_track_count >= 2
        && !late
    {
        return VideoTextRole::PresentationSlide;
    }

    VideoTextRole::SceneText
}

fn is_subtitle_like(context: VideoTextSemanticContext<'_>, summary: &TrackSummary) -> bool {
    if summary.sample_count < 2 {
        return false;
    }
    let Some(region) = summary.representative_region else {
        return false;
    };
    let (center_x, center_y) = box_center(region);
    let x_ratio = center_x / f64::from(context.width);
    let y_ratio = center_y / f64::from(context.height);
    if !(0.20..=0.80).contains(&x_ratio) || y_ratio < 0.72 {
        return false;
    }
    if summary.text.len() > 180 || summary.text.split_whitespace().count() > 24 {
        return false;
    }
    match (summary.start_seconds, summary.end_seconds) {
        (Some(start), Some(end)) => end >= start && end - start <= 8.0,
        _ => true,
    }
}

fn is_temporally_stable(summary: &TrackSummary) -> bool {
    if summary.sample_count < 2 {
        return false;
    }
    match (summary.start_seconds, summary.end_seconds) {
        (Some(start), Some(end)) => end > start,
        _ => match (summary.start_frame, summary.end_frame) {
            (Some(start), Some(end)) => end > start,
            _ => true,
        },
    }
}

fn is_late_video(summary: &TrackSummary, duration: Option<f64>) -> bool {
    let (Some(start), Some(duration)) = (summary.start_seconds, duration) else {
        return false;
    };
    duration > 0.0 && start >= duration * 0.80
}

fn effective_video_duration(
    context: VideoTextSemanticContext<'_>,
    summaries: &[TrackSummary],
) -> Option<f64> {
    context
        .duration_seconds
        .or_else(|| {
            context
                .scenes
                .iter()
                .map(|scene| scene.end.timestamp.seconds())
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                .max_by(f64::total_cmp)
        })
        .or_else(|| {
            summaries
                .iter()
                .filter_map(|summary| summary.end_seconds)
                .max_by(f64::total_cmp)
        })
}

fn semantic_evidence(
    context: VideoTextSemanticContext<'_>,
    summary: &TrackSummary,
    scene_stats: &BTreeMap<u64, SceneTextStats>,
    duration: Option<f64>,
) -> Result<Vec<Evidence>> {
    let mut evidence = vec![
        integer_evidence(
            "visual-analysis:video-text-evidence:sample-count",
            i64::from(summary.sample_count),
        )?,
        integer_evidence(
            "visual-analysis:video-text-evidence:scene-count",
            summary.scene_indices.len().min(i64::MAX as usize) as i64,
        )?,
        boolean_evidence(
            "visual-analysis:video-text-evidence:temporally-stable",
            is_temporally_stable(summary),
        )?,
        boolean_evidence(
            "visual-analysis:video-text-evidence:late-video",
            is_late_video(summary, duration),
        )?,
    ];

    if let Some(region) = summary.representative_region {
        let (center_x, center_y) = box_center(region);
        evidence.push(number_evidence(
            "visual-analysis:video-text-evidence:center-x-ratio",
            center_x / f64::from(context.width),
        )?);
        evidence.push(number_evidence(
            "visual-analysis:video-text-evidence:center-y-ratio",
            center_y / f64::from(context.height),
        )?);
        evidence.push(boolean_evidence(
            "visual-analysis:video-text-evidence:bottom-band",
            center_y / f64::from(context.height) >= 0.72,
        )?);
    }

    if let Some(score) = summary.average_ocr_score {
        evidence.push(number_evidence(
            "visual-analysis:video-text-evidence:average-ocr-score",
            score,
        )?);
    }

    let strongest_stats = summary
        .scene_indices
        .iter()
        .filter_map(|scene_index| scene_stats.get(scene_index))
        .copied()
        .max_by_key(|stats| (stats.track_count, stats.stable_track_count))
        .unwrap_or_default();
    evidence.push(integer_evidence(
        "visual-analysis:video-text-evidence:scene-text-track-count",
        strongest_stats.track_count.min(i64::MAX as usize) as i64,
    )?);
    evidence.push(integer_evidence(
        "visual-analysis:video-text-evidence:scene-stable-track-count",
        strongest_stats
            .stable_track_count
            .min(i64::MAX as usize) as i64,
    )?);

    if let Some(track_id) = &summary.upstream_track_id {
        evidence.push(text_evidence(
            "visual-analysis:video-text-evidence:upstream-track-id",
            track_id.clone(),
        )?);
    }

    Ok(evidence)
}

fn infer_scene_index(scenes: &[Scene], frame: u64) -> Option<u64> {
    scenes
        .iter()
        .position(|scene| {
            frame >= scene.start.frame_index && frame <= scene.end.frame_index
        })
        .map(|index| index as u64)
}

fn scene_entity_id(context: VideoTextSemanticContext<'_>, scene_index: u64) -> EntityId {
    let index = scene_index.to_string();
    if let Some(scene) = context.scenes.get(scene_index as usize) {
        let start = scene.start.frame_index.to_string();
        let end = scene.end.frame_index.to_string();
        EntityId::derive(&[
            "visual-analysis",
            "video-scene-v1",
            context.video_id,
            &index,
            &start,
            &end,
        ])
    } else {
        EntityId::derive(&[
            "visual-analysis",
            "video-scene-v1",
            context.video_id,
            &index,
        ])
    }
}

fn semantic_producer() -> Result<Producer> {
    Ok(Producer::new(producer_id(
        "visual-analysis:video-text-semantic-heuristics",
    )?)
    .with_version(CLASSIFIER_VERSION))
}

fn concept(value: &str) -> Result<ConceptId> {
    ConceptId::new(value).map_err(semantic_error)
}

fn producer_id(value: &str) -> Result<ProducerId> {
    ProducerId::new(value).map_err(semantic_error)
}

fn evidence_key(value: &str) -> Result<EvidenceKey> {
    EvidenceKey::new(value).map_err(semantic_error)
}

fn semantic_error(error: SemanticContractError) -> DetectError {
    DetectError::InvalidArgument(error.to_string())
}

fn integer_evidence(key: &str, value: i64) -> Result<Evidence> {
    Ok(Evidence::Observation {
        feature: evidence_key(key)?,
        value: EvidenceValue::Integer(value),
    })
}

fn boolean_evidence(key: &str, value: bool) -> Result<Evidence> {
    Ok(Evidence::Observation {
        feature: evidence_key(key)?,
        value: EvidenceValue::Boolean(value),
    })
}

fn number_evidence(key: &str, value: f64) -> Result<Evidence> {
    Ok(Evidence::Observation {
        feature: evidence_key(key)?,
        value: EvidenceValue::Number(FiniteNumber::new(value).map_err(semantic_error)?),
    })
}

fn text_evidence(key: &str, value: String) -> Result<Evidence> {
    Ok(Evidence::Observation {
        feature: evidence_key(key)?,
        value: EvidenceValue::Text(value),
    })
}

fn relation_sort_key(relation: &Relation) -> (String, String) {
    let subject = match &relation.subject {
        SemanticNodeRef::Entity(id) => id.as_str().to_string(),
        SemanticNodeRef::Concept(id) => id.as_str().to_string(),
    };
    let object = match &relation.object {
        SemanticNodeRef::Entity(id) => id.as_str().to_string(),
        SemanticNodeRef::Concept(id) => id.as_str().to_string(),
    };
    (subject, object)
}

fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn box_center(region: BoundingBox) -> (f64, f64) {
    (
        f64::from(region.x) + f64::from(region.width) / 2.0,
        f64::from(region.y) + f64::from(region.height) / 2.0,
    )
}

fn bounding_box_iou(left: BoundingBox, right: BoundingBox) -> f64 {
    let left_x2 = u64::from(left.x) + u64::from(left.width);
    let left_y2 = u64::from(left.y) + u64::from(left.height);
    let right_x2 = u64::from(right.x) + u64::from(right.width);
    let right_y2 = u64::from(right.y) + u64::from(right.height);
    let intersection_width = left_x2
        .min(right_x2)
        .saturating_sub(u64::from(left.x).max(u64::from(right.x)));
    let intersection_height = left_y2
        .min(right_y2)
        .saturating_sub(u64::from(left.y).max(u64::from(right.y)));
    let intersection = intersection_width.saturating_mul(intersection_height);
    if intersection == 0 {
        return 0.0;
    }
    let left_area = u64::from(left.width).saturating_mul(u64::from(left.height));
    let right_area = u64::from(right.width).saturating_mul(u64::from(right.height));
    let union = left_area.saturating_add(right_area).saturating_sub(intersection);
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn i64_from_u64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use semantic_core::{Evidence, EvidenceValue};
    use video_analysis_core::{FramePosition, Observation, ObservationKind, Scene};

    use super::*;

    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    fn position(frame: u64) -> FramePosition {
        FramePosition::from_frame_index(frame, Rational64::new(30, 1))
    }

    fn scene(start: u64, end: u64) -> Scene {
        Scene {
            start: position(start),
            end: position(end),
        }
    }

    fn text_observation(
        text: &str,
        frame: u64,
        scene_index: Option<u64>,
        region: BoundingBox,
    ) -> Observation {
        let mut observation = Observation::new("ocr", ObservationKind::Text)
            .at_frame(position(frame))
            .text(text)
            .score(0.88)
            .region(region);
        if let Some(scene_index) = scene_index {
            observation = observation.in_scene(scene_index);
        }
        observation
    }

    fn context<'a>(scenes: &'a [Scene], duration_seconds: f64) -> VideoTextSemanticContext<'a> {
        VideoTextSemanticContext {
            video_id: "fixture-video",
            width: WIDTH,
            height: HEIGHT,
            duration_seconds: Some(duration_seconds),
            scenes,
        }
    }

    #[test]
    fn associates_repeated_ocr_into_a_stable_order_independent_track() {
        let scenes = [scene(0, 299)];
        let observations = vec![
            text_observation(
                "EXIT",
                30,
                Some(0),
                BoundingBox::new(100, 100, 160, 60).unwrap(),
            ),
            text_observation(
                "EXIT",
                60,
                Some(0),
                BoundingBox::new(103, 102, 160, 60).unwrap(),
            ),
        ];
        let forward = analyze_video_text_semantics(context(&scenes, 10.0), &observations).unwrap();
        let mut reversed = observations.clone();
        reversed.reverse();
        let backward = analyze_video_text_semantics(context(&scenes, 10.0), &reversed).unwrap();

        assert_eq!(forward.tracks.len(), 1);
        assert_eq!(forward.tracks[0].sample_count, 2);
        assert_eq!(forward.tracks[0].id, backward.tracks[0].id);
        assert_eq!(forward.tracks[0].role, VideoTextRole::SceneText);
    }

    #[test]
    fn upstream_track_identity_can_survive_ocr_text_changes() {
        let scenes = [scene(0, 299)];
        let observations = vec![
            text_observation(
                "HELLO",
                30,
                Some(0),
                BoundingBox::new(500, 800, 500, 80).unwrap(),
            )
            .track_id("subtitle-1"),
            text_observation(
                "HELLO!",
                60,
                Some(0),
                BoundingBox::new(505, 800, 500, 80).unwrap(),
            )
            .track_id("subtitle-1"),
        ];
        let analysis = analyze_video_text_semantics(context(&scenes, 10.0), &observations).unwrap();

        assert_eq!(analysis.tracks.len(), 1);
        assert_eq!(analysis.tracks[0].sample_count, 2);
        assert!(analysis.annotations[0].evidence.iter().any(|item| matches!(
            item,
            Evidence::Observation { feature, value: EvidenceValue::Text(value) }
                if feature.as_str() == "visual-analysis:video-text-evidence:upstream-track-id"
                    && value == "subtitle-1"
        )));
    }

    #[test]
    fn classifies_bottom_center_short_lived_text_as_subtitle() {
        let scenes = [scene(0, 599)];
        let observations = vec![
            text_observation(
                "I'll see you tomorrow.",
                90,
                Some(0),
                BoundingBox::new(460, 850, 1000, 90).unwrap(),
            ),
            text_observation(
                "I'll see you tomorrow.",
                120,
                Some(0),
                BoundingBox::new(465, 850, 1000, 90).unwrap(),
            ),
        ];
        let analysis = analyze_video_text_semantics(context(&scenes, 20.0), &observations).unwrap();

        assert_eq!(analysis.tracks[0].role, VideoTextRole::Subtitle);
        let role = analysis
            .annotations
            .iter()
            .find(|annotation| {
                annotation.concept.as_str() == "visual-analysis:video-text-role:subtitle"
            })
            .unwrap();
        assert_eq!(role.confidence, None);
        assert!(role.evidence.iter().any(|item| matches!(
            item,
            Evidence::Observation { feature, value: EvidenceValue::Number(_) }
                if feature.as_str() == "visual-analysis:video-text-evidence:average-ocr-score"
        )));
    }

    #[test]
    fn classifies_dense_late_scene_as_end_credits() {
        let scenes = [scene(0, 2399), scene(2400, 2999)];
        let mut observations = Vec::new();
        for (index, text) in ["DIRECTED BY", "ALICE", "CAST", "BOB"].iter().enumerate() {
            observations.push(text_observation(
                text,
                2550 + index as u64,
                Some(1),
                BoundingBox::new(600, 250 + index as u32 * 120, 700, 70).unwrap(),
            ));
        }
        let analysis = analyze_video_text_semantics(context(&scenes, 100.0), &observations).unwrap();

        assert_eq!(analysis.tracks.len(), 4);
        assert!(analysis
            .tracks
            .iter()
            .all(|track| track.role == VideoTextRole::EndCredit));
    }

    #[test]
    fn classifies_stable_multi_text_scene_as_presentation_slide() {
        let scenes = [scene(0, 899)];
        let mut observations = Vec::new();
        for (index, (text, y)) in [
            ("Quarterly Revenue", 120),
            ("Revenue grew 18 percent", 340),
            ("Margin expanded", 520),
        ]
        .iter()
        .enumerate()
        {
            let x = 180 + index as u32 * 20;
            observations.push(text_observation(
                text,
                60,
                Some(0),
                BoundingBox::new(x, *y, 900, 80).unwrap(),
            ));
            observations.push(text_observation(
                text,
                90,
                Some(0),
                BoundingBox::new(x + 2, *y + 1, 900, 80).unwrap(),
            ));
        }
        let analysis = analyze_video_text_semantics(context(&scenes, 30.0), &observations).unwrap();

        assert_eq!(analysis.tracks.len(), 3);
        assert!(analysis
            .tracks
            .iter()
            .all(|track| track.role == VideoTextRole::PresentationSlide));
    }

    #[test]
    fn single_physical_text_defaults_to_scene_text() {
        let scenes = [scene(0, 599)];
        let observations = vec![text_observation(
            "CENTRAL STATION",
            100,
            Some(0),
            BoundingBox::new(100, 250, 500, 120).unwrap(),
        )];
        let analysis = analyze_video_text_semantics(context(&scenes, 20.0), &observations).unwrap();

        assert_eq!(analysis.tracks[0].role, VideoTextRole::SceneText);
    }

    #[test]
    fn infers_scene_membership_from_existing_scene_boundaries() {
        let scenes = [scene(0, 299), scene(300, 599)];
        let observations = vec![text_observation(
            "STORE",
            420,
            None,
            BoundingBox::new(100, 200, 300, 100).unwrap(),
        )];
        let analysis = analyze_video_text_semantics(context(&scenes, 20.0), &observations).unwrap();

        assert_eq!(analysis.tracks[0].scene_indices, vec![1]);
        assert_eq!(analysis.relations.len(), 1);
        assert_eq!(
            analysis.relations[0].predicate.as_str(),
            "visual-analysis:relation:appears-in-scene"
        );
    }
}
