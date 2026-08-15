use video_analysis_core::{
    AnalysisEvent, Observation, ObservationKind, TextSegment, Timebase, Timestamp,
};
use video_analysis_core::{FramePosition, Scene};
use video_analysis_dataset::{
    AnalysisDataset, DatasetRecord, FeatureRecord, FeatureValue, SceneRecord, TextSegmentRecord,
};
use video_analysis_storage::{
    build_manifest, read_dataset_dir, read_dataset_json, read_jsonl, write_dataset_dir,
    write_dataset_json, write_jsonl,
};

fn timestamp(seconds: i64) -> Timestamp {
    Timestamp::new(seconds, Timebase::new(1, 1))
}

fn frame_position(frame_index: u64) -> FramePosition {
    FramePosition {
        frame_index,
        timestamp: timestamp(frame_index as i64),
    }
}

fn scene(start: u64, end: u64) -> Scene {
    Scene {
        start: frame_position(start),
        end: frame_position(end),
    }
}

fn text_segment(index: u64, text: &str) -> TextSegment<'_> {
    TextSegment {
        segment_index: index,
        timestamp: Some(timestamp(index as i64)),
        text,
        language: Some("en"),
        is_final: true,
    }
}

fn dataset_with_scene_text_and_feature() -> AnalysisDataset {
    let mut dataset = AnalysisDataset::empty();
    let scene = scene(0, 2);
    let text = text_segment(0, "Rust video analysis test fixture.");
    dataset.push(DatasetRecord::Scene(SceneRecord::from_scene(0, &scene)));
    dataset.push(DatasetRecord::TextSegment(TextSegmentRecord::from_segment(
        "transcript",
        &text,
    )));
    dataset.extend_observations([Observation::new("fixture", ObservationKind::Text)
        .at_timestamp(timestamp(0))
        .in_scene(0)
        .label("fixture")
        .text(text.text)]);
    dataset
        .extend_events([AnalysisEvent::new("fixture", "text:fixture").at_timestamp(timestamp(0))]);
    dataset.push(DatasetRecord::Feature(
        FeatureRecord::new("fixture.vector", FeatureValue::Vector(vec![1.0, 2.0, 3.0]))
            .scope("global")
            .timestamp(timestamp(0)),
    ));
    dataset
}

#[test]
fn dataset_json_jsonl_and_manifest_round_trip_public_records() {
    let dataset = dataset_with_scene_text_and_feature();
    let dir = tempfile::tempdir().unwrap();
    let json_path = dir.path().join("dataset.json");
    let jsonl_path = dir.path().join("records.jsonl");

    write_dataset_json(&json_path, &dataset).unwrap();
    write_jsonl(&jsonl_path, &dataset).unwrap();

    let from_json = read_dataset_json(&json_path).unwrap();
    let from_jsonl = read_jsonl(&jsonl_path).unwrap();
    assert_eq!(from_json.records, dataset.records);
    assert_eq!(from_jsonl.records, dataset.records);

    let manifest = build_manifest(&dataset, "records.jsonl");
    assert_eq!(manifest.record_count, dataset.records.len() as u64);
    assert_eq!(manifest.record_counts["scene"], 1);
    assert_eq!(manifest.record_counts["feature"], 1);

    let dataset_dir = dir.path().join("dataset-dir");
    let written_manifest = write_dataset_dir(&dataset_dir, &dataset).unwrap();
    assert_eq!(written_manifest.record_count, manifest.record_count);
    let from_dir = read_dataset_dir(&dataset_dir).unwrap();
    assert_eq!(from_dir.records, dataset.records);
}
