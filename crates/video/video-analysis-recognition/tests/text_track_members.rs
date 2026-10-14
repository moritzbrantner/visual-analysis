//! Acceptance tests for issue #77: each `VideoTextTrack` exposes the input
//! observations it groups as `observation_indices`.
//!
//! Expected memberships are derived from the fixtures constructed here (each
//! fixture observation carries the group it is built to belong to), never from
//! the tracker's internals.

use std::collections::{BTreeMap, BTreeSet};

use num_rational::Rational64;
use video_analysis_core::{BoundingBox, FramePosition, Observation, ObservationKind, Scene};
use video_analysis_recognition::{
    analyze_video_text_semantics, VideoTextSemanticAnalysis, VideoTextSemanticContext,
    VideoTextTrack,
};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const EPSILON: f64 = 1e-9;

fn position(frame: u64) -> FramePosition {
    FramePosition::from_frame_index(frame, Rational64::new(30, 1))
}

fn scene(start: u64, end: u64) -> Scene {
    Scene {
        start: position(start),
        end: position(end),
    }
}

fn region(x: u32, y: u32, width: u32, height: u32) -> BoundingBox {
    BoundingBox::new(x, y, width, height).unwrap()
}

fn ocr(text: &str, frame: u64, region: BoundingBox) -> Observation {
    Observation::new("ocr", ObservationKind::Text)
        .at_frame(position(frame))
        .text(text)
        .score(0.9)
        .region(region)
}

/// A non-text observation (detector output) that nevertheless carries text.
fn non_text(text: &str, frame: u64, region: BoundingBox) -> Observation {
    Observation::new("detector", ObservationKind::Object)
        .at_frame(position(frame))
        .text(text)
        .label("sign")
        .score(0.9)
        .region(region)
}

/// A text-kind observation whose text is blank after trimming.
fn blank_text(text: &str, frame: u64, region: BoundingBox) -> Observation {
    ocr(text, frame, region)
}

/// A text-kind observation with no text at all.
fn missing_text(frame: u64, region: BoundingBox) -> Observation {
    Observation::new("ocr", ObservationKind::Text)
        .at_frame(position(frame))
        .score(0.9)
        .region(region)
}

fn context<'a>(scenes: &'a [Scene], duration_seconds: f64) -> VideoTextSemanticContext<'a> {
    VideoTextSemanticContext {
        video_id: "members-fixture",
        width: WIDTH,
        height: HEIGHT,
        duration_seconds: Some(duration_seconds),
        scenes,
    }
}

/// One fixture observation plus the group it is constructed to belong to
/// (`None` for observations that must be filtered out).
struct Fixture {
    observations: Vec<Observation>,
    groups: Vec<Option<&'static str>>,
}

impl Fixture {
    fn new(entries: Vec<(Option<&'static str>, Observation)>) -> Self {
        let (groups, observations) = entries.into_iter().unzip();
        Self {
            observations,
            groups,
        }
    }

    /// Expected member indices per group, sorted ascending.
    fn expected_groups(&self) -> BTreeMap<&'static str, Vec<usize>> {
        let mut groups = BTreeMap::<&'static str, Vec<usize>>::new();
        for (index, group) in self.groups.iter().enumerate() {
            if let Some(group) = group {
                groups.entry(group).or_default().push(index);
            }
        }
        groups
    }

    fn filtered_indices(&self) -> Vec<usize> {
        self.groups
            .iter()
            .enumerate()
            .filter_map(|(index, group)| group.is_none().then_some(index))
            .collect()
    }
}

fn frame_of(observation: &Observation) -> Option<u64> {
    observation.frame.map(|frame| frame.frame_index)
}

fn seconds_of(observation: &Observation) -> Option<f64> {
    observation
        .timestamp
        .or_else(|| observation.frame.map(|frame| frame.timestamp))
        .map(|timestamp| timestamp.seconds())
}

fn is_text_observation(observation: &Observation) -> bool {
    let has_text = observation
        .text
        .as_deref()
        .is_some_and(|text| !text.trim().is_empty());
    let text_like = matches!(observation.kind, ObservationKind::Text)
        || observation.analyzer.to_ascii_lowercase().contains("ocr");
    has_text && text_like
}

/// Contract invariants that hold for any input: indices are sorted, unique,
/// in range, match `sample_count`, reference text observations inside the
/// track's frame/time/scene bounds, and are disjoint across tracks.
fn assert_membership_invariants(
    observations: &[Observation],
    analysis: &VideoTextSemanticAnalysis,
) {
    let mut owner = BTreeMap::<usize, usize>::new();
    for (track_position, track) in analysis.tracks.iter().enumerate() {
        let indices = &track.observation_indices;
        assert!(
            !indices.is_empty(),
            "track {:?} ({}) reports no member observations",
            track.text,
            track.id.as_str()
        );
        assert!(
            indices.windows(2).all(|pair| pair[0] < pair[1]),
            "track {:?} indices must be strictly ascending: {indices:?}",
            track.text
        );
        assert_eq!(
            indices.len(),
            track.sample_count as usize,
            "track {:?} member count must equal sample_count",
            track.text
        );
        for &index in indices {
            assert!(
                index < observations.len(),
                "index {index} out of range for {} observations",
                observations.len()
            );
            let member = &observations[index];
            assert!(
                is_text_observation(member),
                "member {index} of track {:?} is not a text observation",
                track.text
            );
            if let Some(frame) = frame_of(member) {
                let start = track
                    .start_frame
                    .expect("track with framed member has start");
                let end = track.end_frame.expect("track with framed member has end");
                assert!(
                    (start..=end).contains(&frame),
                    "member {index} frame {frame} outside track {:?} [{start}, {end}]",
                    track.text
                );
            }
            if let Some(seconds) = seconds_of(member) {
                let start = track
                    .start_seconds
                    .expect("track with timed member has start");
                let end = track.end_seconds.expect("track with timed member has end");
                assert!(
                    seconds >= start - EPSILON && seconds <= end + EPSILON,
                    "member {index} at {seconds}s outside track {:?} [{start}, {end}]",
                    track.text
                );
            }
            if let Some(scene_index) = member.scene_index {
                assert!(
                    track.scene_indices.contains(&scene_index),
                    "member {index} scene {scene_index} missing from track {:?} scenes {:?}",
                    track.text,
                    track.scene_indices
                );
            }
            if let Some(previous) = owner.insert(index, track_position) {
                panic!("observation {index} belongs to tracks {previous} and {track_position}");
            }
        }
    }
    let total: usize = analysis
        .tracks
        .iter()
        .map(|track| track.sample_count as usize)
        .sum();
    assert_eq!(
        owner.len(),
        total,
        "members of different tracks must be disjoint"
    );
}

fn track_bounds(track: &VideoTextTrack) -> (Option<u64>, Option<u64>) {
    (track.start_frame, track.end_frame)
}

/// Asserts that every track groups exactly one fixture group, and that each
/// fixture group maps to exactly one track whose indices equal the group.
fn assert_exact_membership(fixture: &Fixture, analysis: &VideoTextSemanticAnalysis) {
    assert_membership_invariants(&fixture.observations, analysis);
    let expected = fixture.expected_groups();
    assert_eq!(
        analysis.tracks.len(),
        expected.len(),
        "fixture is built to yield one track per group"
    );
    for (group, indices) in &expected {
        let frames = indices
            .iter()
            .filter_map(|&index| frame_of(&fixture.observations[index]))
            .collect::<Vec<_>>();
        let bounds = (frames.iter().copied().min(), frames.iter().copied().max());
        let matching = analysis
            .tracks
            .iter()
            .filter(|track| track_bounds(track) == bounds)
            .filter(|track| {
                indices.iter().any(|&index| {
                    fixture.observations[index]
                        .text
                        .as_deref()
                        .is_some_and(|text| text.eq_ignore_ascii_case(&track.text))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matching.len(),
            1,
            "group {group:?} with frame bounds {bounds:?} must match exactly one track"
        );
        let track = matching[0];
        assert_eq!(
            &track.observation_indices, indices,
            "group {group:?} membership"
        );
        assert_eq!(track.sample_count as usize, indices.len());
    }

    let members = analysis
        .tracks
        .iter()
        .flat_map(|track| track.observation_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    for filtered in fixture.filtered_indices() {
        assert!(
            !members.contains(&filtered),
            "filtered observation {filtered} must belong to no track"
        );
    }
    let all_text = (0..fixture.observations.len())
        .filter(|&index| is_text_observation(&fixture.observations[index]))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        members, all_text,
        "every text observation belongs to exactly one track"
    );
}

/// Repeated same-text samples across frames, deliberately out of input order,
/// interleaved with non-text and blank observations, plus a reappearance
/// after a long disappearance and an unrelated text.
fn multi_frame_fixture() -> Fixture {
    let exit = region(100, 100, 160, 60);
    Fixture::new(vec![
        (
            Some("exit-early"),
            ocr("EXIT", 60, region(103, 102, 160, 60)),
        ),
        (None, non_text("EXIT", 45, exit)),
        (Some("exit-early"), ocr("EXIT", 30, exit)),
        (None, blank_text("   ", 50, exit)),
        (None, missing_text(55, exit)),
        (
            Some("exit-early"),
            ocr("EXIT", 90, region(101, 101, 160, 60)),
        ),
        (Some("exit-late"), ocr("EXIT", 2000, exit)),
        (None, blank_text("", 2010, exit)),
        (
            Some("exit-late"),
            ocr("EXIT", 2030, region(102, 100, 160, 60)),
        ),
        (Some("cafe"), ocr("CAFE", 30, region(900, 300, 200, 60))),
    ])
}

#[test]
fn multi_frame_same_text_tracks_report_exactly_their_observations() {
    let fixture = multi_frame_fixture();
    let scenes = [scene(0, 2999)];
    let analysis =
        analyze_video_text_semantics(context(&scenes, 100.0), &fixture.observations).unwrap();

    assert_exact_membership(&fixture, &analysis);
    let expected = fixture.expected_groups();
    assert_eq!(expected["exit-early"], vec![0, 2, 5]);
    assert_eq!(expected["exit-late"], vec![6, 8]);
    assert_eq!(expected["cafe"], vec![9]);
    assert_eq!(fixture.filtered_indices(), vec![1, 3, 4, 7]);
}

#[test]
fn upstream_track_ids_report_exactly_their_observations() {
    let bottom = region(500, 900, 500, 80);
    let fixture = Fixture::new(vec![
        (
            Some("subtitle-1"),
            ocr("HELLO", 30, bottom).track_id("subtitle-1"),
        ),
        (
            Some("subtitle-2"),
            ocr("WORLD", 30, region(200, 100, 300, 60)).track_id("subtitle-2"),
        ),
        // Same upstream id, but not a text observation: filtered out.
        (None, non_text("HELLO", 75, bottom).track_id("subtitle-1")),
        (
            Some("subtitle-1"),
            ocr("HELLO!", 60, region(505, 900, 500, 80)).track_id("subtitle-1"),
        ),
        // Same upstream id, but blank text: filtered out.
        (None, blank_text(" ", 80, bottom).track_id("subtitle-1")),
        // Same text and place, but no upstream id: its own track.
        (Some("untracked"), ocr("HELLO", 150, bottom)),
        (
            Some("subtitle-2"),
            ocr("WORLD", 120, region(200, 100, 300, 60)).track_id("subtitle-2"),
        ),
    ]);
    let scenes = [scene(0, 299)];
    let analysis =
        analyze_video_text_semantics(context(&scenes, 10.0), &fixture.observations).unwrap();

    assert_exact_membership(&fixture, &analysis);
    let expected = fixture.expected_groups();
    assert_eq!(expected["subtitle-1"], vec![0, 3]);
    assert_eq!(expected["subtitle-2"], vec![1, 6]);
    assert_eq!(expected["untracked"], vec![5]);
}

#[test]
fn two_scene_text_tracks_report_exactly_their_observations() {
    // The same text at the same place in two scenes forms two tracks; scene
    // membership is inferred from the boundaries or given explicitly.
    let sign = region(100, 200, 300, 100);
    let fixture = Fixture::new(vec![
        (Some("scene-1"), ocr("OPEN", 400, sign)),
        (Some("scene-0"), ocr("OPEN", 100, sign)),
        (None, non_text("OPEN", 150, sign).in_scene(0)),
        (Some("scene-1"), ocr("OPEN", 310, sign).in_scene(1)),
        (Some("scene-0"), ocr("OPEN", 200, sign).in_scene(0)),
        (None, missing_text(320, sign).in_scene(1)),
    ]);
    let scenes = [scene(0, 300), scene(300, 600)];
    let analysis =
        analyze_video_text_semantics(context(&scenes, 20.0), &fixture.observations).unwrap();

    assert_exact_membership(&fixture, &analysis);
    let expected = fixture.expected_groups();
    assert_eq!(expected["scene-0"], vec![1, 4]);
    assert_eq!(expected["scene-1"], vec![0, 3]);
    for track in &analysis.tracks {
        assert_eq!(track.scene_indices.len(), 1);
        let scene_index = track.scene_indices[0];
        for &index in &track.observation_indices {
            let frame = frame_of(&fixture.observations[index]).unwrap();
            assert!(
                frame >= scenes[scene_index as usize].start.frame_index
                    && frame < scenes[scene_index as usize].end.frame_index,
                "member {index} lies outside scene {scene_index}"
            );
        }
    }
}

#[test]
fn indices_follow_the_caller_input_order() {
    // Track identities are order independent; their indices must always refer
    // to the slice the caller passed, so reversing the input mirrors them.
    let fixture = multi_frame_fixture();
    let scenes = [scene(0, 2999)];
    let forward =
        analyze_video_text_semantics(context(&scenes, 100.0), &fixture.observations).unwrap();
    let mut reversed = fixture.observations.clone();
    reversed.reverse();
    let backward = analyze_video_text_semantics(context(&scenes, 100.0), &reversed).unwrap();
    assert_membership_invariants(&reversed, &backward);

    let last = fixture.observations.len() - 1;
    assert_eq!(forward.tracks.len(), backward.tracks.len());
    for track in &forward.tracks {
        let mirrored = backward
            .tracks
            .iter()
            .find(|candidate| candidate.id == track.id)
            .expect("track identity is order independent");
        let mut expected = track
            .observation_indices
            .iter()
            .map(|index| last - index)
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(mirrored.observation_indices, expected);
        assert!(!mirrored.observation_indices.is_empty());
    }
}

#[test]
fn only_filtered_observations_yield_no_tracks() {
    let sign = region(100, 200, 300, 100);
    let observations = vec![
        non_text("EXIT", 30, sign),
        blank_text("  ", 60, sign),
        missing_text(90, sign),
    ];
    let scenes = [scene(0, 299)];
    let analysis = analyze_video_text_semantics(context(&scenes, 10.0), &observations).unwrap();
    assert!(analysis.tracks.is_empty());
}

#[test]
fn observation_indices_round_trip_through_serde() {
    let fixture = multi_frame_fixture();
    let scenes = [scene(0, 2999)];
    let analysis =
        analyze_video_text_semantics(context(&scenes, 100.0), &fixture.observations).unwrap();
    // Tracks only: annotation evidence carries f32-derived OCR scores whose
    // JSON round trip is outside this contract.
    let json = serde_json::to_string(&analysis.tracks).unwrap();
    let decoded: Vec<VideoTextTrack> = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, analysis.tracks);

    let mut track = analysis.tracks[0].clone();
    track.observation_indices = vec![0, 2, 5];
    let value = serde_json::to_value(&track).unwrap();
    assert_eq!(value["observation_indices"], serde_json::json!([0, 2, 5]));
    let decoded: VideoTextTrack = serde_json::from_value(value).unwrap();
    assert_eq!(decoded, track);
}

#[test]
fn tracks_serialized_without_observation_indices_deserialize_empty() {
    let fixture = multi_frame_fixture();
    let scenes = [scene(0, 2999)];
    let analysis =
        analyze_video_text_semantics(context(&scenes, 100.0), &fixture.observations).unwrap();
    let mut track = analysis.tracks[0].clone();
    track.observation_indices = vec![1, 2];

    let mut value = serde_json::to_value(&track).unwrap();
    let removed = value.as_object_mut().unwrap().remove("observation_indices");
    assert!(
        removed.is_some(),
        "field is serialized as observation_indices"
    );
    let decoded: VideoTextTrack = serde_json::from_value(value).unwrap();
    assert!(decoded.observation_indices.is_empty());
    track.observation_indices = Vec::new();
    assert_eq!(decoded, track);

    let mut analysis_value = serde_json::to_value(&analysis).unwrap();
    for track in analysis_value["tracks"].as_array_mut().unwrap() {
        track.as_object_mut().unwrap().remove("observation_indices");
    }
    let decoded: VideoTextSemanticAnalysis = serde_json::from_value(analysis_value).unwrap();
    assert!(decoded
        .tracks
        .iter()
        .all(|track| track.observation_indices.is_empty()));
}
