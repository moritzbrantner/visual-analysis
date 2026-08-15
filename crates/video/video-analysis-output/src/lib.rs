#![doc = include_str!("../README.md")]

pub mod surface;
use std::io::{self, Write};

use serde::Serialize;
use video_analysis_core::{Cut, DetectionResult, FramePosition, MetricsStore, Scene};

/// Writes scene list CSV.
pub fn write_scene_list_csv(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    writeln!(
        writer,
        "Scene Number,Start Frame,Start Timecode,Start Seconds,End Frame,End Timecode,End Seconds"
    )?;
    for (index, scene) in scenes.iter().enumerate() {
        let start_seconds = scene.start.timestamp.seconds();
        let end_seconds = scene.end.timestamp.seconds();
        writeln!(
            writer,
            "{},{},{},{:.6},{},{},{:.6}",
            index + 1,
            scene.start.frame_index,
            seconds_to_timecode(start_seconds),
            start_seconds,
            scene.end.frame_index,
            seconds_to_timecode(end_seconds),
            end_seconds
        )?;
    }
    Ok(())
}

/// Writes stats CSV.
pub fn write_stats_csv(mut writer: impl Write, metrics: &MetricsStore) -> io::Result<()> {
    let keys: Vec<_> = metrics.keys().collect();
    write!(writer, "Frame Number,Frame Index")?;
    for key in &keys {
        write!(writer, ",{key}")?;
    }
    writeln!(writer)?;
    for (frame_index, row) in metrics.rows() {
        write!(writer, "{},{}", frame_index + 1, frame_index)?;
        for key in &keys {
            match row.get(*key) {
                Some(value) => write!(writer, ",{value}")?,
                None => write!(writer, ",")?,
            }
        }
        writeln!(writer)?;
    }
    Ok(())
}

/// Writes scene list HTML.
pub fn write_scene_list_html(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    writeln!(
        writer,
        "<!doctype html><meta charset=\"utf-8\"><title>Scene List</title><table>"
    )?;
    writeln!(
        writer,
        "<tr><th>Scene</th><th>Start Frame</th><th>Start</th><th>End Frame</th><th>End</th></tr>"
    )?;
    for (index, scene) in scenes.iter().enumerate() {
        writeln!(
            writer,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            index + 1,
            scene.start.frame_index,
            seconds_to_timecode(scene.start.timestamp.seconds()),
            scene.end.frame_index,
            seconds_to_timecode(scene.end.timestamp.seconds())
        )?;
    }
    writeln!(writer, "</table>")?;
    Ok(())
}

/// Writes a CMX-style EDL scene list.
pub fn write_scene_list_edl(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    writeln!(writer, "TITLE: Scene List")?;
    writeln!(writer, "FCM: NON-DROP FRAME")?;
    for (index, scene) in scenes.iter().enumerate() {
        let event = index + 1;
        let start = frame_position_to_edit_timecode(scene.start);
        let end = frame_position_to_edit_timecode(scene.end);
        writeln!(
            writer,
            "{event:03}  AX       V     C        {start} {end} {start} {end}"
        )?;
        writeln!(writer, "* FROM CLIP NAME: Scene {event}")?;
    }
    Ok(())
}

/// Writes a minimal Final Cut Pro 7 XML scene list.
pub fn write_scene_list_fcp7(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(writer, r#"<!DOCTYPE xmeml>"#)?;
    writeln!(writer, r#"<xmeml version="5">"#)?;
    writeln!(writer, r#"  <sequence id="scene-list">"#)?;
    writeln!(writer, r#"    <name>Scene List</name>"#)?;
    writeln!(writer, r#"    <media><video><track>"#)?;
    for (index, scene) in scenes.iter().enumerate() {
        let name = xml_escape(&format!("Scene {}", index + 1));
        writeln!(writer, r#"      <clipitem id="scene-{}">"#, index + 1)?;
        writeln!(writer, r#"        <name>{name}</name>"#)?;
        writeln!(
            writer,
            r#"        <start>{}</start>"#,
            scene.start.frame_index
        )?;
        writeln!(writer, r#"        <end>{}</end>"#, scene.end.frame_index)?;
        writeln!(writer, r#"        <in>{}</in>"#, scene.start.frame_index)?;
        writeln!(writer, r#"        <out>{}</out>"#, scene.end.frame_index)?;
        writeln!(writer, r#"      </clipitem>"#)?;
    }
    writeln!(writer, r#"    </track></video></media>"#)?;
    writeln!(writer, r#"  </sequence>"#)?;
    writeln!(writer, r#"</xmeml>"#)?;
    Ok(())
}

/// Writes a minimal Final Cut Pro XML scene list.
pub fn write_scene_list_fcpx(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(writer, r#"<fcpxml version="1.10">"#)?;
    writeln!(
        writer,
        r#"  <library><event name="Scene List"><project name="Scene List">"#
    )?;
    writeln!(writer, r#"    <sequence><spine>"#)?;
    for (index, scene) in scenes.iter().enumerate() {
        let duration = scene
            .end
            .timestamp
            .seconds()
            .max(scene.start.timestamp.seconds())
            - scene.start.timestamp.seconds();
        writeln!(
            writer,
            r#"      <gap name="Scene {}" offset="{:.6}s" duration="{:.6}s"/>"#,
            index + 1,
            scene.start.timestamp.seconds(),
            duration
        )?;
    }
    writeln!(writer, r#"    </spine></sequence>"#)?;
    writeln!(writer, r#"  </project></event></library>"#)?;
    writeln!(writer, r#"</fcpxml>"#)?;
    Ok(())
}

/// Writes a compact OpenTimelineIO-compatible JSON scene list.
pub fn write_scene_list_otio(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    let clips = scenes
        .iter()
        .enumerate()
        .map(|(index, scene)| {
            serde_json::json!({
                "OTIO_SCHEMA": "Clip.2",
                "name": format!("Scene {}", index + 1),
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": rational_time_json(scene.start),
                    "duration": rational_time_json_duration(scene.start, scene.end)
                },
                "metadata": {
                    "videoAnalysis": {
                        "startFrame": scene.start.frame_index,
                        "endFrame": scene.end.frame_index
                    }
                }
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "OTIO_SCHEMA": "Timeline.1",
        "name": "Scene List",
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "children": [{
                "OTIO_SCHEMA": "Track.1",
                "name": "Scenes",
                "kind": "Video",
                "children": clips
            }]
        }
    });
    serde_json::to_writer_pretty(&mut writer, &value).map_err(io::Error::other)?;
    writeln!(writer)?;
    Ok(())
}

/// Writes an FFmpeg qpfile that marks every scene start as an I-frame.
pub fn write_scene_list_qp(mut writer: impl Write, scenes: &[Scene]) -> io::Result<()> {
    for scene in scenes {
        writeln!(writer, "{} I", scene.start.frame_index)?;
    }
    Ok(())
}

/// Writes detection result JSON.
pub fn write_detection_result_json(
    mut writer: impl Write,
    result: &DetectionResult,
) -> io::Result<()> {
    serde_json::to_writer_pretty(&mut writer, &DetectionResultJson::from(result))
        .map_err(io::Error::other)?;
    writeln!(writer)?;
    Ok(())
}

/// Writes detection outputs.
pub fn write_detection_outputs(
    result: &DetectionResult,
    scenes_writer: impl Write,
    stats_writer: Option<impl Write>,
) -> io::Result<()> {
    write_scene_list_csv(scenes_writer, &result.scenes)?;
    if let Some(stats_writer) = stats_writer {
        write_stats_csv(stats_writer, &result.metrics)?;
    }
    Ok(())
}

fn seconds_to_timecode(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let total_seconds = total_ms / 1000;
    let ms = total_ms % 1000;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{ms:03}")
}

fn frame_position_to_edit_timecode(position: FramePosition) -> String {
    let fps = edit_timecode_fps(position).max(1);
    let total_frames = position.frame_index;
    let frames = total_frames % fps;
    let total_seconds = total_frames / fps;
    let seconds = total_seconds % 60;
    let minutes = (total_seconds / 60) % 60;
    let hours = total_seconds / 3600;
    format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}")
}

fn edit_timecode_fps(position: FramePosition) -> u64 {
    if position.timestamp.timebase.num <= 0 || position.timestamp.timebase.den <= 0 {
        return 30;
    }
    (position.timestamp.timebase.den as f64 / position.timestamp.timebase.num as f64)
        .round()
        .max(1.0) as u64
}

fn rational_time_json(position: FramePosition) -> serde_json::Value {
    serde_json::json!({
        "OTIO_SCHEMA": "RationalTime.1",
        "value": position.timestamp.pts,
        "rate": position.timestamp.timebase.den as f64 / position.timestamp.timebase.num as f64
    })
}

fn rational_time_json_duration(start: FramePosition, end: FramePosition) -> serde_json::Value {
    let rate = start.timestamp.timebase.den as f64 / start.timestamp.timebase.num as f64;
    let duration = if start.timestamp.timebase == end.timestamp.timebase {
        end.timestamp.pts.saturating_sub(start.timestamp.pts)
    } else {
        ((end.timestamp.seconds() - start.timestamp.seconds()).max(0.0) * rate).round() as i64
    };
    serde_json::json!({
        "OTIO_SCHEMA": "RationalTime.1",
        "value": duration,
        "rate": rate
    })
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[derive(Debug, Serialize)]
struct DetectionResultJson<'a> {
    scenes: Vec<SceneJson>,
    cuts: Vec<CutJson<'a>>,
    metrics: MetricsStoreJson<'a>,
    frames_processed: u64,
}

impl<'a> From<&'a DetectionResult> for DetectionResultJson<'a> {
    fn from(result: &'a DetectionResult) -> Self {
        Self {
            scenes: result.scenes.iter().map(SceneJson::from).collect(),
            cuts: result.cuts.iter().map(CutJson::from).collect(),
            metrics: MetricsStoreJson {
                keys: result.metrics.keys().collect(),
                rows: result.metrics.rows(),
            },
            frames_processed: result.frames_processed,
        }
    }
}

#[derive(Debug, Serialize)]
struct SceneJson {
    start: FramePositionJson,
    end: FramePositionJson,
}

impl From<&Scene> for SceneJson {
    fn from(scene: &Scene) -> Self {
        Self {
            start: FramePositionJson::from(scene.start),
            end: FramePositionJson::from(scene.end),
        }
    }
}

#[derive(Debug, Serialize)]
struct CutJson<'a> {
    position: FramePositionJson,
    detector: &'a str,
    score: Option<f32>,
}

impl<'a> From<&'a Cut> for CutJson<'a> {
    fn from(cut: &'a Cut) -> Self {
        Self {
            position: FramePositionJson::from(cut.position),
            detector: cut.detector,
            score: cut.score,
        }
    }
}

#[derive(Debug, Serialize)]
struct FramePositionJson {
    frame_index: u64,
    timestamp: TimestampJson,
}

impl From<FramePosition> for FramePositionJson {
    fn from(position: FramePosition) -> Self {
        Self {
            frame_index: position.frame_index,
            timestamp: TimestampJson {
                pts: position.timestamp.pts,
                timebase: TimebaseJson {
                    num: position.timestamp.timebase.num,
                    den: position.timestamp.timebase.den,
                },
                seconds: position.timestamp.seconds(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct TimestampJson {
    pts: i64,
    timebase: TimebaseJson,
    seconds: f64,
}

#[derive(Debug, Serialize)]
struct TimebaseJson {
    num: i32,
    den: i32,
}

#[derive(Debug, Serialize)]
struct MetricsStoreJson<'a> {
    keys: Vec<&'a str>,
    rows: &'a std::collections::BTreeMap<u64, std::collections::BTreeMap<String, f64>>,
}

#[cfg(test)]
mod tests {
    use num_rational::Rational64;
    use video_analysis_core::{FramePosition, MetricsSink, MetricsStore, Scene};

    use super::*;

    #[test]
    fn writes_stats_header_in_key_order() {
        let mut metrics = MetricsStore::default();
        metrics.set_metric(0, "z", 1.0);
        metrics.set_metric(0, "a", 2.0);
        metrics.set_metric(0, "combined.content.raw", 3.0);
        let mut out = Vec::new();
        write_stats_csv(&mut out, &metrics).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("Frame Number,Frame Index,a,combined.content.raw,z\n"));
    }

    #[test]
    fn writes_scene_csv_rows() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(10, 1));
        let scenes = vec![Scene {
            start: pos(0),
            end: pos(10),
        }];
        let mut out = Vec::new();
        write_scene_list_csv(&mut out, &scenes).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Scene Number"));
        assert!(text.contains("1,0,00:00:00.000"));
    }

    #[test]
    fn writes_html_timecodes_without_truncating() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(10, 1));
        let scenes = vec![Scene {
            start: pos(0),
            end: pos(12_345),
        }];
        let mut out = Vec::new();
        write_scene_list_html(&mut out, &scenes).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("00:20:34.500"));
    }

    #[test]
    fn writes_scene_edl_events() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(30, 1));
        let scenes = vec![Scene {
            start: pos(0),
            end: pos(30),
        }];
        let mut out = Vec::new();
        write_scene_list_edl(&mut out, &scenes).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("TITLE: Scene List"));
        assert!(text.contains("001  AX"));
        assert!(text.contains("00:00:00:00 00:00:01:00"));
    }

    #[test]
    fn writes_scene_fcp7_xml() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(30, 1));
        let scenes = vec![Scene {
            start: pos(0),
            end: pos(30),
        }];
        let mut out = Vec::new();
        write_scene_list_fcp7(&mut out, &scenes).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("<xmeml version=\"5\">"));
        assert!(text.contains("<clipitem id=\"scene-1\">"));
    }

    #[test]
    fn writes_scene_fcpx_xml() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(30, 1));
        let scenes = vec![Scene {
            start: pos(0),
            end: pos(30),
        }];
        let mut out = Vec::new();
        write_scene_list_fcpx(&mut out, &scenes).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("<fcpxml version=\"1.10\">"));
        assert!(text.contains("duration=\"1.000000s\""));
    }

    #[test]
    fn writes_scene_otio_json() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(30, 1));
        let scenes = vec![Scene {
            start: pos(0),
            end: pos(30),
        }];
        let mut out = Vec::new();
        write_scene_list_otio(&mut out, &scenes).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["OTIO_SCHEMA"], "Timeline.1");
        assert_eq!(
            value["tracks"]["children"][0]["children"][0]["name"],
            "Scene 1"
        );
    }

    #[test]
    fn writes_scene_qpfile() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(30, 1));
        let scenes = vec![Scene {
            start: pos(15),
            end: pos(30),
        }];
        let mut out = Vec::new();
        write_scene_list_qp(&mut out, &scenes).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "15 I\n");
    }

    #[test]
    fn writes_detection_json_snapshot() {
        let pos = |frame| FramePosition::from_frame_index(frame, Rational64::new(10, 1));
        let result = DetectionResult {
            scenes: vec![Scene {
                start: pos(0),
                end: pos(10),
            }],
            frames_processed: 11,
            ..DetectionResult::default()
        };
        let mut out = Vec::new();
        write_detection_result_json(&mut out, &result).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"frames_processed\": 11"));
        assert!(text.contains("\"seconds\": 1.0"));
    }
}
