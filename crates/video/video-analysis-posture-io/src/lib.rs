#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use video_analysis_core::{DetectError, Result};
use video_analysis_posture::{Keypoint, PoseEstimate, Skeleton, StickFigure3d};

/// Reads coco keypoints JSON.
pub fn read_coco_keypoints_json(path: impl AsRef<Path>) -> Result<Vec<PoseEstimate>> {
    let data = fs::read(path.as_ref())?;
    let document: CocoKeypointsDocument = serde_json::from_slice(&data)
        .map_err(|err| invalid_argument(format!("failed to parse COCO JSON: {err}")))?;
    let skeleton = Skeleton::coco_17();
    document
        .annotations
        .into_iter()
        .map(|annotation| annotation.into_pose_estimate(&skeleton))
        .collect()
}

/// Writes coco keypoints JSON.
pub fn write_coco_keypoints_json(path: impl AsRef<Path>, poses: &[PoseEstimate]) -> Result<()> {
    let skeleton = Skeleton::coco_17();
    let document = CocoKeypointsDocument {
        annotations: poses
            .iter()
            .enumerate()
            .map(|(index, pose)| {
                CocoAnnotation::from_pose_estimate(index as u64 + 1, &skeleton, pose)
            })
            .collect(),
        categories: vec![CocoCategory {
            id: 1,
            name: "person".to_string(),
            keypoints: skeleton.keypoints.clone(),
            skeleton: skeleton
                .edges
                .iter()
                .filter_map(|edge| {
                    let from = skeleton
                        .keypoints
                        .iter()
                        .position(|name| name == &edge.from)?;
                    let to = skeleton
                        .keypoints
                        .iter()
                        .position(|name| name == &edge.to)?;
                    Some(vec![from as u32 + 1, to as u32 + 1])
                })
                .collect(),
        }],
    };
    let encoded = serde_json::to_vec_pretty(&document)
        .map_err(|err| invalid_argument(format!("failed to encode COCO JSON: {err}")))?;
    fs::write(path, encoded)?;
    Ok(())
}

/// Writes stick figure PLY.
pub fn write_stick_figure_ply(path: impl AsRef<Path>, figure: &StickFigure3d) -> Result<()> {
    figure.validate()?;
    let mut output = String::new();
    output.push_str("ply\nformat ascii 1.0\n");
    output.push_str(&format!("element vertex {}\n", figure.keypoints.len()));
    output.push_str("property float x\nproperty float y\nproperty float z\n");
    let edges = indexed_segments(figure);
    output.push_str(&format!("element edge {}\n", edges.len()));
    output.push_str("property int vertex1\nproperty int vertex2\nend_header\n");
    for keypoint in &figure.keypoints {
        output.push_str(&format!(
            "{} {} {}\n",
            keypoint.position.x, keypoint.position.y, keypoint.position.z
        ));
    }
    for (a, b) in edges {
        output.push_str(&format!("{a} {b}\n"));
    }
    fs::write(path, output)?;
    Ok(())
}

/// Writes stick figure gltf.
pub fn write_stick_figure_gltf(path: impl AsRef<Path>, figure: &StickFigure3d) -> Result<()> {
    figure.validate()?;
    let edges = indexed_segments(figure);
    let mut bytes = Vec::new();
    for keypoint in &figure.keypoints {
        bytes.extend_from_slice(&keypoint.position.x.to_le_bytes());
        bytes.extend_from_slice(&keypoint.position.y.to_le_bytes());
        bytes.extend_from_slice(&keypoint.position.z.to_le_bytes());
    }
    let mut index_bytes = Vec::new();
    for (a, b) in &edges {
        index_bytes.extend_from_slice(&(*a as u32).to_le_bytes());
        index_bytes.extend_from_slice(&(*b as u32).to_le_bytes());
    }
    let index_offset = bytes.len() as u32;
    bytes.extend_from_slice(&index_bytes);

    let document = MinimalLineGltf {
        asset: GltfAsset {
            version: "2.0".to_string(),
        },
        buffers: vec![GltfBuffer {
            byte_length: bytes.len() as u32,
            uri: format!(
                "data:application/octet-stream;base64,{}",
                BASE64.encode(bytes)
            ),
        }],
        buffer_views: vec![
            GltfBufferView {
                buffer: 0,
                byte_offset: 0,
                byte_length: (figure.keypoints.len() * 12) as u32,
                target: Some(34962),
            },
            GltfBufferView {
                buffer: 0,
                byte_offset: index_offset,
                byte_length: index_bytes.len() as u32,
                target: Some(34963),
            },
        ],
        accessors: vec![
            GltfAccessor {
                buffer_view: 0,
                byte_offset: 0,
                component_type: 5126,
                count: figure.keypoints.len() as u32,
                accessor_type: "VEC3".to_string(),
            },
            GltfAccessor {
                buffer_view: 1,
                byte_offset: 0,
                component_type: 5125,
                count: (edges.len() * 2) as u32,
                accessor_type: "SCALAR".to_string(),
            },
        ],
        meshes: vec![GltfMesh {
            primitives: vec![GltfPrimitive {
                attributes: GltfAttributes { position: Some(0) },
                indices: Some(1),
                mode: Some(1),
            }],
        }],
        scenes: vec![GltfScene { nodes: vec![0] }],
        nodes: vec![GltfNode { mesh: Some(0) }],
        scene: Some(0),
    };
    let encoded = serde_json::to_vec_pretty(&document)
        .map_err(|err| invalid_argument(format!("failed to encode glTF JSON: {err}")))?;
    fs::write(path, encoded)?;
    Ok(())
}

fn indexed_segments(figure: &StickFigure3d) -> Vec<(usize, usize)> {
    let index_by_name = figure
        .keypoints
        .iter()
        .enumerate()
        .map(|(index, keypoint)| (keypoint.name.clone(), index))
        .collect::<BTreeMap<_, _>>();
    figure
        .skeleton
        .edges
        .iter()
        .filter_map(|edge| {
            let from = index_by_name.get(&edge.from)?;
            let to = index_by_name.get(&edge.to)?;
            Some((*from, *to))
        })
        .collect()
}

fn invalid_argument(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(message.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CocoKeypointsDocument {
    #[serde(default)]
    annotations: Vec<CocoAnnotation>,
    #[serde(default)]
    categories: Vec<CocoCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CocoAnnotation {
    id: u64,
    #[serde(default = "default_category_id")]
    category_id: u64,
    #[serde(default)]
    keypoints: Vec<f32>,
    score: Option<f32>,
}

impl CocoAnnotation {
    fn into_pose_estimate(self, skeleton: &Skeleton) -> Result<PoseEstimate> {
        let expected = skeleton.keypoints.len() * 3;
        if self.keypoints.len() != expected {
            return Err(invalid_argument(format!(
                "COCO annotation {} expected {} keypoint values, found {}",
                self.id,
                expected,
                self.keypoints.len()
            )));
        }
        let keypoints = skeleton
            .keypoints
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let base = index * 3;
                let x = self.keypoints[base];
                let y = self.keypoints[base + 1];
                let visibility = self.keypoints[base + 2];
                let mut keypoint = Keypoint::new(name.clone(), x, y)?;
                keypoint.visible = Some(visibility > 0.0);
                if visibility > 0.0 {
                    keypoint.score = Some(visibility.min(1.0));
                }
                Ok(keypoint)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut pose = PoseEstimate::new(keypoints)?.id(self.id.to_string());
        if let Some(score) = self.score {
            pose = pose.score(score)?;
        }
        Ok(pose)
    }

    fn from_pose_estimate(id: u64, skeleton: &Skeleton, pose: &PoseEstimate) -> Self {
        let mut encoded = Vec::with_capacity(skeleton.keypoints.len() * 3);
        for name in &skeleton.keypoints {
            if let Some(keypoint) = pose.keypoint(name) {
                encoded.push(keypoint.x);
                encoded.push(keypoint.y);
                encoded.push(if keypoint.visible.unwrap_or(true) {
                    keypoint.score.unwrap_or(1.0)
                } else {
                    0.0
                });
            } else {
                encoded.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
        }
        Self {
            id,
            category_id: 1,
            keypoints: encoded,
            score: pose.score,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CocoCategory {
    id: u64,
    name: String,
    keypoints: Vec<String>,
    skeleton: Vec<Vec<u32>>,
}

fn default_category_id() -> u64 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MinimalLineGltf {
    asset: GltfAsset,
    buffers: Vec<GltfBuffer>,
    #[serde(rename = "bufferViews")]
    buffer_views: Vec<GltfBufferView>,
    accessors: Vec<GltfAccessor>,
    meshes: Vec<GltfMesh>,
    scenes: Vec<GltfScene>,
    nodes: Vec<GltfNode>,
    scene: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfAsset {
    version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfBuffer {
    #[serde(rename = "byteLength")]
    byte_length: u32,
    uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfBufferView {
    buffer: usize,
    #[serde(rename = "byteOffset")]
    byte_offset: u32,
    #[serde(rename = "byteLength")]
    byte_length: u32,
    target: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfAccessor {
    #[serde(rename = "bufferView")]
    buffer_view: usize,
    #[serde(rename = "byteOffset")]
    byte_offset: u32,
    #[serde(rename = "componentType")]
    component_type: u32,
    count: u32,
    #[serde(rename = "type")]
    accessor_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfMesh {
    primitives: Vec<GltfPrimitive>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfPrimitive {
    attributes: GltfAttributes,
    indices: Option<usize>,
    mode: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfAttributes {
    #[serde(rename = "POSITION")]
    position: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfScene {
    nodes: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GltfNode {
    mesh: Option<usize>,
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use video_analysis_posture::Point3;
    use video_analysis_posture::{Keypoint3d, Pose3dEstimate};

    #[test]
    fn coco_json_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("poses.json");
        let pose = PoseEstimate::new([
            Keypoint::new("nose", 10.0, 20.0).unwrap(),
            Keypoint::new("left_eye", 9.0, 19.0).unwrap(),
            Keypoint::new("right_eye", 11.0, 19.0).unwrap(),
            Keypoint::new("left_ear", 8.0, 19.0).unwrap(),
            Keypoint::new("right_ear", 12.0, 19.0).unwrap(),
            Keypoint::new("left_shoulder", 8.0, 30.0).unwrap(),
            Keypoint::new("right_shoulder", 12.0, 30.0).unwrap(),
            Keypoint::new("left_elbow", 7.0, 40.0).unwrap(),
            Keypoint::new("right_elbow", 13.0, 40.0).unwrap(),
            Keypoint::new("left_wrist", 6.0, 50.0).unwrap(),
            Keypoint::new("right_wrist", 14.0, 50.0).unwrap(),
            Keypoint::new("left_hip", 9.0, 50.0).unwrap(),
            Keypoint::new("right_hip", 11.0, 50.0).unwrap(),
            Keypoint::new("left_knee", 9.0, 70.0).unwrap(),
            Keypoint::new("right_knee", 11.0, 70.0).unwrap(),
            Keypoint::new("left_ankle", 9.0, 90.0).unwrap(),
            Keypoint::new("right_ankle", 11.0, 90.0).unwrap(),
        ])
        .unwrap();
        write_coco_keypoints_json(&path, &[pose]).unwrap();
        let loaded = read_coco_keypoints_json(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].keypoints.len(), 17);
    }

    #[test]
    fn stick_figure_exports_write_files() {
        let dir = tempdir().unwrap();
        let pose = Pose3dEstimate::new([
            Keypoint3d::new("left_shoulder", Point3::new(0.0, 0.0, 0.0)).unwrap(),
            Keypoint3d::new("right_shoulder", Point3::new(1.0, 0.0, 0.0)).unwrap(),
            Keypoint3d::new("left_elbow", Point3::new(0.0, -1.0, 0.0)).unwrap(),
            Keypoint3d::new("right_elbow", Point3::new(1.0, -1.0, 0.0)).unwrap(),
        ])
        .unwrap();
        let figure = pose
            .to_stick_figure(
                Skeleton::new([
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                ])
                .edge("left_shoulder", "right_shoulder")
                .edge("left_shoulder", "left_elbow")
                .edge("right_shoulder", "right_elbow"),
            )
            .unwrap();

        let ply = dir.path().join("figure.ply");
        write_stick_figure_ply(&ply, &figure).unwrap();
        assert!(ply.exists());

        let gltf = dir.path().join("figure.gltf");
        write_stick_figure_gltf(&gltf, &figure).unwrap();
        assert!(gltf.exists());
    }
}
