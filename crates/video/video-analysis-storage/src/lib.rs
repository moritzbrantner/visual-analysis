#![doc = include_str!("../README.md")]

pub mod surface;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use video_analysis_dataset::{AnalysisDataset, DatasetMetadata, DatasetRecord};

#[derive(Debug, Error)]
/// Variants describing storage error.
pub enum StorageError {
    #[error("I/O error: {0}")]
    /// The I/O variant.
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    /// The JSON variant.
    Json(String),
    #[error("invalid manifest: {0}")]
    /// The invalid manifest variant.
    InvalidManifest(String),
    #[error("unsupported dataset format: {0}")]
    /// The unsupported format variant.
    UnsupportedFormat(String),
}

/// Type alias for result.
pub type Result<T> = std::result::Result<T, StorageError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for dataset manifest.
pub struct DatasetManifest {
    /// The schema version value.
    pub schema_version: u32,
    /// The dataset name value.
    pub dataset_name: Option<String>,
    /// The record count value.
    pub record_count: u64,
    /// The record counts value.
    pub record_counts: BTreeMap<String, u64>,
    /// The files value.
    pub files: Vec<DatasetFile>,
    /// The attributes value.
    pub attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// Data type for dataset file.
pub struct DatasetFile {
    /// Filesystem path for this value.
    pub path: String,
    /// The format value.
    pub format: DatasetFileFormat,
    /// The records value.
    pub records: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Variants describing dataset file format.
pub enum DatasetFileFormat {
    /// The JSON lines variant.
    JsonLines,
    #[serde(other)]
    /// The unsupported variant.
    Unsupported,
}

/// Writes jsonl.
pub fn write_jsonl(path: impl AsRef<Path>, dataset: &AnalysisDataset) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for record in &dataset.records {
        let line = serde_json::to_string(record).map_err(json_error)?;
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Reads jsonl.
pub fn read_jsonl(path: impl AsRef<Path>) -> Result<AnalysisDataset> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut dataset = AnalysisDataset::empty();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<DatasetRecord>(&line)
            .map_err(|err| StorageError::Json(format!("line {}: {err}", line_index + 1)))?;
        dataset.push(record);
    }
    Ok(dataset)
}

/// Writes dataset JSON.
pub fn write_dataset_json(path: impl AsRef<Path>, dataset: &AnalysisDataset) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), dataset).map_err(json_error)
}

/// Reads dataset JSON.
pub fn read_dataset_json(path: impl AsRef<Path>) -> Result<AnalysisDataset> {
    let file = File::open(path)?;
    serde_json::from_reader(BufReader::new(file)).map_err(json_error)
}

/// Builds manifest.
pub fn build_manifest(
    dataset: &AnalysisDataset,
    records_path: impl AsRef<Path>,
) -> DatasetManifest {
    let mut record_counts = BTreeMap::new();
    for record in &dataset.records {
        *record_counts.entry(record.kind().to_string()).or_default() += 1;
    }

    DatasetManifest {
        schema_version: dataset.metadata.schema_version,
        dataset_name: dataset.metadata.name.clone(),
        record_count: dataset.records.len() as u64,
        record_counts,
        files: vec![DatasetFile {
            path: records_path.as_ref().to_string_lossy().into_owned(),
            format: DatasetFileFormat::JsonLines,
            records: dataset.records.len() as u64,
        }],
        attributes: dataset.metadata.attributes.clone(),
    }
}

/// Writes manifest.
pub fn write_manifest(path: impl AsRef<Path>, manifest: &DatasetManifest) -> Result<()> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), manifest).map_err(json_error)
}

/// Reads manifest.
pub fn read_manifest(path: impl AsRef<Path>) -> Result<DatasetManifest> {
    let file = File::open(path)?;
    serde_json::from_reader(BufReader::new(file)).map_err(json_error)
}

/// Writes dataset dir.
pub fn write_dataset_dir(
    dir: impl AsRef<Path>,
    dataset: &AnalysisDataset,
) -> Result<DatasetManifest> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    let records_path = dir.join("records.jsonl");
    write_jsonl(&records_path, dataset)?;
    let manifest = build_manifest(dataset, "records.jsonl");
    write_manifest(dir.join("manifest.json"), &manifest)?;
    Ok(manifest)
}

/// Reads dataset dir.
pub fn read_dataset_dir(dir: impl AsRef<Path>) -> Result<AnalysisDataset> {
    let dir = dir.as_ref();
    let manifest = read_manifest(dir.join("manifest.json"))?;
    let file = manifest.files.first().ok_or_else(|| {
        StorageError::InvalidManifest("manifest must contain at least one file".to_string())
    })?;
    match file.format {
        DatasetFileFormat::JsonLines => {}
        DatasetFileFormat::Unsupported => {
            return Err(StorageError::UnsupportedFormat(file.path.clone()));
        }
    }

    let mut dataset = read_jsonl(dir.join(&file.path))?;
    dataset.metadata = DatasetMetadata {
        schema_version: manifest.schema_version,
        name: manifest.dataset_name,
        source: None,
        created_at: None,
        attributes: manifest.attributes,
    };
    if dataset.records.len() as u64 != file.records
        || dataset.records.len() as u64 != manifest.record_count
    {
        return Err(StorageError::InvalidManifest(format!(
            "manifest expected {} records, loaded {}",
            manifest.record_count,
            dataset.records.len()
        )));
    }
    Ok(dataset)
}

fn json_error(error: serde_json::Error) -> StorageError {
    StorageError::Json(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use num_rational::Rational64;
    use tempfile::tempdir;
    use video_analysis_core::{FramePosition, Observation, ObservationKind};
    use video_analysis_dataset::{AnalysisDataset, DatasetMetadata, ObservationRecord};

    use super::*;

    fn position(frame_index: u64) -> FramePosition {
        FramePosition::from_frame_index(frame_index, Rational64::new(30, 1))
    }

    fn dataset() -> AnalysisDataset {
        let mut dataset = AnalysisDataset::new(DatasetMetadata {
            name: Some("test".to_string()),
            ..DatasetMetadata::default()
        });
        dataset.push(DatasetRecord::Observation(
            ObservationRecord::from_observation(
                Observation::new("objects", ObservationKind::Object)
                    .at_frame(position(1))
                    .label("person"),
            ),
        ));
        dataset.push(DatasetRecord::Feature(
            video_analysis_dataset::FeatureRecord::number("score", 0.8),
        ));
        dataset
    }

    #[test]
    fn jsonl_round_trips_mixed_dataset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let dataset = dataset();

        write_jsonl(&path, &dataset).unwrap();
        let loaded = read_jsonl(&path).unwrap();

        assert_eq!(loaded.records, dataset.records);
    }

    #[test]
    fn dataset_dir_round_trips_records_and_manifest() {
        let dir = tempdir().unwrap();
        let dataset = dataset();

        let manifest = write_dataset_dir(dir.path(), &dataset).unwrap();
        let loaded = read_dataset_dir(dir.path()).unwrap();

        assert_eq!(manifest.record_count, 2);
        assert_eq!(loaded.metadata.name.as_deref(), Some("test"));
        assert_eq!(loaded.records, dataset.records);
    }

    #[test]
    fn manifest_record_counts_match_records() {
        let dataset = dataset();
        let manifest = build_manifest(&dataset, "records.jsonl");

        assert_eq!(manifest.record_count, 2);
        assert_eq!(manifest.record_counts["observation"], 1);
        assert_eq!(manifest.record_counts["feature"], 1);
    }

    #[test]
    fn empty_jsonl_loads_empty_dataset() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        File::create(&path).unwrap();

        let loaded = read_jsonl(&path).unwrap();
        assert!(loaded.records.is_empty());
    }

    #[test]
    fn malformed_jsonl_reports_line_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "{{bad json").unwrap();

        let err = read_jsonl(&path).unwrap_err();
        assert!(matches!(err, StorageError::Json(message) if message.contains("line 1")));
    }

    #[test]
    fn unsupported_manifest_format_reports_error() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("records.csv"), "nope").unwrap();
        fs::write(
            dir.path().join("manifest.json"),
            r#"{
                "schema_version": 1,
                "dataset_name": null,
                "record_count": 0,
                "record_counts": {},
                "files": [{"path": "records.csv", "format": "csv", "records": 0}],
                "attributes": {}
            }"#,
        )
        .unwrap();

        let err = read_dataset_dir(dir.path()).unwrap_err();
        assert!(matches!(err, StorageError::UnsupportedFormat(_)));
    }
}
