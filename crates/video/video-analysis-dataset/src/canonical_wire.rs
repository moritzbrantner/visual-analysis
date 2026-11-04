//! Versioned dataset persistence adapter; Foundation owns rational-time semantics.

use media_core::{Timebase, Timestamp};
use serde_json::{Map, Value};

use super::{AnalysisDataset, AnalysisDatasetBorrowed, AnalysisDatasetOwned, DatasetRecord, TimestampRecord};

pub(crate) const CANONICAL_SCHEMA_VERSION: u32 = 3;

pub(crate) fn encode_dataset(dataset: &AnalysisDataset) -> Result<Value, String> {
    let mut value = serde_json::to_value(AnalysisDatasetBorrowed {
        metadata: &dataset.metadata,
        records: &dataset.records,
    })
    .map_err(|error| error.to_string())?;
    encode_timestamps(&mut value)?;
    Ok(value)
}

pub(crate) fn decode_dataset(mut value: Value) -> Result<AnalysisDataset, String> {
    decode_timestamps(&mut value, true)?;
    let wire: AnalysisDatasetOwned =
        serde_json::from_value(value).map_err(|error| error.to_string())?;
    Ok(AnalysisDataset {
        metadata: wire.metadata,
        records: wire.records,
    })
}

pub(crate) fn encode_record(record: &DatasetRecord) -> Result<Value, String> {
    let mut value = serde_json::to_value(record).map_err(|error| error.to_string())?;
    encode_timestamps(&mut value)?;
    Ok(value)
}

pub(crate) fn decode_record(mut value: Value) -> Result<DatasetRecord, String> {
    decode_timestamps(&mut value, false)?;
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn legacy_timestamp(map: &Map<String, Value>) -> bool {
    map.len() == 4
        && ["pts", "timebase_num", "timebase_den", "seconds"]
            .iter()
            .all(|key| map.contains_key(*key))
}

fn canonical_timestamp(map: &Map<String, Value>) -> bool {
    map.len() == 2 && map.contains_key("pts") && map.contains_key("timebase")
}

fn encode_timestamps(value: &mut Value) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            for item in items {
                encode_timestamps(item)?;
            }
        }
        Value::Object(map) if legacy_timestamp(map) => {
            let legacy: TimestampRecord =
                serde_json::from_value(Value::Object(map.clone())).map_err(|e| e.to_string())?;
            let timebase = Timebase::try_new(legacy.timebase_num, legacy.timebase_den)
                .map_err(|e| e.to_string())?;
            let canonical = Timestamp::try_new(legacy.pts, timebase)
                .map_err(|e| e.to_string())?;
            if !legacy.seconds.is_finite() || legacy.seconds != canonical.seconds() {
                return Err(format!(
                    "legacy timestamp seconds disagree with rational PTS at {} ticks",
                    legacy.pts
                ));
            }
            *value = serde_json::to_value(canonical).map_err(|e| e.to_string())?;
        }
        Value::Object(map) => {
            for field in map.values_mut() {
                encode_timestamps(field)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn decode_timestamps(value: &mut Value, require_canonical: bool) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            for item in items {
                decode_timestamps(item, require_canonical)?;
            }
        }
        Value::Object(map) if require_canonical && legacy_timestamp(map) => {
            return Err("schema v3 cannot contain legacy timestamps with derived seconds".into());
        }
        Value::Object(map) if canonical_timestamp(map) => {
            let canonical: Timestamp =
                serde_json::from_value(Value::Object(map.clone())).map_err(|e| e.to_string())?;
            canonical.validate().map_err(|e| e.to_string())?;
            *value = serde_json::to_value(TimestampRecord::from(canonical))
                .map_err(|e| e.to_string())?;
        }
        Value::Object(map) => {
            for field in map.values_mut() {
                decode_timestamps(field, require_canonical)?;
            }
        }
        _ => {}
    }
    Ok(())
}
