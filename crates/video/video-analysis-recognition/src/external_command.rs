use std::collections::BTreeMap;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use video_analysis_core::{DetectError, Result, TextSegment, VideoFrame};

use model_runtime::{DownloadedModel, ModelRuntimeBackend, ModelTask};

use crate::{RawPrediction, TextModelBackend, VisionModelBackend};

#[derive(Debug, Clone)]
/// Data type for external command model.
pub struct ExternalCommandModel {
    command: PathBuf,
    args: Vec<String>,
    model: DownloadedModel,
}

impl ExternalCommandModel {
    /// Creates a new value.
    pub fn new(command: impl Into<PathBuf>, model: DownloadedModel) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            model,
        }
    }

    /// Returns arg.
    pub fn arg(mut self, value: impl Into<String>) -> Self {
        self.args.push(value.into());
        self
    }

    /// Returns args.
    pub fn args(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(values.into_iter().map(Into::into));
        self
    }

    /// Returns model.
    pub fn model(&self) -> &DownloadedModel {
        &self.model
    }

    /// Returns persistent.
    pub fn persistent(self) -> PersistentExternalCommandModel {
        PersistentExternalCommandModel::new(self.command, self.model).args(self.args)
    }

    fn run(&mut self, input: ExternalModelInput<'_>) -> Result<Vec<RawPrediction>> {
        let request = external_model_request(&self.model, input);
        let payload = serde_json::to_vec(&request)
            .map_err(|err| DetectError::Source(format!("failed to encode model request: {err}")))?;

        let mut child = Command::new(&self.command)
            .args(&self.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| DetectError::Source("model command stdin is unavailable".to_string()))?;
        let write_result = stdin.write_all(&payload);
        drop(stdin);

        let output = child.wait_with_output()?;
        if let Err(err) = write_result {
            if err.kind() != io::ErrorKind::BrokenPipe {
                return Err(err.into());
            }
        }
        if !output.status.success() {
            return Err(DetectError::Source(format!(
                "model command `{}` failed with status {}: {}",
                self.command.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let response: ExternalModelResponse =
            serde_json::from_slice(&output.stdout).map_err(|err| {
                DetectError::Source(format!(
                    "model command `{}` returned invalid JSON: {err}",
                    self.command.display()
                ))
            })?;
        Ok(response.predictions)
    }
}

/// Data type for persistent external command model.
pub struct PersistentExternalCommandModel {
    command: PathBuf,
    args: Vec<String>,
    model: DownloadedModel,
    child: Option<PersistentCommandChild>,
}

struct PersistentCommandChild {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl PersistentExternalCommandModel {
    /// Creates a new value.
    pub fn new(command: impl Into<PathBuf>, model: DownloadedModel) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            model,
            child: None,
        }
    }

    /// Returns arg.
    pub fn arg(mut self, value: impl Into<String>) -> Self {
        self.args.push(value.into());
        self
    }

    /// Returns args.
    pub fn args(mut self, values: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(values.into_iter().map(Into::into));
        self
    }

    /// Returns model.
    pub fn model(&self) -> &DownloadedModel {
        &self.model
    }

    /// Returns stop.
    pub fn stop(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            drop(child.stdin);
            let status = child.child.wait()?;
            if !status.success() {
                return Err(DetectError::Source(format!(
                    "persistent model command `{}` exited with status {status}",
                    self.command.display()
                )));
            }
        }
        Ok(())
    }

    fn run(&mut self, input: ExternalModelInput<'_>) -> Result<Vec<RawPrediction>> {
        let request = external_model_request(&self.model, input);
        let payload = serde_json::to_vec(&request)
            .map_err(|err| DetectError::Source(format!("failed to encode model request: {err}")))?;
        let command = self.command.display().to_string();
        let child = self.child()?;

        child.stdin.write_all(&payload)?;
        child.stdin.write_all(b"\n")?;
        child.stdin.flush()?;

        let mut line = String::new();
        let bytes = child.stdout.read_line(&mut line)?;
        if bytes == 0 {
            return Err(DetectError::Source(format!(
                "persistent model command `{command}` closed stdout"
            )));
        }
        let response: ExternalModelResponse = serde_json::from_str(&line).map_err(|err| {
            DetectError::Source(format!(
                "persistent model command `{command}` returned invalid JSON: {err}"
            ))
        })?;
        Ok(response.predictions)
    }

    fn child(&mut self) -> Result<&mut PersistentCommandChild> {
        if self.child.is_none() {
            let mut child = Command::new(&self.command)
                .args(&self.args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()?;
            let stdin = child.stdin.take().ok_or_else(|| {
                DetectError::Source("persistent model command stdin is unavailable".to_string())
            })?;
            let stdout = child.stdout.take().ok_or_else(|| {
                DetectError::Source("persistent model command stdout is unavailable".to_string())
            })?;
            self.child = Some(PersistentCommandChild {
                child,
                stdin,
                stdout: BufReader::new(stdout),
            });
        }
        Ok(self.child.as_mut().expect("persistent model child exists"))
    }
}

impl Drop for PersistentExternalCommandModel {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.child.kill();
            let _ = child.child.wait();
        }
    }
}

impl VisionModelBackend for ExternalCommandModel {
    fn task(&self) -> ModelTask {
        self.model.spec.task.clone()
    }

    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::External
    }

    fn predict_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<RawPrediction>> {
        self.run(ExternalModelInput::VideoFrame {
            width: frame.width,
            height: frame.height,
            pixel_format: match frame.pixel_format {
                video_analysis_core::PixelFormat::Rgb24 => "rgb24",
                video_analysis_core::PixelFormat::Bgr24 => "bgr24",
            },
            stride: frame.stride,
            data_base64: BASE64.encode(frame.data),
        })
    }
}

impl VisionModelBackend for PersistentExternalCommandModel {
    fn task(&self) -> ModelTask {
        self.model.spec.task.clone()
    }

    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::External
    }

    fn predict_frame(&mut self, frame: &VideoFrame<'_>) -> Result<Vec<RawPrediction>> {
        self.run(ExternalModelInput::VideoFrame {
            width: frame.width,
            height: frame.height,
            pixel_format: match frame.pixel_format {
                video_analysis_core::PixelFormat::Rgb24 => "rgb24",
                video_analysis_core::PixelFormat::Bgr24 => "bgr24",
            },
            stride: frame.stride,
            data_base64: BASE64.encode(frame.data),
        })
    }
}

impl TextModelBackend for ExternalCommandModel {
    fn task(&self) -> ModelTask {
        self.model.spec.task.clone()
    }

    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::External
    }

    fn predict_text(&mut self, segment: &TextSegment<'_>) -> Result<Vec<RawPrediction>> {
        self.run(ExternalModelInput::Text {
            text: segment.text,
            language: segment.language,
            is_final: segment.is_final,
        })
    }
}

impl TextModelBackend for PersistentExternalCommandModel {
    fn task(&self) -> ModelTask {
        self.model.spec.task.clone()
    }

    fn runtime_backend(&self) -> ModelRuntimeBackend {
        ModelRuntimeBackend::External
    }

    fn predict_text(&mut self, segment: &TextSegment<'_>) -> Result<Vec<RawPrediction>> {
        self.run(ExternalModelInput::Text {
            text: segment.text,
            language: segment.language,
            is_final: segment.is_final,
        })
    }
}

fn external_model_request<'a>(
    model: &'a DownloadedModel,
    input: ExternalModelInput<'a>,
) -> ExternalModelRequest<'a> {
    ExternalModelRequest {
        task: model.spec.task.as_protocol_str(),
        model: ExternalModelInfo {
            name: model.spec.name.clone(),
            repo_id: model.spec.repo_id_value().unwrap_or_default().to_string(),
            revision: model.spec.revision_value().unwrap_or_default().to_string(),
            files: model
                .files
                .iter()
                .map(|(key, path)| (key.clone(), path.to_string_lossy().into_owned()))
                .collect(),
        },
        input,
    }
}

#[derive(Debug, Serialize)]
struct ExternalModelRequest<'a> {
    task: &'a str,
    model: ExternalModelInfo,
    input: ExternalModelInput<'a>,
}

#[derive(Debug, Serialize)]
struct ExternalModelInfo {
    name: String,
    repo_id: String,
    revision: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExternalModelInput<'a> {
    VideoFrame {
        width: u32,
        height: u32,
        pixel_format: &'static str,
        stride: usize,
        data_base64: String,
    },
    Text {
        text: &'a str,
        language: Option<&'a str>,
        is_final: bool,
    },
}

#[derive(Debug, Deserialize)]
struct ExternalModelResponse {
    #[serde(default)]
    predictions: Vec<RawPrediction>,
}
