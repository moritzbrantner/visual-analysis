//! Shared pipe ownership for streaming audio/video decoders.
use std::collections::VecDeque;
use std::io::{self, Read};
use std::process::{Child, ChildStdout};
use std::thread::JoinHandle;

const STDERR_LIMIT: usize = 64 * 1024;

pub(crate) struct DecoderProcess {
    child: Child,
    stdout: Option<ChildStdout>,
    stderr: Option<JoinHandle<io::Result<Vec<u8>>>>,
    terminal: Option<Result<(), String>>,
}

impl DecoderProcess {
    pub(crate) fn new(child: Child) -> io::Result<Self> {
        // Establish cleanup ownership before any fallible pipe/thread setup.
        let mut process = Self {
            child,
            stdout: None,
            stderr: None,
            terminal: None,
        };
        process.stdout = Some(
            process
                .child
                .stdout
                .take()
                .ok_or_else(|| io::Error::other("ffmpeg stdout pipe was not available"))?,
        );
        let stderr = process
            .child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("ffmpeg stderr pipe was not available"))?;
        process.stderr = Some(
            std::thread::Builder::new()
                .name("ffmpeg-stderr".into())
                .spawn(move || drain_tail(stderr))?,
        );
        Ok(process)
    }

    fn finish(&mut self) -> io::Result<()> {
        if self.terminal.is_none() {
            self.stdout.take();
            let status = self.child.wait();
            let diagnostic = self
                .stderr
                .take()
                .map(|thread| {
                    thread
                        .join()
                        .map_err(|_| io::Error::other("ffmpeg stderr reader panicked"))
                        .and_then(|result| result)
                })
                .transpose();
            self.terminal = Some(match (status, diagnostic) {
                (Ok(status), Ok(_)) if status.success() => Ok(()),
                (Ok(status), Ok(bytes)) => Err(format!(
                    "ffmpeg exited with {status}: {}",
                    String::from_utf8_lossy(&bytes.unwrap_or_default()).trim()
                )),
                (Err(error), _) | (_, Err(error)) => {
                    Err(format!("ffmpeg completion failed: {error}"))
                }
            });
        }
        match self.terminal.as_ref().unwrap() {
            Ok(()) => Ok(()),
            Err(error) => Err(io::Error::other(error.clone())),
        }
    }
}

impl Read for DecoderProcess {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.terminal.is_some() {
            self.finish()?;
            return Ok(0);
        }
        let count = self.stdout.as_mut().unwrap().read(output)?;
        if count == 0 {
            self.finish()?;
        }
        Ok(count)
    }
}

impl Drop for DecoderProcess {
    fn drop(&mut self) {
        // Closing the output and killing the child unblocks both pipe readers.
        self.stdout.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(stderr) = self.stderr.take() {
            let _ = stderr.join();
        }
    }
}

fn drain_tail(mut input: impl Read) -> io::Result<Vec<u8>> {
    let mut tail = VecDeque::with_capacity(STDERR_LIMIT);
    let mut buffer = [0_u8; 8192];
    loop {
        let count = match input.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        let excess = (tail.len() + count).saturating_sub(STDERR_LIMIT);
        tail.drain(..excess);
        tail.extend(&buffer[..count]);
    }
    Ok(tail.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_retention_is_bounded_and_keeps_the_tail() {
        let mut input = vec![b'x'; STDERR_LIMIT * 4];
        input.extend_from_slice(b"last error");
        let tail = drain_tail(input.as_slice()).unwrap();
        assert_eq!(tail.len(), STDERR_LIMIT);
        assert!(tail.ends_with(b"last error"));
    }

    #[cfg(unix)]
    fn process(script: &str) -> DecoderProcess {
        use std::process::{Command, Stdio};
        DecoderProcess::new(
            Command::new("sh")
                .args(["-c", script])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    #[cfg(unix)]
    fn a_complete_frame_before_failure_does_not_hide_the_exit_error() {
        let mut process = process("printf abc; printf 'decoder failed' >&2; exit 7");
        let mut bytes = [0; 3];
        process.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"abc");
        for _ in 0..2 {
            let error = process.read(&mut bytes).unwrap_err().to_string();
            assert!(
                error.contains('7') && error.contains("decoder failed"),
                "{error}"
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn stderr_larger_than_a_pipe_is_drained_before_reading_stdout() {
        // An undrained pipe would deadlock. The workflow has a process timeout;
        // recv_timeout supplies a focused diagnosis without wall-clock speed gating.
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut child = process("head -c 1048576 /dev/zero >&2; printf abc");
            let mut output = Vec::new();
            let result = child.read_to_end(&mut output).map(|_| output);
            let _ = send.send(result);
        });
        assert_eq!(
            receive
                .recv_timeout(std::time::Duration::from_secs(15))
                .expect("stderr drain deadlocked")
                .unwrap(),
            b"abc"
        );
    }

    #[test]
    #[cfg(unix)]
    fn successful_eof_is_repeatable() {
        let mut child = process("printf abc");
        let mut output = Vec::new();
        child.read_to_end(&mut output).unwrap();
        assert_eq!(output, b"abc");
        assert_eq!(child.read(&mut [0]).unwrap(), 0);
    }
}
