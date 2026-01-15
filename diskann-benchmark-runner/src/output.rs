/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Return the default implementation of [`Output`] that sends:
///
/// * Prints to `stdout`.
/// * Progress Bars to `stderr`.
pub fn default() -> DefaultOutput {
    DefaultOutput::new()
}

/// To enable testing and output redirection, the output stream that tests print to is hidden
/// behind this `Output` trait, which consists of two parts:
///
/// * `sink`: A `&mut dyn std::io::Write` which is the target for all test prints.
/// * `draw_target`: The target that all progress bars should use.
pub trait Output {
    fn sink(&mut self) -> &mut dyn std::io::Write;
    fn draw_target(&self) -> indicatif::ProgressDrawTarget;
}

/// This allows `&mut dyn Output` to be used as the receiver of the `write!` macro.
impl std::io::Write for &mut dyn Output {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.sink().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.sink().flush()
    }
}

/// A default output that sends:
///
/// * Prints to `stdout`
/// * Progress Bars to `stderr`.
#[derive(Debug)]
pub struct DefaultOutput(std::io::Stdout);

impl DefaultOutput {
    /// Construct a new [`DefaultOutput`].
    pub fn new() -> Self {
        Self(std::io::stdout())
    }
}

impl Default for DefaultOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl Output for DefaultOutput {
    fn sink(&mut self) -> &mut dyn std::io::Write {
        &mut self.0
    }

    // Note: Progress bars use `stderr` as the default draw target.
    fn draw_target(&self) -> indicatif::ProgressDrawTarget {
        indicatif::ProgressDrawTarget::stderr()
    }
}

/// An output that suppresses all prints and progress bars.
#[derive(Debug)]
pub struct Sink(std::io::Sink);

impl Sink {
    /// Construct a new [`Sink`] output.
    pub fn new() -> Self {
        Self(std::io::sink())
    }
}

impl Default for Sink {
    fn default() -> Self {
        Self::new()
    }
}

impl Output for Sink {
    fn sink(&mut self) -> &mut dyn std::io::Write {
        &mut self.0
    }

    // Note: Progress bars use `stderr` as the default draw target.
    fn draw_target(&self) -> indicatif::ProgressDrawTarget {
        indicatif::ProgressDrawTarget::hidden()
    }
}

/// The `Memory` allows test output to be captured directly in a buffer.
///
/// Integration tests can use this so each tests runs in its own environment.
///
/// Progress bars are suppressed.
#[derive(Debug)]
pub struct Memory(Vec<u8>);

impl Memory {
    /// Construct a new [`Memory`] with an empty buffer.
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Consume `self`, returning the interior buffer with all messages that have been
    /// written to `self`.
    pub fn into_inner(self) -> Vec<u8> {
        self.0
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

impl Output for Memory {
    fn sink(&mut self) -> &mut dyn std::io::Write {
        &mut self.0
    }

    // Hide the progress bar when piping to a buffer.
    fn draw_target(&self) -> indicatif::ProgressDrawTarget {
        indicatif::ProgressDrawTarget::hidden()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;

    #[test]
    fn test_memory() {
        let mut buf = Memory::new();
        {
            let mut output: &mut dyn Output = &mut buf;
            writeln!(output, "hello world").unwrap();
            writeln!(output, "test: {}", 10).unwrap();
            output.flush().unwrap();

            assert!(output.draw_target().is_hidden());
        }
        let bytes = buf.into_inner();
        let message = str::from_utf8(&bytes).unwrap();
        let mut lines = message.lines();
        let first = lines.next().unwrap();
        assert_eq!(first, "hello world");

        let second = lines.next().unwrap();
        assert_eq!(second, "test: 10");

        assert!(lines.next().is_none());
    }

    #[test]
    fn test_default() {
        let mut d = default();
        let mut s = &mut d as &mut dyn Output;

        // This is not a reliable test when running under tooling like LLVM-cov.
        // assert!(!s.draw_target().is_hidden());

        writeln!(s, "test").unwrap();
    }

    #[test]
    fn test_sink() {
        let mut d = Sink::new();
        let mut s = &mut d as &mut dyn Output;

        assert!(s.draw_target().is_hidden());
        writeln!(s, "test").unwrap();
    }
}
