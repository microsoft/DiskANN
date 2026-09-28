/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{path::Path, sync::LazyLock};

/// Normalize a string for comparison.
///
/// Steps taken:
///
/// 1. All leading trailing whitespace is removed.
/// 2. Windows line-endings `\r\n` are replaced with `\n`.
#[doc(hidden)]
pub fn normalize(s: String) -> String {
    let trimmed = s.trim().to_string();
    trimmed.replace("\r\n", "\n")
}

/// Replace all occurrences of `path` in `s` with `replacement`.
///
/// This is useful for scrubbing non-deterministic paths (e.g. temp directories) from test
/// output before comparison.
#[doc(hidden)]
pub fn scrub_path(s: String, path: &std::path::Path, replacement: &str) -> String {
    s.replace(&path.display().to_string(), replacement)
        .replace("\\", "/")
}

// There does not appear to be a supported was of checking whether backtraces are
// enabled without first actually capturing a backtrace.
static BACKTRACE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    use std::backtrace::{Backtrace, BacktraceStatus};
    Backtrace::capture().status() == BacktraceStatus::Captured
});

/// Strip the backtrace from the string representation of an [`anyhow::Error`] debug
/// diagnostic if running with backtraces enabled.
///
/// This works even if multiple [`anyhow::Error`]s are present.
#[doc(hidden)]
pub fn strip_backtrace(s: String) -> String {
    if !*BACKTRACE_ENABLED {
        return s;
    }

    // Prints with stack traces will looks something like
    // ```
    // while processing input 2 of 2
    //
    // Caused by:
    //     unknown variant `f32`, expected one of `float64`, `float32`, <snip>
    //
    // Stack backtrace:
    //    0: somestuff
    //        more stuff
    // maybe a note
    //
    // ```
    // Importantly, there is an empty line before the stacktrace starts.
    //
    // The loop simply looks for the `Stack backtrace:` line and then ignores lines from
    // that point on until an empty line is observed.
    //
    // When `Stack backtrace:` is observed and a previous empty line exists - that line is
    // removed.
    //
    // This seems to handle cases where printouts have multiple errors just fine.
    let mut in_stacktrace = false;
    let mut lines = Vec::new();
    for line in s.lines() {
        if in_stacktrace {
            if line.is_empty() {
                in_stacktrace = false;
                lines.push(line)
            }
        } else if line == "Stack backtrace:" {
            in_stacktrace = true;

            // Remove a previous empty line (if any).
            if let Some(previous) = lines.last() {
                if previous.is_empty() {
                    lines.pop();
                }
            }
        } else {
            lines.push(line);
        }
    }

    lines.join("\n")
}

//--------------//
// Crate Shared //
//--------------//

// Read the entire contents of a file to a string.
pub(crate) fn read_to_string<P: AsRef<Path>>(path: P, ctx: &str) -> String {
    match std::fs::read_to_string(path.as_ref()) {
        Ok(s) => normalize(s),
        Err(err) => panic!(
            "failed to read {} {:?} with error: {}",
            ctx,
            path.as_ref(),
            err
        ),
    }
}

const ENV: &str = "DISKANN_TEST";

// Check if `DISKANN_TEST=overwrite` is configured. Return `true` if so - otherwise
// return `false`.
//
// If `DISKANN_TEST` is set but its value is not `overwrite` - panic.
pub(crate) fn overwrite() -> bool {
    match std::env::var(ENV) {
        Ok(v) => {
            if v == "overwrite" {
                true
            } else {
                panic!(
                    "Unknown value for {}: \"{}\". Expected \"overwrite\"",
                    ENV, v
                );
            }
        }
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("Value for {} is not unicode", ENV);
        }
    }
}
