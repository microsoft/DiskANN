/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::LazyLock;

/// Normalize a string for comparison.
///
/// Steps taken:
///
/// 1. All leading trailing whitespace is removed.
/// 2. Windows line-endings `\n\r` are replaced with `\n`.
#[doc(hidden)]
pub fn normalize(s: String) -> String {
    let trimmed = s.trim().to_string();
    trimmed.replace("\r\n", "\n")
}

// There does not appear to be a supported was of checking whether backtraces are
// enabled without first actually capturing a backtrace.
static BACKTRACE_ENABLED: LazyLock<bool> = LazyLock::new(|| {
    use std::backtrace::{Backtrace, BacktraceStatus};
    Backtrace::capture().status() == BacktraceStatus::Captured
});

/// Strip the backtrace from the string representation of an [`anyhow::Error`] debug
/// diagnostic if running with backtraces enabled.
#[doc(hidden)]
pub fn strip_backtrace(s: String) -> String {
    if !*BACKTRACE_ENABLED {
        return s;
    }

    // Split into lines until we see `Stack backtrace`, then drop the empty
    //
    // Prints with stack traces will looks something like
    // ```
    // while processing input 2 of 2
    //
    // Caused by:
    //     unknown variant `f32`, expected one of `float64`, `float32`, <snip>
    //
    // Stack backtrace:
    //    0:
    // ```
    // This works by splitting the output into lines - looking for the keyword
    // `Stack backtrace` and taking all lines up to that point.
    let mut stacktrace_found = false;
    let lines: Vec<_> = s
        .lines()
        .take_while(|l| {
            stacktrace_found = *l == "Stack backtrace:";
            !stacktrace_found
        })
        .collect();

    if lines.is_empty() {
        String::new()
    } else if stacktrace_found {
        // When `anyhow` inserts a backtrace - it separates the body of the error from
        // the stack trace with a newline. This strips that newline.
        //
        // Indexing is okay because we've already handled the empty case.
        lines[..lines.len() - 1].join("\n")
    } else {
        // No stacktrace found - do not strip a trailing empty line.
        lines.join("\n")
    }
}
