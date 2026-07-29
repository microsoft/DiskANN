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

/// The value substituted for non-deterministic provenance fields.
#[doc(hidden)]
pub const SCRUBBED: &str = "$SCRUBBED";

/// Replace the values inside every `provenance` object of a JSON document with
/// [`SCRUBBED`], returning the re-serialized document.
///
/// [`Provenance`](crate::Provenance) records the revision, host and wall-clock time of a
/// run, so an expected results file containing real values would differ on every run and on
/// every machine. Blanking the values while keeping the keys means the expected files still
/// assert that provenance is emitted, and with which fields.
///
/// Both sides of a comparison are passed through this, so the re-serialization is
/// self-cancelling. Input that does not parse as JSON, or that carries no provenance, is
/// returned verbatim rather than reformatted.
#[doc(hidden)]
pub fn scrub_provenance(s: String) -> String {
    // Returns whether anything was scrubbed. Note the use of `|=` rather than any
    // short-circuiting operator: every block in the document has to be visited, so the
    // recursion must not stop at the first one found.
    fn scrub(value: &mut serde_json::Value) -> bool {
        let mut found = false;
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map.iter_mut() {
                    match child.as_object_mut() {
                        Some(fields) if key == "provenance" => {
                            fields
                                .values_mut()
                                .for_each(|field| *field = SCRUBBED.into());
                            found = true;
                        }
                        _ => found |= scrub(child),
                    }
                }
            }
            serde_json::Value::Array(items) => {
                for item in items.iter_mut() {
                    found |= scrub(item);
                }
            }
            _ => {}
        }
        found
    }

    let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&s) else {
        return s;
    };
    if !scrub(&mut value) {
        return s;
    }
    serde_json::to_string_pretty(&value).map_or(s, normalize)
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn scrub_value(value: serde_json::Value) -> serde_json::Value {
        serde_json::from_str(&scrub_provenance(value.to_string())).unwrap()
    }

    #[test]
    fn provenance_values_are_blanked_but_the_keys_are_kept() {
        // Keeping the keys is the point: the expected files still fail if a field is
        // dropped or renamed, they just stop depending on its value.
        let scrubbed = scrub_value(serde_json::json!([{
            "input": {"dim": 8},
            "results": {"qps": 1234.5},
            "provenance": {"version": "0.53.0", "git_sha": "abc123", "host": "somebox"},
        }]));

        assert_eq!(scrubbed[0]["input"]["dim"], 8);
        assert_eq!(scrubbed[0]["results"]["qps"], 1234.5);
        assert_eq!(scrubbed[0]["provenance"]["version"], SCRUBBED);
        assert_eq!(scrubbed[0]["provenance"]["git_sha"], SCRUBBED);
        assert_eq!(scrubbed[0]["provenance"]["host"], SCRUBBED);
    }

    #[test]
    fn runs_differing_only_in_provenance_compare_equal() {
        let run = |sha: &str, host: &str, time: u64| {
            scrub_provenance(
                serde_json::json!([{
                    "input": {"dim": 8},
                    "results": {"qps": 1234.5},
                    "provenance": {"git_sha": sha, "host": host, "unix_time": time},
                }])
                .to_string(),
            )
        };

        assert_eq!(run("abc123", "boxa", 1), run("def456", "boxb", 2));
    }

    #[test]
    fn differences_outside_provenance_still_compare_unequal() {
        let run = |qps: f64| {
            scrub_provenance(
                serde_json::json!([{
                    "results": {"qps": qps},
                    "provenance": {"git_sha": "abc123"},
                }])
                .to_string(),
            )
        };

        assert_ne!(run(1234.5), run(1234.6));
    }

    #[test]
    fn nested_and_absent_provenance_are_both_handled() {
        // The scrubber walks the whole document rather than assuming a fixed depth, and a
        // document without provenance has to survive the round trip untouched.
        let scrubbed = scrub_value(serde_json::json!({
            "outer": [{"inner": {"provenance": {"host": "somebox"}}}],
            "untouched": {"host": "somebox"},
        }));

        assert_eq!(
            scrubbed["outer"][0]["inner"]["provenance"]["host"],
            SCRUBBED
        );
        assert_eq!(scrubbed["untouched"]["host"], "somebox");
    }

    #[test]
    fn a_provenance_key_that_is_not_an_object_is_left_alone() {
        // Benchmarks are free to use the word for their own purposes; only the runner's
        // provenance block, which is always an object, should be rewritten.
        let scrubbed = scrub_value(serde_json::json!({"provenance": "hand written"}));
        assert_eq!(scrubbed["provenance"], "hand written");
    }

    #[test]
    fn documents_without_provenance_are_returned_verbatim() {
        // Re-serializing would reorder keys and restyle the file, so output that has no
        // provenance to blank must come back byte for byte.
        let text = r#"[{"status":"error","tolerance":{"error_when_checked":false}}]"#.to_string();
        assert_eq!(scrub_provenance(text.clone()), text);
    }

    #[test]
    fn every_provenance_block_in_a_document_is_scrubbed() {
        // One block per result entry, so stopping at the first would leave the rest of the
        // file machine-dependent.
        let scrubbed = scrub_value(serde_json::json!([
            {"provenance": {"host": "boxa"}},
            {"provenance": {"host": "boxb"}},
            {"nested": {"provenance": {"host": "boxc"}}},
        ]));

        assert_eq!(scrubbed[0]["provenance"]["host"], SCRUBBED);
        assert_eq!(scrubbed[1]["provenance"]["host"], SCRUBBED);
        assert_eq!(scrubbed[2]["nested"]["provenance"]["host"], SCRUBBED);
    }

    #[test]
    fn non_json_input_is_returned_unchanged() {
        // `stdout.txt` goes through the same comparison path and is not JSON.
        let text = "running 2 tests\nall passed".to_string();
        assert_eq!(scrub_provenance(text.clone()), text);
    }
}
