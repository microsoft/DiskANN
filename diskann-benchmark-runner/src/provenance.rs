/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Identifying information about the run that produced a set of results.
//!
//! A results file is only useful for comparison if you can tell what produced it. Recording
//! the revision, the machine, and the wall-clock time alongside the numbers means a result
//! that looks anomalous months later can be attributed rather than guessed at.
//!
//! Every field is best-effort and independently optional. Provenance is metadata about a
//! benchmark, not part of it, so a missing hostname or a build outside a git checkout
//! records `null` rather than failing a run that may have taken hours.

use std::{
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

/// Identifying information about the process producing a set of benchmark results.
///
/// Obtained via [`Provenance::current`], which captures the values once and reuses them, so
/// every result within a single run reports a consistent timestamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Version of the benchmark runner the binary was built against.
    pub version: String,

    /// Full commit hash of the source that was compiled, if built inside a git checkout.
    pub git_sha: Option<String>,

    /// Whether tracked files were modified relative to [`git_sha`](Self::git_sha).
    ///
    /// Advisory. This is sampled when the runner is compiled, so it can understate a tree
    /// edited afterwards without triggering a rebuild. A `true` here means the results are
    /// not reproducible from `git_sha` alone; a `false` is good evidence but not a proof.
    pub git_dirty: Option<bool>,

    /// Name of the machine the benchmark ran on.
    ///
    /// Benchmark numbers are only comparable across runs on the same hardware, which makes
    /// this the field most likely to explain an unexpected difference.
    pub host: Option<String>,

    /// Start of the run, in seconds since the Unix epoch.
    ///
    /// Recorded alongside [`utc`](Self::utc) because the numeric form sorts and subtracts
    /// without a date parser.
    pub unix_time: u64,

    /// Start of the run as an ISO 8601 timestamp in UTC, e.g. `2024-05-17T09:31:04Z`.
    pub utc: String,
}

impl Provenance {
    /// Provenance for the current process.
    ///
    /// The result is computed on first use and cached, so the timestamp marks roughly when
    /// the run began rather than when any particular result was serialized.
    pub fn current() -> &'static Self {
        static CURRENT: OnceLock<Provenance> = OnceLock::new();
        CURRENT.get_or_init(|| {
            let unix_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());

            Self {
                version: env!("CARGO_PKG_VERSION").to_string(),
                git_sha: option_env!("DISKANN_GIT_SHA").map(str::to_string),
                git_dirty: option_env!("DISKANN_GIT_DIRTY").map(|s| s == "true"),
                host: host(),
                unix_time,
                utc: format_utc(unix_time),
            }
        })
    }
}

/// Best-effort name of the current machine.
///
/// Deliberately avoids a dependency for something this peripheral. `COMPUTERNAME` is set by
/// Windows; `HOSTNAME` is set by many Unix shells but is not exported by all of them, hence
/// the fall back to the file the kernel exposes.
fn host() -> Option<String> {
    let from_env = ["COMPUTERNAME", "HOSTNAME"]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok());

    from_env
        .chain(std::fs::read_to_string("/etc/hostname"))
        .map(|value| value.trim().to_string())
        .find(|value| !value.is_empty())
}

/// Render seconds since the Unix epoch as an ISO 8601 UTC timestamp.
///
/// Leap seconds are not represented in Unix time, so this is an exact conversion.
fn format_utc(unix_time: u64) -> String {
    let (year, month, day) = civil_from_days((unix_time / 86_400) as i64);
    let seconds_of_day = unix_time % 86_400;

    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds_of_day / 3_600,
        (seconds_of_day / 60) % 60,
        seconds_of_day % 60,
    )
}

/// Convert days since 1970-01-01 into a proleptic Gregorian calendar date.
///
/// This is Howard Hinnant's `civil_from_days`, which shifts the year to start in March so
/// the leap day lands at the end of the 146097-day, 400-year cycle and needs no special
/// casing.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the epoch to 0000-03-01, the start of a 400-year era.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097); // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365; // [0, 399]
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100); // [0, 365]

    // With March as month 0, month lengths follow an exact linear pattern.
    let shifted_month = (5 * day_of_year + 2) / 153; // [0, 11]
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32; // [1, 31]
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32; // [1, 12]

    // Undo the March-based year once January and February are back at the start.
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_renders_as_the_start_of_1970() {
        assert_eq!(format_utc(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn time_of_day_is_split_into_hours_minutes_and_seconds() {
        assert_eq!(format_utc(86_399), "1970-01-01T23:59:59Z");
        assert_eq!(format_utc(3_661), "1970-01-01T01:01:01Z");
    }

    #[test]
    fn known_timestamps_round_trip_against_reference_values() {
        // Spot values cross-checked against `date -u -d @<n>`, chosen to exercise the
        // century and 400-year leap rules the calendar conversion has to get right.
        for (unix_time, expected) in [
            (951_782_400, "2000-02-29T00:00:00Z"), // 2000 is a leap year (divisible by 400)
            (1_709_164_800, "2024-02-29T00:00:00Z"), // 2024 is an ordinary leap year
            (4_107_542_400, "2100-03-01T00:00:00Z"), // 2100 is not (divisible by 100)
            (1_700_000_000, "2023-11-14T22:13:20Z"),
            (2_147_483_647, "2038-01-19T03:14:07Z"), // 32-bit time_t overflow
        ] {
            assert_eq!(format_utc(unix_time), expected, "for {unix_time}");
        }
    }

    #[test]
    fn every_day_of_a_leap_year_advances_by_exactly_one() {
        // Walks 2024 day by day, which covers the leap day and every month boundary. A
        // conversion that drifts anywhere in the year fails the day-count check.
        let start = 1_704_067_200; // 2024-01-01T00:00:00Z
        let dates: Vec<_> = (0..366)
            .map(|day| civil_from_days((start / 86_400) + day))
            .collect();

        assert_eq!(dates.first(), Some(&(2024, 1, 1)));
        assert_eq!(dates.last(), Some(&(2024, 12, 31)));
        assert!(dates.contains(&(2024, 2, 29)));
        assert_eq!(
            dates.iter().collect::<std::collections::HashSet<_>>().len(),
            366,
            "every day in the year should be distinct",
        );
    }

    #[test]
    fn current_provenance_is_stable_within_a_process() {
        // Callers rely on every result in a run carrying the same timestamp, which only
        // holds if the value is captured once.
        assert_eq!(Provenance::current(), Provenance::current());
        assert!(std::ptr::eq(Provenance::current(), Provenance::current()));
    }

    #[test]
    fn current_provenance_reports_the_crate_version() {
        assert_eq!(Provenance::current().version, env!("CARGO_PKG_VERSION"));
        assert!(!Provenance::current().version.is_empty());
    }

    #[test]
    fn current_provenance_round_trips_through_json() {
        let value = serde_json::to_value(Provenance::current()).unwrap();
        let parsed: Provenance = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(&parsed, Provenance::current());

        // The key set is the part downstream tooling depends on, so pin it explicitly.
        let mut keys: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "git_dirty",
                "git_sha",
                "host",
                "unix_time",
                "utc",
                "version"
            ],
        );
    }

    #[test]
    fn missing_revision_information_is_recorded_as_null() {
        // Building outside a git checkout is supported and must serialize cleanly rather
        // than being omitted, so consumers can tell "unknown" from "absent field".
        let unknown = Provenance {
            version: "0.0.0".to_string(),
            git_sha: None,
            git_dirty: None,
            host: None,
            unix_time: 0,
            utc: format_utc(0),
        };

        let value = serde_json::to_value(&unknown).unwrap();
        assert!(value["git_sha"].is_null());
        assert!(value["git_dirty"].is_null());
        assert!(value["host"].is_null());
        assert_eq!(value["utc"], "1970-01-01T00:00:00Z");
    }
}
