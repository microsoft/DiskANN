/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::fmt;

#[cfg(feature = "perf_test")]
use opentelemetry::{
    global,
    trace::{get_active_span, Tracer},
    KeyValue,
};
use tracing::info;

use diskann_providers::utils::Timer;

/// Target for logging latency events.
pub const LATENCY_LOG_TARGET: &str = "latency_event";

mod scenario {
    pub const DISK_INDEX_BUILD_SCENARIO: &str = "DiskIndexBuild";
}

#[derive(Debug)]
pub enum DiskIndexBuildCheckpoint {
    PqConstruction,
    InmemIndexBuild,
    DiskLayout,
}

impl fmt::Display for DiskIndexBuildCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
pub enum BuildMergedVamanaIndexCheckpoint {
    PartitionData,
    BuildIndicesOnShards,
    MergeIndices,
}

impl fmt::Display for BuildMergedVamanaIndexCheckpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub struct PerfLogger {
    inner: Option<PerfLoggerInner>,
}

impl PerfLogger {
    pub fn new<T: std::fmt::Display>(scenario: T, enable_logging: bool) -> Self {
        if enable_logging {
            let inner = PerfLoggerInner {
                scenario: scenario.to_string(),
                timer: Timer::new(),
            };
            Self { inner: Some(inner) }
        } else {
            Self { inner: None }
        }
    }

    pub fn new_disk_index_build_logger() -> Self {
        Self::new(scenario::DISK_INDEX_BUILD_SCENARIO, true)
    }

    /// Starts or restarts the timer for the perf logger.
    pub fn start(&mut self) {
        if let Some(inner) = &mut self.inner {
            inner.start();
        }
    }

    /// Logs the time elapsed since the last checkpoint or since the logger was created.
    ///
    /// This method calculates the elapsed time, logs it with the provided checkpoint name,
    /// and then resets the start time to the current time. If logging is not enabled,
    /// this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - A string slice that holds the name of the checkpoint.
    ///
    /// # Examples
    ///
    /// ```
    /// use diskann_disk::utils::instrumentation::PerfLogger;
    ///
    /// let mut logger = PerfLogger::new("Scenario".to_string(), true);
    /// logger.log_checkpoint("Checkpoint1");
    /// ```
    pub fn log_checkpoint<T: fmt::Display>(&mut self, checkpoint: T) {
        if let Some(inner) = &mut self.inner {
            inner.log_checkpoint(checkpoint);
        }
    }

    /// Returns whether logging is enabled for the perf logger.
    pub fn log_enabled(&self) -> bool {
        self.inner.is_some()
    }
}

struct PerfLoggerInner {
    scenario: String,
    timer: Timer,
}

impl PerfLoggerInner {
    /// Starts or restarts the timer for the perf logger.
    fn start(&mut self) {
        self.timer.reset();
    }

    /// Logs the time elapsed since the last checkpoint or since the logger was created.
    ///
    /// This method calculates the elapsed time, logs it with the provided checkpoint name,
    /// and then resets the start time to the current time. If logging is not enabled,
    /// this method does nothing.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - A string slice that holds the name of the checkpoint.
    ///
    /// # Examples
    ///
    /// ```
    /// use diskann_disk::utils::instrumentation::PerfLogger;
    ///
    /// let mut logger = PerfLogger::new("Scenario".to_string(), true);
    /// logger.log_checkpoint("Checkpoint1");
    /// ```
    fn log_checkpoint<T: fmt::Display>(&mut self, checkpoint: T) {
        info!( target: LATENCY_LOG_TARGET,
            "Time for {} [Checkpoint: {}] completed: {:.3} seconds, {:.3}B cycles, {:.3}% CPU time, peak memory {:.3} GBs",
            self.scenario,
            checkpoint,
            self.timer.elapsed().as_secs_f32(),
            self.timer.elapsed_gcycles(),
            self.timer.get_average_cpu_time_in_percents(),
            self.timer.get_peak_memory_usage()
        );

        #[cfg(feature = "perf_test")]
        {
            let tracer = global::tracer("");
            tracer.in_span(format!("{}-{}", self.scenario, checkpoint), |_context| {
                get_active_span(|span| {
                    span.set_attribute(KeyValue::new(
                        "duration_seconds",
                        self.timer.elapsed().as_secs_f64(),
                    ));
                    span.set_attribute(KeyValue::new(
                        "elapsed_cycles",
                        self.timer.elapsed_gcycles() as f64,
                    ));
                    span.set_attribute(KeyValue::new(
                        "cpu_time",
                        self.timer.get_average_cpu_time_in_percents(),
                    ));
                    span.set_attribute(KeyValue::new(
                        "peak_memory_usage",
                        self.timer.get_peak_memory_usage() as f64,
                    ));
                });
            });
        }

        self.timer.reset();
    }
}

#[cfg(test)]
mod perf_logger_tests {
    use super::*;

    #[test]
    fn test_log() {
        let scenario = "test";
        let mut logger = PerfLogger::new(scenario, true);
        assert!(logger.log_enabled());
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);
        logger.start();
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);
    }

    #[test]
    fn test_log_disabled() {
        let scenario = "test";
        let mut logger = PerfLogger::new(scenario, false);
        assert!(!logger.log_enabled());
        logger.log_checkpoint(DiskIndexBuildCheckpoint::PqConstruction);
        logger.start();
        logger.log_checkpoint(DiskIndexBuildCheckpoint::InmemIndexBuild);
    }
}
