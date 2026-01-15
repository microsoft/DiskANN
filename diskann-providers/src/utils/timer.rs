/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::time::{Duration, Instant};

use diskann_platform::*;
#[cfg(feature = "perf_test")]
use opentelemetry::{
    KeyValue, global,
    trace::{Tracer, get_active_span},
};

#[derive(Clone)]
pub struct Timer {
    check_point: Instant,
    cycles: Option<u64>,
    start_process_time: Option<u64>,
    start_system_time: Option<u64>,
    number_of_processors: Option<u64>,
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    pub fn new() -> Timer {
        let cycles = get_process_cycle_time();
        Timer {
            check_point: Instant::now(),
            cycles,
            start_process_time: get_process_time(),
            start_system_time: get_system_time(),
            number_of_processors: get_number_of_processors(),
        }
    }

    pub fn reset(&mut self) {
        self.check_point = Instant::now();
        self.cycles = get_process_cycle_time();
        self.start_process_time = get_process_time();
        self.start_system_time = get_system_time();
    }

    pub fn elapsed(&self) -> Duration {
        Instant::now().duration_since(self.check_point)
    }

    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }

    pub fn elapsed_gcycles(&self) -> f32 {
        let cur_cycles = get_process_cycle_time();
        if let (Some(cur_cycles), Some(cycles)) = (cur_cycles, self.cycles) {
            let spent_cycles =
                ((cur_cycles - cycles) as f64 * 1.0f64) / (1024 * 1024 * 1024) as f64;
            return spent_cycles as f32;
        }

        0.0
    }

    // Returns the average CPU time in percents (100% means that only one core is used)
    pub fn get_average_cpu_time_in_percents(&self) -> f64 {
        let cur_process_time = get_process_time();
        let cur_system_time = get_system_time();
        if let (
            Some(cur_process_time),
            Some(cur_system_time),
            Some(start_process_time),
            Some(start_system_time),
            Some(number_of_processors),
        ) = (
            cur_process_time,
            cur_system_time,
            self.start_process_time,
            self.start_system_time,
            self.number_of_processors,
        ) {
            let process_time_delta = cur_process_time - start_process_time;
            let system_time_delta = cur_system_time - start_system_time;

            if system_time_delta > 0 {
                return (process_time_delta as f64) / (system_time_delta as f64)
                    * number_of_processors as f64
                    * 100f64;
            }
        }

        0.0
    }

    pub fn get_peak_memory_usage(&self) -> f32 {
        let memory_in_bytes = get_peak_workingset_size();
        if let Some(bytes) = memory_in_bytes {
            let memory_in_gbs = bytes as f64 / (1024 * 1024 * 1024) as f64;
            return memory_in_gbs as f32;
        }

        0.0
    }

    pub fn elapsed_seconds_for_step(&self, step: &str) -> String {
        #[cfg(feature = "perf_test")]
        {
            let tracer = global::tracer("");
            tracer.in_span(format!("InMemIndexBuild-{:?}", step), |_context| {
                get_active_span(|span| {
                    span.set_attribute(KeyValue::new("duration_seconds", self.elapsed_seconds()));
                    span.set_attribute(KeyValue::new(
                        "elapsed_cycles",
                        self.elapsed_gcycles() as f64,
                    ));
                    span.set_attribute(KeyValue::new(
                        "cpu_time",
                        self.get_average_cpu_time_in_percents(),
                    ));
                    span.set_attribute(KeyValue::new(
                        "peak_memory_usage",
                        self.get_peak_memory_usage() as f64,
                    ));
                });
            });
        }
        format!(
            "Time for {}: {:.3} seconds, {:.3}B cycles, {:.3}% CPU time, peak memory {:.3} GBs",
            step,
            self.elapsed_seconds(),
            self.elapsed_gcycles(),
            self.get_average_cpu_time_in_percents(),
            self.get_peak_memory_usage()
        )
    }
}

#[cfg(test)]
mod timer_tests {
    use std::{thread, time};

    use super::*;

    #[test]
    #[allow(clippy::if_same_then_else)]
    fn test_new() {
        let timer = Timer::new();
        if cfg!(windows) {
            assert!(timer.cycles.is_some());
        } else if cfg!(target_os = "linux") {
            assert!(timer.cycles.is_some());
        } else {
            panic!("No timer::test_new defined for current configuration");
        }
    }

    #[test]
    fn test_reset() {
        let mut timer = Timer::new();
        let checkpoint_before_reset = timer.check_point;

        for _ in 0..10 {
            timer.reset();
            let checkpoint_after_reset = timer.check_point;
            if checkpoint_after_reset > checkpoint_before_reset {
                return;
            }

            thread::sleep(time::Duration::from_millis(10));
        }

        timer.reset();

        assert!(
            timer.check_point > checkpoint_before_reset,
            "Timer::reset() did not update check_point"
        );
    }

    #[test]
    fn test_elapsed() {
        let timer = Timer::new();
        let t0 = timer.elapsed();

        for _ in 0..10 {
            let t1 = timer.elapsed();
            if t1 > t0 {
                return;
            }

            thread::sleep(time::Duration::from_millis(10));
        }

        assert!(
            timer.elapsed() > t0,
            "Timer::elapsed() did not increase over time"
        );
    }

    #[test]
    fn test_elapsed_seconds_for_step() {
        let timer = Timer::new();
        let log = timer.elapsed_seconds_for_step("Test step");
        assert!(log.contains("Time for Test step:"));
    }

    #[test]
    fn test_get_average_cpu_time_in_percents() {
        let timer = Timer::new();
        thread::sleep(time::Duration::from_millis(10));
        assert!(timer.get_average_cpu_time_in_percents() >= 0f64);
    }

    #[test]
    fn test_get_peak_memory_usage() {
        let timer = Timer::new();
        let peak_memory_usage = timer.get_peak_memory_usage();
        assert!(peak_memory_usage >= 0.0);
    }
}
