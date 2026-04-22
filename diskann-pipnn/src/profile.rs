/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Phase-profiling timers for the PiPNN bench suite.
//!
//! Gated behind the `bench-profiling` Cargo feature. When the feature is
//! disabled (the default), all timers compile to zero-cost no-op stubs —
//! `PhaseTimer::start` returns a ZST, its `Drop` is empty, and the snapshot
//! functions return empty vectors. Production builds are byte-identical to
//! builds that never referenced this module.
//!
//! Usage:
//! ```ignore
//! {
//!     let _t = diskann_pipnn::profile::PhaseTimer::start("gemm");
//!     sgemm_aat(a, m, k, out);
//! } // timer records on scope exit
//! ```

#[cfg(feature = "bench-profiling")]
mod inner {
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};

    type Local = Arc<Mutex<HashMap<&'static str, Duration>>>;

    fn registry() -> &'static Mutex<Vec<Local>> {
        static R: OnceLock<Mutex<Vec<Local>>> = OnceLock::new();
        R.get_or_init(|| Mutex::new(Vec::new()))
    }

    thread_local! {
        static LOCAL: Local = {
            let arc: Local = Arc::new(Mutex::new(HashMap::new()));
            registry().lock().push(arc.clone());
            arc
        };
    }

    pub struct PhaseTimer {
        name: &'static str,
        t0: Instant,
    }

    impl PhaseTimer {
        #[inline]
        pub fn start(name: &'static str) -> Self {
            Self {
                name,
                t0: Instant::now(),
            }
        }
    }

    impl Drop for PhaseTimer {
        fn drop(&mut self) {
            let elapsed = self.t0.elapsed();
            let name = self.name;
            LOCAL.with(|local| {
                let mut map = local.lock();
                *map.entry(name).or_insert(Duration::ZERO) += elapsed;
            });
        }
    }

    /// Collect and merge per-thread accumulators. Result is sorted by name.
    pub fn take_snapshot() -> Vec<(&'static str, Duration)> {
        let mut merged: HashMap<&'static str, Duration> = HashMap::new();
        for arc in registry().lock().iter() {
            let map = arc.lock();
            for (k, v) in map.iter() {
                *merged.entry(*k).or_insert(Duration::ZERO) += *v;
            }
        }
        let mut result: Vec<_> = merged.into_iter().collect();
        result.sort_by_key(|(k, _)| *k);
        result
    }

    /// Clear all per-thread accumulators. Call between phases to get clean
    /// per-phase breakdowns.
    pub fn reset() {
        for arc in registry().lock().iter() {
            arc.lock().clear();
        }
    }
}

#[cfg(not(feature = "bench-profiling"))]
mod inner {
    use std::time::Duration;

    pub struct PhaseTimer;

    impl PhaseTimer {
        #[inline(always)]
        pub fn start(_name: &'static str) -> Self {
            Self
        }
    }

    #[inline(always)]
    pub fn take_snapshot() -> Vec<(&'static str, Duration)> {
        Vec::new()
    }

    #[inline(always)]
    pub fn reset() {}
}

pub use inner::*;
