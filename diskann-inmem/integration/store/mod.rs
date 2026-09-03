/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Concurrency stress test for the in-memory [stores](diskann_inmem::integration::store).
//!
//! Reader, writer, and retirer threads hammer the epoch-based store concurrently while a
//! per-guard invariant checker verifies the store's safety guarantees:
//!
//! 1. Reads are never torn.
//! 2. A readable value is stable for the lifetime of a single reader guard.
//! 3. A slot never resurrects (`readable -> unreadable -> readable`) within one guard.
//!
//! This module exposes shared functionality that is instantiated by different implementations.

#![expect(
    clippy::unwrap_used,
    reason = "this code works mainly as an integration test"
)]

use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

use diskann_benchmark_runner::{Registry, RegistryError, utils::fmt::KeyValue};
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};
use serde::{Deserialize, Serialize};

/// Number of slots a reader inspects per guard. Kept small so guards are short-lived,
/// allowing the epoch to advance and reclamation to make progress.
const READER_WINDOW: usize = 64;

/// Number of times a reader re-reads its window within a single guard. Re-reading is what
/// exercises the value-stability and no-resurrection invariants.
const READER_PASSES: usize = 4;

/// How often (in retirer iterations) a retirer attempts to reclaim retired slots.
const RECLAIM_EVERY: u64 = 16;

mod checked;
mod invasive;

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    invasive::register(registry)?;
    checked::register(registry)?;

    Ok(())
}

///////////
// Input //
///////////

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Setup {
    /// Number of reader threads. Must be below `epoch_guard_slots`.
    readers: usize,
    /// Number of writer threads.
    writers: usize,
    /// Number of retirer threads.
    retirers: usize,
    /// Number of writable (non-frozen) slots.
    capacity: usize,
    /// The number of epoch guard slots.
    epoch_guard_slots: usize,
    /// The capacity of the freelist recycle queue capacity.
    freelist_recycle_capacity: usize,
    /// Retirers only retire while the live published population exceeds this watermark.
    low_watermark: usize,
    /// Wall-clock cap for the run, in seconds. Zero means unbounded (rely on `max_ops`).
    duration_secs: u64,
    /// Total-operation cap across all worker threads. Zero means unbounded (rely on
    /// `duration_secs`).
    max_ops: u64,
    /// Seed for the worker pseudo-random number generators.
    seed: u64,
}

impl Setup {
    fn check(&self) -> anyhow::Result<()> {
        if self.readers == 0 || self.writers == 0 {
            anyhow::bail!("`readers` and `writers` must be non-zero");
        }
        if self.readers >= self.epoch_guard_slots {
            anyhow::bail!(
                "`readers` ({}) must be below the epoch guard capacity ({})",
                self.readers,
                self.epoch_guard_slots,
            );
        }
        if self.capacity == 0 {
            anyhow::bail!("`capacity` must be non-zero");
        }
        if self.low_watermark > self.capacity {
            anyhow::bail!(
                "`low_watermark` ({}) must not exceed `capacity` ({})",
                self.low_watermark,
                self.capacity,
            );
        }
        if self.duration_secs == 0 && self.max_ops == 0 {
            anyhow::bail!("at least one of `duration_secs` or `max_ops` must be non-zero");
        }

        Ok(())
    }

    fn example() -> Self {
        Setup {
            readers: 8,
            writers: 4,
            retirers: 2,
            capacity: 4096,
            epoch_guard_slots: 256,
            freelist_recycle_capacity: 1024,
            low_watermark: 1024,
            duration_secs: 5,
            max_ops: 50_000_000,
            seed: 0xA5A5_1234_DEAD_BEEF,
        }
    }
}

impl std::fmt::Display for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("readers", &self.readers);
        kv.push("writers", &self.writers);
        kv.push("retirers", &self.retirers);
        kv.push("capacity", &self.capacity);
        kv.push("epoch_guard_slots", &self.epoch_guard_slots);
        kv.push("freelist_recycle_capacity", &self.freelist_recycle_capacity);
        kv.push("low_watermark", &self.low_watermark);
        kv.push("duration_secs", &self.duration_secs);
        kv.push("max_ops", &self.max_ops);
        kv.push("seed", &self.seed);
        write!(f, "{}", kv)
    }
}

/// A testable store.
///
/// Readers are split into a [`ReaderState`], which is used to produce a shorter lived [`Reader`].
///
/// The reason for this is two fold:
///
/// 1. We rely on [`Reader`]s being dropped to allow epochs to advance.
/// 2. A separate [`ReaderState`] allows correctness checking data structures (e.g. hash maps)
///    to be allocated once and used for the duration of the test.
///
///    Since the goal is to hammer the underlying store as hard as possible, amortizing
///    allocations makes a non-negligible difference.
trait Testable: std::fmt::Debug + Sized + Sync {
    /// The writer.
    type Writer<'a>: Writer
    where
        Self: 'a;

    /// Shared reader-state to enable allocation amortization.
    type ReaderState<'a>: ReaderState
    where
        Self: 'a;

    /// Construct a [`Writer`] into a slot. Returns `None` is an available slot could not
    /// be found.
    fn writer(&self) -> Option<Self::Writer<'_>>;

    /// Create the [`ReaderState`]. This will be called once per reader thread.
    fn reader_state<'a>(
        &'a self,
        capacity_hint: usize,
        shared: &'a Shared<Self>,
    ) -> Self::ReaderState<'a>;

    /// Attempt to retire slot `i`. Return `true` on success, otherwise return `false`.
    fn retire(&self, i: usize) -> bool;

    /// Attempt to advance the epoch and reclaim retired slots.
    ///
    /// Return `None` if we failed to advance the epoch. Otherwise, return the number of
    /// slots reclaimed.
    fn reclaim(&self) -> Option<usize>;

    /// Return the number of readable slots. Assume indices `[0..readable_slots)` are valid
    /// for reading.
    fn readable_slots(&self) -> usize;

    /// Return the number of writable slots. Assume indices `[0..writable_slots)` are valid
    /// for writing.
    fn writable_slots(&self) -> usize;
}

/// A writable slot.
trait Writer: std::fmt::Debug {
    /// Perform any writes, using `stamp` as a unique tag.
    fn write(self, stamp: u64);
}

/// Amortized shared state for reader tasks.
trait ReaderState: std::fmt::Debug {
    /// The type of the [`Reader`].
    type Reader<'a>: Reader;

    /// Attempt to run the closure `f` on a [`Reader`] into the state's parent store.
    ///
    /// Return `true` if a reader was obtained and `f` was called. Otherwise return `false`.
    ///
    /// The callback style mechanism is used to allow implementations to stack-allocate some
    /// variables. This allows validation implementations to hold onto references into the
    /// parent store by doing the following:
    ///
    /// 1. Stack allocate the internal "reader" to the store. The reader will have an epoch
    ///    guard.
    ///
    /// 2. Borrow from that reader to construct [`Self::Reader`], allowing [`Self::Reader`]
    ///    to borrow items directly from the internal reader.
    #[must_use]
    fn try_with_reader<F>(&mut self, f: F) -> bool
    where
        F: FnOnce(Self::Reader<'_>);
}

/// A read validator for a [`Testable`].
trait Reader: std::fmt::Debug {
    /// Observe the state of `i`, panicking if an invalid transition has been observed.
    fn observe(&mut self, i: usize);
}

////////////
// Output //
////////////

/// Summary statistics produced by a [`Shared`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    elapsed_secs: f64,
    reads: u64,
    acquires_ok: u64,
    acquires_fail: u64,
    retires_ok: u64,
    retires_fail: u64,
    reclaims: u64,
    /// Observed `readable -> unreadable` transitions across all reader guards.
    transitions: u64,
    /// Peak observed live (published, not-yet-retired) population.
    peak_live: usize,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("elapsed_secs", &self.elapsed_secs);
        kv.push("reads", &self.reads);
        kv.push("acquires_ok", &self.acquires_ok);
        kv.push("acquires_fail", &self.acquires_fail);
        kv.push("retires_ok", &self.retires_ok);
        kv.push("retires_fail", &self.retires_fail);
        kv.push("reclaims", &self.reclaims);
        kv.push("transitions", &self.transitions);
        kv.push("peak_live", &self.peak_live);
        write!(f, "{}", kv)
    }
}

////////////
// Shared //
////////////

struct Local<'a> {
    counter: u64,
    parent: &'a AtomicU64,
}

impl<'a> Local<'a> {
    fn new(parent: &'a AtomicU64) -> Self {
        Self { counter: 0, parent }
    }

    fn add(&mut self, by: u64) {
        self.counter += by;

        if self.counter >= 2048 {
            self.parent.fetch_add(self.counter, Relaxed);
            self.counter = 0;
        }
    }
}

impl Drop for Local<'_> {
    fn drop(&mut self) {
        self.parent.fetch_add(self.counter, Relaxed);
    }
}

struct LocalMax<'a> {
    max: usize,
    parent: &'a AtomicUsize,
}

impl<'a> LocalMax<'a> {
    fn new(parent: &'a AtomicUsize) -> Self {
        Self { max: 0, parent }
    }

    fn max(&mut self, m: usize) {
        self.max = self.max.max(m);
    }
}

impl Drop for LocalMax<'_> {
    fn drop(&mut self) {
        self.parent.fetch_max(self.max, Relaxed);
    }
}

fn run_benchmark<T>(store: T, setup: &Setup) -> anyhow::Result<Stats>
where
    T: Testable,
{
    let writable = store.writable_slots();
    let readable = store.readable_slots();
    let start = Instant::now();

    let shared = Shared {
        store,
        slots: readable,
        readable: Uniform::new(0, readable)?,
        writable: Uniform::new(0, writable)?,
        low_watermark: setup.low_watermark,
        max_ops: if setup.max_ops == 0 {
            u64::MAX
        } else {
            setup.max_ops
        },
        deadline: if setup.duration_secs == 0 {
            // Effectively unbounded; the op cap terminates the run.
            start + Duration::from_secs(u64::from(u32::MAX))
        } else {
            start + Duration::from_secs(setup.duration_secs)
        },
        stop: AtomicBool::new(false),
        violation: Mutex::new(Vec::new()),
        // Stamp 0 is reserved for the zeroed frozen point.
        stamp: AtomicU64::new(1),
        live: AtomicUsize::new(0),
        peak_live: AtomicUsize::new(0),
        ops: AtomicU64::new(0),
        reads: AtomicU64::new(0),
        acquires_ok: AtomicU64::new(0),
        acquires_fail: AtomicU64::new(0),
        retires_ok: AtomicU64::new(0),
        retires_fail: AtomicU64::new(0),
        reclaims: AtomicU64::new(0),
        transitions: AtomicU64::new(0),
    };

    std::thread::scope(|scope| {
        let shared = &shared;
        for _ in 0..setup.writers {
            scope.spawn(move || shared.writer());
        }
        for t in 0..setup.retirers {
            let seed = setup.seed ^ (0x2000_0000 + t as u64);
            scope.spawn(move || shared.retirer(seed));
        }
        for t in 0..setup.readers {
            let seed = setup.seed ^ (0x4000_0000 + t as u64);
            scope.spawn(move || shared.reader(seed));
        }
    });

    let errors: Vec<_> = std::mem::take(&mut *shared.violation.lock().unwrap());
    if !errors.is_empty() {
        anyhow::bail!("invariants violated: {:?}", errors);
    }

    let elapsed = start.elapsed();
    let stats = Stats {
        elapsed_secs: elapsed.as_secs_f64(),
        reads: shared.reads.load(Relaxed),
        acquires_ok: shared.acquires_ok.load(Relaxed),
        acquires_fail: shared.acquires_fail.load(Relaxed),
        retires_ok: shared.retires_ok.load(Relaxed),
        retires_fail: shared.retires_fail.load(Relaxed),
        reclaims: shared.reclaims.load(Relaxed),
        transitions: shared.transitions.load(Relaxed),
        peak_live: shared.peak_live.load(Relaxed),
    };

    Ok(stats)
}

/// State shared by all worker threads for the duration of a run.
#[derive(Debug)]
struct Shared<T> {
    store: T,
    slots: usize,
    readable: Uniform<usize>,
    writable: Uniform<usize>,
    low_watermark: usize,
    max_ops: u64,
    deadline: Instant,

    stop: AtomicBool,
    violation: Mutex<Vec<String>>,

    stamp: AtomicU64,
    live: AtomicUsize,
    peak_live: AtomicUsize,

    ops: AtomicU64,
    reads: AtomicU64,
    acquires_ok: AtomicU64,
    acquires_fail: AtomicU64,
    retires_ok: AtomicU64,
    retires_fail: AtomicU64,
    reclaims: AtomicU64,
    transitions: AtomicU64,
}

impl<T> Shared<T>
where
    T: Testable,
{
    /// Record an observed invariant violation and signal all workers to stop.
    fn record_violation(&self, message: String) {
        let mut slot = self.violation.lock().unwrap();
        slot.push(message);
        self.stop.store(true, Relaxed);
    }

    /// Return `true` once any termination condition is met.
    fn should_stop(&self) -> bool {
        self.stop.load(Relaxed)
            || self.ops.load(Relaxed) >= self.max_ops
            || Instant::now() >= self.deadline
    }

    //---------//
    // Workers //
    //---------//

    fn writer(&self) {
        let mut ops = Local::new(&self.ops);
        let mut acquires_ok = Local::new(&self.acquires_ok);
        let mut acquires_fail = Local::new(&self.acquires_fail);

        let mut peak_live = LocalMax::new(&self.peak_live);

        while !self.should_stop() {
            ops.add(1);
            match self.store.writer() {
                Some(writer) => {
                    let stamp = self.stamp.fetch_add(1, Relaxed);
                    writer.write(stamp);

                    let live = self.live.fetch_add(1, Relaxed) + 1;
                    peak_live.max(live);
                    acquires_ok.add(1);
                }
                None => {
                    acquires_fail.add(1);
                    std::thread::yield_now();
                }
            }
        }
    }

    fn retirer(&self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut iteration: u64 = 0;

        let mut ops = Local::new(&self.ops);
        let mut retires_ok = Local::new(&self.retires_ok);
        let mut retires_fail = Local::new(&self.retires_fail);
        let mut reclaims = Local::new(&self.reclaims);

        while !self.should_stop() {
            ops.add(1);
            iteration += 1;

            // Flow control: keep a steady readable population.
            if self.live.load(Relaxed) > self.low_watermark {
                let i = rng.sample(self.writable);
                if self.store.retire(i) {
                    self.live.fetch_sub(1, Relaxed);
                    retires_ok.add(1);
                } else {
                    retires_fail.add(1);
                }
            }

            if iteration.is_multiple_of(RECLAIM_EVERY)
                && let Some(reclaimed) = self.store.reclaim()
            {
                reclaims.add(reclaimed as u64);
            }

            std::thread::yield_now();
        }
    }

    fn reader(&self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let slots = self.slots;
        let window = READER_WINDOW.min(slots);

        let mut ops = Local::new(&self.ops);
        let mut reads = Local::new(&self.reads);

        let mut reader_state = self.store.reader_state(window, self);

        while !self.should_stop() {
            ops.add(1);

            let succeeded = reader_state.try_with_reader(|mut reader| {
                let start = rng.sample(self.readable);
                for _ in 0..READER_PASSES {
                    for k in 0..window {
                        let i = (start + k) % slots;
                        reader.observe(i);
                        reads.add(1);
                    }
                }
            });

            // All guard slots are occupied; back off and retry.
            if !succeeded {
                std::thread::yield_now();
            };
        }
    }
}
