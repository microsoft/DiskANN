/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Concurrency stress test for the in-memory [`Store`](diskann_inmem::integration::store::Store).
//!
//! Reader, writer, and retirer threads hammer the epoch-based store concurrently while a
//! per-guard invariant checker verifies the store's safety guarantees:
//!
//! 1. Reads are never torn.
//! 2. A readable value is stable for the lifetime of a single reader guard.
//! 3. A slot never resurrects (`readable -> unreadable -> readable`) within one guard.

#![expect(
    clippy::unwrap_used,
    reason = "this code works mainly as an integration test"
)]

use std::{
    collections::HashMap,
    io::Write,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

use diskann_benchmark_runner::{
    Benchmark, Checker, Checkpoint, Input, Output,
    benchmark::{MatchContext, Score},
    utils::fmt::KeyValue,
};
use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};
use serde::{Deserialize, Serialize};

use diskann_inmem::integration::store::{Config, Store};

/// Maximum number of concurrent reader guards supported by the epoch registry.
const GUARD_CAPACITY: usize = 256;

/// Number of slots a reader inspects per guard. Kept small so guards are short-lived,
/// allowing the epoch to advance and reclamation to make progress.
const READER_WINDOW: usize = 64;

/// Number of times a reader re-reads its window within a single guard. Re-reading is what
/// exercises the value-stability and no-resurrection invariants.
const READER_PASSES: usize = 4;

/// How often (in retirer iterations) a retirer attempts to reclaim retired slots.
const RECLAIM_EVERY: u64 = 16;

///////////
// Input //
///////////

/// Configuration for a [`StoreStress`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStressInput {
    /// Number of reader threads. Must be below [`GUARD_CAPACITY`].
    readers: usize,
    /// Number of writer threads.
    writers: usize,
    /// Number of retirer threads.
    retirers: usize,
    /// Number of writable (non-frozen) slots.
    capacity: usize,
    /// Bytes per entry. Must be a non-zero multiple of 8 (the stamp lane width).
    entry_bytes: usize,
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

impl StoreStressInput {
    fn check(self) -> anyhow::Result<Self> {
        if self.readers == 0 || self.writers == 0 {
            anyhow::bail!("`readers` and `writers` must be non-zero");
        }
        if self.readers >= GUARD_CAPACITY {
            anyhow::bail!(
                "`readers` ({}) must be below the epoch guard capacity ({GUARD_CAPACITY})",
                self.readers,
            );
        }
        if self.capacity == 0 {
            anyhow::bail!("`capacity` must be non-zero");
        }
        if self.entry_bytes == 0 || !self.entry_bytes.is_multiple_of(8) {
            anyhow::bail!(
                "`entry_bytes` ({}) must be a non-zero multiple of 8",
                self.entry_bytes,
            );
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
        Ok(self)
    }
}

impl Input for StoreStressInput {
    type Raw = Self;

    fn tag() -> &'static str {
        "store-stress"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Self::check(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        StoreStressInput {
            readers: 8,
            writers: 4,
            retirers: 2,
            capacity: 4096,
            entry_bytes: 128,
            epoch_guard_slots: 256,
            freelist_recycle_capacity: 1024,
            low_watermark: 1024,
            duration_secs: 5,
            max_ops: 50_000_000,
            seed: 0xA5A5_1234_DEAD_BEEF,
        }
    }
}

impl std::fmt::Display for StoreStressInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = KeyValue::new();
        kv.push("readers", &self.readers);
        kv.push("writers", &self.writers);
        kv.push("retirers", &self.retirers);
        kv.push("capacity", &self.capacity);
        kv.push("entry_bytes", &self.entry_bytes);
        kv.push("epoch_guard_slots", &self.epoch_guard_slots);
        kv.push("freelist_recycle_capacity", &self.freelist_recycle_capacity);
        kv.push("low_watermark", &self.low_watermark);
        kv.push("duration_secs", &self.duration_secs);
        kv.push("max_ops", &self.max_ops);
        kv.push("seed", &self.seed);
        write!(f, "{}", kv)
    }
}

////////////
// Output //
////////////

/// Summary statistics produced by a [`StoreStress`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStressStats {
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

impl std::fmt::Display for StoreStressStats {
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

/////////////
// Payload //
/////////////

/// Fill `buf` with `stamp` replicated across every 8-byte lane.
fn write_stamp(buf: &mut [u8], stamp: u64) {
    let bytes = stamp.to_ne_bytes();
    for lane in buf.chunks_exact_mut(8) {
        lane.copy_from_slice(&bytes);
    }
}

/// Read the stamp from `buf`, returning `Err` if any 8-byte lane disagrees (a torn read).
fn read_stamp(buf: &[u8]) -> Result<u64, ()> {
    let (lanes, _) = buf.as_chunks::<8>();
    let mut lanes = lanes.iter();
    let first = u64::from_ne_bytes(*lanes.next().ok_or(())?);
    for lane in lanes {
        if u64::from_ne_bytes(*lane) != first {
            return Err(());
        }
    }
    Ok(first)
}

////////////////
// Invariants //
////////////////

/// Per-guard observation of a single slot.
#[derive(Debug, Clone, Copy)]
enum SlotObservations {
    /// The slot was observed readable with the given stamp.
    Readable(u64),
    /// The slot was observed readable and then became unreadable (retired).
    Retired,
}

/// Feed a single observation of slot `i` into the per-guard checker, recording a violation
/// on the shared state if a safety invariant is broken.
fn observe(
    shared: &Shared,
    observed: &mut HashMap<usize, SlotObservations>,
    i: usize,
    read: Option<&[u8]>,
) {
    match (observed.get(&i).copied(), read) {
        // Not yet observed readable; an unreadable slot tells us nothing actionable.
        (None, None) => {}
        // First readable observation: record the stamp (after a tearing check).
        (None, Some(bytes)) => match read_stamp(bytes) {
            Ok(stamp) => {
                observed.insert(i, SlotObservations::Readable(stamp));
            }
            Err(()) => record_violation(shared, format!("torn read at slot {i}")),
        },
        // Still readable: the value must be identical and untorn.
        (Some(SlotObservations::Readable(prev)), Some(bytes)) => match read_stamp(bytes) {
            Ok(stamp) if stamp != prev => record_violation(
                shared,
                format!("slot {i} value changed within guard: {prev} -> {stamp}"),
            ),
            Ok(_) => {}
            Err(()) => record_violation(shared, format!("torn read at slot {i}")),
        },
        // Readable -> unreadable: an allowed, terminal transition.
        (Some(SlotObservations::Readable(_)), None) => {
            observed.insert(i, SlotObservations::Retired);
            shared.transitions.fetch_add(1, Relaxed);
        }
        // Resurrection: a slot that retired came back to life within the same guard.
        (Some(SlotObservations::Retired), Some(_)) => record_violation(
            shared,
            format!("resurrection at slot {i}: unreadable -> readable within one guard"),
        ),
        (Some(SlotObservations::Retired), None) => {}
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

/// State shared by all worker threads for the duration of a run.
struct Shared {
    store: Store,
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

/// Record an observed invariant violation and signal all workers to stop.
fn record_violation(shared: &Shared, message: String) {
    let mut slot = shared.violation.lock().unwrap();
    slot.push(message);
    shared.stop.store(true, Relaxed);
}

/// Return `true` once any termination condition is met.
fn should_stop(shared: &Shared) -> bool {
    shared.stop.load(Relaxed)
        || shared.ops.load(Relaxed) >= shared.max_ops
        || Instant::now() >= shared.deadline
}

/////////////
// Workers //
/////////////

fn writer(shared: &Shared) {
    let mut ops = Local::new(&shared.ops);
    let mut acquires_ok = Local::new(&shared.acquires_ok);
    let mut acquires_fail = Local::new(&shared.acquires_fail);

    let mut peak_live = LocalMax::new(&shared.peak_live);

    while !should_stop(shared) {
        ops.add(1);
        match shared.store.acquire() {
            Some(mut writer) => {
                let stamp = shared.stamp.fetch_add(1, Relaxed);
                write_stamp(writer.as_mut_slice(), stamp);
                writer.publish();

                let live = shared.live.fetch_add(1, Relaxed) + 1;
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

fn retirer(shared: &Shared, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut iteration: u64 = 0;

    let mut retires_ok = Local::new(&shared.retires_ok);
    let mut retires_fail = Local::new(&shared.retires_fail);
    let mut reclaims = Local::new(&shared.reclaims);

    while !should_stop(shared) {
        shared.ops.fetch_add(1, Relaxed);
        iteration += 1;

        // Flow control: keep a steady readable population.
        if shared.live.load(Relaxed) > shared.low_watermark {
            let i = rng.sample(shared.writable);
            if shared.store.retire(i) {
                shared.live.fetch_sub(1, Relaxed);
                retires_ok.add(1);
            } else {
                retires_fail.add(1);
            }
        }

        if iteration.is_multiple_of(RECLAIM_EVERY)
            && let Some(reclaimed) = shared.store.reclaim()
        {
            reclaims.add(reclaimed as u64);
        }

        std::thread::yield_now();
    }
}

fn reader(shared: &Shared, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let slots = shared.slots;
    let window = READER_WINDOW.min(slots);
    let mut observations = HashMap::with_capacity(window);

    let mut ops = Local::new(&shared.ops);
    let mut reads = Local::new(&shared.reads);

    while !should_stop(shared) {
        ops.add(1);
        let Some(guard) = shared.store.reader() else {
            // All guard slots are occupied; back off and retry.
            std::thread::yield_now();
            continue;
        };

        observations.clear();
        let start = rng.sample(shared.readable);
        for _ in 0..READER_PASSES {
            for k in 0..window {
                let i = (start + k) % slots;
                observe(shared, &mut observations, i, guard.read(i));
                reads.add(1);
            }
        }
    }
}

///////////////
// Benchmark //
///////////////

/// The store concurrency stress benchmark.
#[derive(Debug)]
pub struct StoreStress;

impl Benchmark for StoreStress {
    type Input = StoreStressInput;
    type Output = StoreStressStats;

    fn try_match(&self, _input: &StoreStressInput, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "concurrency stress test for the in-memory store (readers/writers/retirers)"
        )
    }

    fn run(
        &self,
        input: &StoreStressInput,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<StoreStressStats> {
        let config = Config {
            capacity: input.capacity,
            entry_bytes: input.entry_bytes,
            epoch_guard_slots: input.epoch_guard_slots,
            freelist_recycle_capacity: input.freelist_recycle_capacity,
        };

        let store = Store::new(config);
        let writable = store.writable();
        let slots = store.slots();
        let start = Instant::now();

        let shared = Shared {
            store,
            slots,
            readable: Uniform::new(0, slots)?,
            writable: Uniform::try_from(writable)?,
            low_watermark: input.low_watermark,
            max_ops: if input.max_ops == 0 {
                u64::MAX
            } else {
                input.max_ops
            },
            deadline: if input.duration_secs == 0 {
                // Effectively unbounded; the op cap terminates the run.
                start + Duration::from_secs(u64::from(u32::MAX))
            } else {
                start + Duration::from_secs(input.duration_secs)
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

        writeln!(output, "{}", input)?;

        std::thread::scope(|scope| {
            let shared = &shared;
            for _ in 0..input.writers {
                scope.spawn(move || writer(shared));
            }
            for t in 0..input.retirers {
                let seed = input.seed ^ (0x2000_0000 + t as u64);
                scope.spawn(move || retirer(shared, seed));
            }
            for t in 0..input.readers {
                let seed = input.seed ^ (0x4000_0000 + t as u64);
                scope.spawn(move || reader(shared, seed));
            }
        });

        let errors: Vec<_> = std::mem::take(&mut *shared.violation.lock().unwrap());
        if !errors.is_empty() {
            anyhow::bail!("invariants violated: {:?}", errors);
        }

        let elapsed = start.elapsed();
        let stats = StoreStressStats {
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

        writeln!(output, "{}", stats)?;
        Ok(stats)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_sure_example_parses() {
        let _ = StoreStressInput::check(StoreStressInput::example()).unwrap();
    }
}
