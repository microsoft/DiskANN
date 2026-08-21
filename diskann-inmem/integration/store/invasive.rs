/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, io::Write, sync::atomic::Ordering::Relaxed};

use diskann_benchmark_runner as dbr;
use diskann_inmem::integration::store::invasive;
use serde::{Deserialize, Serialize};

pub(super) fn register(registry: &mut dbr::Registry) -> Result<(), dbr::RegistryError> {
    registry.register("invasive-store-stress-test", Stress)
}

/// Configuration for a [`StoreStress`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Input {
    /// Shared stress test setup.
    setup: super::Setup,

    /// Bytes per entry. Must be a non-zero multiple of 8 (the stamp lane width).
    entry_bytes: usize,
}

impl Input {
    fn check(self) -> anyhow::Result<Self> {
        self.setup.check()?;

        if self.entry_bytes == 0 || !self.entry_bytes.is_multiple_of(8) {
            anyhow::bail!(
                "`entry_bytes` ({}) must be a non-zero multiple of 8",
                self.entry_bytes,
            );
        }

        Ok(self)
    }
}

impl dbr::Input for Input {
    type Raw = Self;

    fn tag() -> &'static str {
        "store-stress-invasive"
    }

    fn from_raw(raw: Self::Raw, _checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        raw.check()
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        Input {
            setup: super::Setup::example(),
            entry_bytes: 128,
        }
    }
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Input { setup, entry_bytes } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("setup", &setup);
        kv.push("entry_bytes", &entry_bytes);
        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
struct Stress;

impl dbr::Benchmark for Stress {
    type Input = Input;
    type Output = super::StoreStressStats;

    fn try_match(
        &self,
        _input: &Input,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "concurrency stress test for the invasive in-memory store"
        )
    }

    fn run(
        &self,
        input: &Input,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        let config = invasive::Config {
            capacity: input.setup.capacity,
            entry_bytes: input.entry_bytes,
            epoch_guard_slots: input.setup.epoch_guard_slots,
            freelist_recycle_capacity: input.setup.freelist_recycle_capacity,
        };

        writeln!(output, "{}", input)?;
        let stats = super::run_benchmark(invasive::Store::new(config), &input.setup)?;
        writeln!(output, "{}", stats)?;
        Ok(stats)
    }
}

/// Per-guard observation of a single slot.
#[derive(Debug, Clone, Copy)]
enum SlotObservations {
    /// The slot was observed readable with the given stamp.
    Readable(u64),
    /// The slot was observed readable and then became unreadable (retired).
    Retired,
}

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

impl super::Testable for invasive::Store {
    type Writer<'a> = invasive::Writer<'a>;
    type ReaderState<'a> = ReaderState<'a>;

    fn writer(&self) -> Option<Self::Writer<'_>> {
        <invasive::Store>::acquire(self)
    }

    fn reader_state<'a>(
        &'a self,
        capacity_hint: usize,
        shared: &'a super::Shared<Self>,
    ) -> Self::ReaderState<'a> {
        let observed = HashMap::with_capacity(capacity_hint);
        ReaderState {
            store: self,
            observed,
            shared,
        }
    }

    fn retire(&self, i: usize) -> bool {
        <invasive::Store>::retire(self, i)
    }

    fn reclaim(&self) -> Option<usize> {
        <invasive::Store>::reclaim(self)
    }

    fn readable_slots(&self) -> usize {
        <invasive::Store>::slots(self)
    }

    fn writable_slots(&self) -> usize {
        <invasive::Store>::writable(self)
    }
}

impl super::Writer for invasive::Writer<'_> {
    fn write(mut self, stamp: u64) {
        write_stamp(self.as_mut_slice(), stamp);
        self.publish();
    }
}

#[derive(Debug)]
pub(super) struct ReaderState<'a> {
    store: &'a invasive::Store,
    observed: HashMap<usize, SlotObservations>,
    shared: &'a super::Shared<invasive::Store>,
}

impl super::ReaderState for ReaderState<'_> {
    type Reader<'a> = Reader<'a>;

    fn try_with_reader<F>(&mut self, f: F) -> bool
    where
        F: FnOnce(Self::Reader<'_>),
    {
        let Some(reader) = self.store.reader() else {
            return false;
        };
        self.observed.clear();

        f(Reader {
            reader,
            observed: &mut self.observed,
            shared: self.shared,
        });

        true
    }
}

#[derive(Debug)]
pub(super) struct Reader<'a> {
    reader: invasive::Reader<'a>,
    observed: &'a mut HashMap<usize, SlotObservations>,
    shared: &'a super::Shared<invasive::Store>,
}

impl super::Reader for Reader<'_> {
    /// Feed a single observation of slot `i` into the per-guard checker, recording a
    /// violation on the shared state if a safety invariant is broken.
    fn observe(&mut self, i: usize) {
        let read = self.reader.read(i);
        let observed = self.observed.get(&i).copied();

        match (observed, read) {
            // Not yet observed readable; an unreadable slot tells us nothing actionable.
            (None, None) => {}
            // First readable observation: record the stamp (after a tearing check).
            (None, Some(bytes)) => match read_stamp(bytes) {
                Ok(stamp) => {
                    self.observed.insert(i, SlotObservations::Readable(stamp));
                }
                Err(()) => self
                    .shared
                    .record_violation(format!("torn read at slot {i}")),
            },
            // Still readable: the value must be identical and untorn.
            (Some(SlotObservations::Readable(prev)), Some(bytes)) => match read_stamp(bytes) {
                Ok(stamp) if stamp != prev => self.shared.record_violation(format!(
                    "slot {i} value changed within guard: {prev} -> {stamp}"
                )),
                Ok(_) => {}
                Err(()) => self
                    .shared
                    .record_violation(format!("torn read at slot {i}")),
            },
            // Readable -> unreadable: an allowed, terminal transition.
            (Some(SlotObservations::Readable(_)), None) => {
                self.observed.insert(i, SlotObservations::Retired);
                self.shared.transitions.fetch_add(1, Relaxed);
            }
            // Resurrection: a slot that retired came back to life within the same guard.
            (Some(SlotObservations::Retired), Some(_)) => self.shared.record_violation(format!(
                "resurrection at slot {i}: unreadable -> readable within one guard"
            )),
            (Some(SlotObservations::Retired), None) => {}
        }
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
        let _ = Input::check(<Input as dbr::Input>::example()).unwrap();
    }
}
