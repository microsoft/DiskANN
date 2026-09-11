/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, io::Write, sync::atomic::Ordering::Relaxed};

use diskann_benchmark_runner as dbr;
use diskann_inmem::integration::store::checked;
use serde::{Deserialize, Serialize};

pub(super) fn register(registry: &mut dbr::Registry) -> Result<(), dbr::RegistryError> {
    registry.register("store-stress-test-checked", Stress)
}

/// Configuration for a [`Stress`] run.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Input {
    /// Shared stress test setup.
    setup: super::Setup,
}

impl Input {
    fn check(self) -> anyhow::Result<Self> {
        self.setup.check()?;
        Ok(self)
    }
}

impl dbr::Input for Input {
    type Raw = Self;

    fn tag() -> &'static str {
        "store-stress-checked"
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
        }
    }
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Input { setup } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("setup", &setup);
        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
struct Stress;

impl dbr::Benchmark for Stress {
    type Input = Input;
    type Output = super::Stats;

    fn try_match(
        &self,
        _input: &Input,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "concurrency stress test for the checked in-memory store")
    }

    fn run(
        &self,
        input: &Input,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        let config = checked::Config {
            capacity: input.setup.capacity,
            epoch_guard_slots: input.setup.epoch_guard_slots,
            freelist_recycle_capacity: input.setup.freelist_recycle_capacity,
        };

        writeln!(output, "{}", input)?;
        let stats = super::run_benchmark(checked::Store::new(config), &input.setup)?;
        writeln!(output, "{}", stats)?;
        Ok(stats)
    }
}

impl super::Testable for checked::Store {
    type Writer<'a> = checked::Writer<'a>;
    type ReaderState<'a> = ReaderState<'a>;

    fn writer(&self) -> Option<Self::Writer<'_>> {
        <checked::Store>::acquire(self)
    }

    fn reader_state<'a>(
        &'a self,
        capacity_hint: usize,
        shared: &'a super::Shared<Self>,
    ) -> Self::ReaderState<'a> {
        ReaderState {
            store: self,
            shared,
            capacity_hint,
        }
    }

    fn retire(&self, i: usize) -> bool {
        <checked::Store>::retire(self, i)
    }

    fn reclaim(&self) -> Option<usize> {
        <checked::Store>::reclaim(self)
    }

    fn readable_slots(&self) -> usize {
        <checked::Store>::readable_slots(self)
    }

    fn writable_slots(&self) -> usize {
        <checked::Store>::writable_slots(self)
    }
}

impl super::Writer for checked::Writer<'_> {
    fn write(mut self, stamp: u64) {
        self.set(stamp);
        self.publish();
    }
}

#[derive(Debug)]
pub(super) struct ReaderState<'a> {
    store: &'a checked::Store,
    shared: &'a super::Shared<checked::Store>,
    capacity_hint: usize,
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
        let observed = HashMap::with_capacity(self.capacity_hint);

        f(Reader {
            reader: &reader,
            observed,
            shared: self.shared,
        });

        true
    }
}

#[derive(Debug)]
pub(super) struct Reader<'a> {
    observed: HashMap<usize, checked::Value<'a>>,
    reader: &'a checked::Reader<'a>,
    shared: &'a super::Shared<checked::Store>,
}

impl super::Reader for Reader<'_> {
    /// Feed a single observation of slot `i` into the per-guard checker, recording a
    /// violation on the shared state if a safety invariant is broken.
    fn observe(&mut self, i: usize) {
        let read = self.reader.read(i);
        let observed = self.observed.get(&i).map(|v| v.get());

        match (observed, read) {
            // Not yet observed readable; an unreadable slot tells us nothing actionable.
            (None, None) => {}
            // First readable observation: record the stamp (after a tearing check).
            (None, Some(value)) => {
                self.observed.insert(i, value);
            }
            // Still readable: the value must be identical and untorn.
            (Some(previous), Some(value)) => {
                if previous != value.get() {
                    self.shared.record_violation(format!(
                        "slot {i} value changed within guard: {} -> {}",
                        previous,
                        value.get(),
                    ))
                }
            }
            // Readable -> unreadable: an allowed, terminal transition.
            (Some(_), None) => {
                self.shared.transitions.fetch_add(1, Relaxed);
            }
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
