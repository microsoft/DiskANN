/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, io::Write};

use diskann_benchmark_runner as dbr;
use diskann_inmem::integration::store::simple;
use serde::{Deserialize, Serialize};

pub(super) fn register(registry: &mut dbr::Registry) -> Result<(), dbr::RegistryError> {
    registry.register("simple-store-stress-test", Stress)
}

/// Configuration for a [`Stress`] run.
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
        "store-stress-simple"
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
    type Output = super::Stats;

    fn try_match(
        &self,
        _input: &Input,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "concurrency stress test for the simple in-memory store")
    }

    fn run(
        &self,
        input: &Input,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        let config = simple::Config {
            capacity: input.setup.capacity,
            entry_bytes: input.entry_bytes,
            epoch_guard_slots: input.setup.epoch_guard_slots,
            freelist_recycle_capacity: input.setup.freelist_recycle_capacity,
        };

        writeln!(output, "Simple Store Stress Test\n")?;
        writeln!(output, "{}", input)?;
        let stats = super::run_benchmark(simple::Store::new(config), &input.setup)?;
        writeln!(output, "{}", stats)?;
        Ok(stats)
    }
}

impl super::Testable for simple::Store {
    type Writer<'a> = simple::Writer<'a>;
    type ReaderState<'a> = ReaderState<'a>;

    fn writer(&self) -> Option<Self::Writer<'_>> {
        <simple::Store>::acquire(self)
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
        <simple::Store>::retire(self, i)
    }

    fn reclaim(&self) -> Option<usize> {
        <simple::Store>::reclaim(self)
    }

    fn readable_slots(&self) -> usize {
        <simple::Store>::readable_slots(self)
    }

    fn writable_slots(&self) -> usize {
        <simple::Store>::writable_slots(self)
    }
}

impl super::Writer for simple::Writer<'_> {
    fn write(mut self, stamp: u64) {
        super::intrusive::write_stamp(self.as_mut_slice(), stamp);
        self.publish();
    }
}

#[derive(Debug)]
pub(super) struct ReaderState<'a> {
    store: &'a simple::Store,
    observed: HashMap<usize, super::intrusive::SlotObservations>,
    shared: &'a super::Shared<simple::Store>,
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
    reader: simple::Reader<'a>,
    observed: &'a mut HashMap<usize, super::intrusive::SlotObservations>,
    shared: &'a super::Shared<simple::Store>,
}

impl super::Reader for Reader<'_> {
    /// Feed a single observation of slot `i` into the per-guard checker, recording a
    /// violation on the shared state if a safety invariant is broken.
    fn observe(&mut self, i: usize) {
        super::intrusive::observe(self.observed, i, self.reader.read(i), self.shared)
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
