/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Directed stress tests for `Registry`.

use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    num::NonZeroUsize,
    sync::{
        Barrier,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use rand::{Rng, distr::StandardUniform};

use crate::{
    epoch::{Registry, U62},
    tag::{AtomicTag, Tag},
};

// Stress test for epoch advancement.

type Data = [u32; 4];

struct Slot {
    tag: AtomicTag,
    payload: UnsafeCell<MaybeUninit<Box<Data>>>,
}

impl Slot {
    fn new() -> Self {
        Self {
            tag: AtomicTag::new(Tag::AVAILABLE),
            payload: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    fn try_claim<F>(&self, payload: Data, f: F)
    where
        F: FnOnce(),
    {
        if self
            .tag
            .compare_exchange(
                Tag::AVAILABLE,
                Tag::OWNED,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // SAFETY: By transitioning from AVAILABLE to OWNED, we've acquired ownership
            // of this slot and are thus free to write to the `UnsafeCell`.
            unsafe { &mut *self.payload.get() }.write(Box::new(payload));
            f();
            self.tag.store(Tag::PUBLISHED, Ordering::Release);
        }
    }

    fn try_read(&self) -> Option<&Data> {
        if self.tag.load(Ordering::Acquire).can_read() {
            // SAFETY: We've checked that we can read this cell.
            let payload = unsafe { &*self.payload.get() };

            // SAFETY: Items that can be read **must** be initialized.
            Some(unsafe { payload.assume_init_ref() })
        } else {
            None
        }
    }

    #[must_use]
    fn retire(&self) -> bool {
        let tag = self.tag.load(Ordering::Relaxed);
        if tag != Tag::PUBLISHED {
            return false;
        }

        self.tag
            .compare_exchange(
                Tag::PUBLISHED,
                Tag::RETIRING,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    unsafe fn make_available(&self) {
        assert_eq!(self.tag.load(Ordering::Relaxed), Tag::RETIRING);

        // SAFETY: Items tagged as `RETIRING` must be initialized.
        unsafe { (&mut *self.payload.get()).assume_init_drop() };

        if self
            .tag
            .compare_exchange(
                Tag::RETIRING,
                Tag::AVAILABLE,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .is_err()
        {
            panic!("concurrency violation");
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.tag.load(Ordering::Relaxed) != Tag::AVAILABLE {
            let payload = self.payload.get_mut();

            // SAFETY: We have exclusive access and by convention, if the tag is not
            // available, then the corresponding payload is initialized.
            unsafe { payload.assume_init_drop() };
        }
    }
}

// SAFETY: We control concurrency, so can safely share this.
unsafe impl Sync for Slot {}

fn make_payload(epoch: U62, index: usize) -> Data {
    let epoch = epoch.value();
    [
        index as u32,
        epoch as u32,
        (epoch >> 32) as u32,
        (index as u32) ^ (epoch as u32) ^ ((epoch >> 32) as u32),
    ]
}

fn verify_payload(data: &Data) -> (usize, U62) {
    let checksum = data[0] ^ data[1] ^ data[2];
    assert_eq!(
        data[3], checksum,
        "torn or corrupted read: payload {data:?}, expected checksum {checksum}"
    );
    let index = data[0] as usize;
    let epoch = data[1] as u64 | ((data[2] as u64) << 32);
    (index, U62::new(epoch).unwrap())
}

struct Record {
    epoch: U62,
    index: usize,
    data: Data,
}

fn read_job(
    registry: &Registry,
    slots: &[Slot],
    stop_at: U62,
    retire_rate: f64,
    active: &AtomicUsize,
) -> Vec<Record> {
    assert!(retire_rate > 0.0);
    assert!(retire_rate < 1.0);

    let mut records = Vec::new();
    let mut rng = rand::rng();

    loop {
        let mut reads = Vec::<&Data>::new();
        let guard = registry.retire().unwrap();
        if guard.test_epoch() >= stop_at {
            break;
        }

        for (i, slot) in slots.iter().enumerate() {
            if let Some(read) = slot.try_read() {
                reads.push(read);

                let sample: f64 = rng.sample(StandardUniform);
                if sample < retire_rate && slot.retire() {
                    guard.retire(i as u32);
                    active.fetch_sub(1, Ordering::Release);

                    std::thread::yield_now();
                    records.push(Record {
                        epoch: guard.test_epoch(),
                        index: i,
                        data: *read,
                    });
                }
            }
        }
    }

    records
}

fn retire_job(registry: &Registry, slots: &[Slot], stop_at: U62, active: &AtomicUsize) {
    loop {
        let epoch = registry.ready_epoch();
        if epoch >= stop_at {
            return;
        }

        if active.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
            continue;
        }

        if let Some(drain) = registry.try_advance() {
            for i in drain {
                // SAFETY: retrieving from the drain gives us exclusive access.
                unsafe { slots[i as usize].make_available() };
            }
        }
    }
}

fn write_job(registry: &Registry, slots: &[Slot], stop_at: U62, active: &AtomicUsize) {
    loop {
        let epoch = registry.ready_epoch();
        if epoch >= stop_at {
            return;
        }

        for (i, slot) in slots.iter().enumerate() {
            slot.try_claim(make_payload(epoch, i), || {
                active.fetch_add(1, Ordering::Relaxed);
            });
        }

        std::thread::yield_now();
    }
}

#[test]
fn registry_stress_test() {
    let registry = Registry::new();
    let slots: Vec<_> = std::iter::repeat_with(Slot::new).take(10).collect();
    let active = AtomicUsize::new(0);

    let stop_at = if cfg!(miri) {
        U62::new(11).unwrap()
    } else {
        U62::new(50_000).unwrap()
    };
    let retire_rate = if cfg!(miri) { 0.95 } else { 0.1 };

    // We use two threads for each job to be extra adversarial.
    let barrier = std::sync::Barrier::new(6);
    let result = std::thread::scope(|s| {
        // Spin up readers.
        let r0 = s.spawn(|| {
            barrier.wait();
            read_job(&registry, &slots, stop_at, retire_rate, &active)
        });

        let r1 = s.spawn(|| {
            barrier.wait();
            read_job(&registry, &slots, stop_at, retire_rate, &active)
        });

        // Spin up writers
        s.spawn(|| {
            barrier.wait();
            write_job(&registry, &slots, stop_at, &active);
        });

        s.spawn(|| {
            barrier.wait();
            write_job(&registry, &slots, stop_at, &active);
        });

        // Spin up retirers
        s.spawn(|| {
            barrier.wait();
            retire_job(&registry, &slots, stop_at, &active);
        });
        s.spawn(|| {
            barrier.wait();
            retire_job(&registry, &slots, stop_at, &active);
        });

        let mut r0 = r0.join().unwrap();
        let r1 = r1.join().unwrap();
        r0.extend(r1);
        r0
    });

    for record in &result {
        let (index, write_epoch) = verify_payload(&record.data);

        // The index encoded in the payload must match the slot we read from.
        assert_eq!(
            index, record.index,
            "slot identity mismatch: payload says slot {index}, record says slot {}",
            record.index
        );

        // The slot was written at `write_gen` and read at `record.generation.
        // Since generations increase (newer = larger), write_gen <= record.generation
        // means the write happened at or before the reader's epoch.
        //
        // Note that a reader can observe one epoch change during its tenure, so we *can*
        // observe writes from one higher epoch.
        assert!(
            write_epoch <= record.epoch.checked_add(1).unwrap(),
            "read data from the future: write_gen={write_epoch}, read_gen={}",
            record.epoch
        );
    }
}

/// This test works by having mixed readers and writers hammering the registry.
///
/// Each writer writes monotonically two atomic variables separated by a
/// `std::thread::yield_now`. A thread performing snapshots excludes writers and then probes
/// the slots being written.
///
/// The idea is that if there is a writer, the control thread will eventually observe two
/// different values on a pair of slots, which is an error.
///
/// Additionally, it will take a snapshot of all slots and verify that a reread over the
/// slots yields the same value.
#[test]
fn writer_exclusion_stress_test() {
    const WRITERS: usize = 4;
    const READERS: usize = 2;

    struct StopWorkers<'a>(&'a AtomicBool);

    impl Drop for StopWorkers<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    let snapshots = if cfg!(miri) { 20 } else { 20_000 };
    let registry = Registry::with_capacity(NonZeroUsize::new(4).unwrap());
    let barrier = Barrier::new(WRITERS + READERS + 1);
    let stop = AtomicBool::new(false);
    let data: [[AtomicUsize; 2]; WRITERS] =
        std::array::from_fn(|_| [AtomicUsize::new(0), AtomicUsize::new(0)]);

    std::thread::scope(|s| {
        let registry = &registry;
        let barrier = &barrier;
        let stop = &stop;
        // Stop workers before scope joins them, including when a snapshot assertion panics.
        let _stop_workers = StopWorkers(stop);
        for fields in &data {
            s.spawn(move || {
                let mut value = 0;
                while !stop.load(Ordering::Relaxed) {
                    let Ok(guard) = registry.write_guard() else {
                        std::thread::yield_now();
                        continue;
                    };
                    let shared = guard.read_guard().share();
                    value += 1;

                    // Payload accesses are relaxed: visibility must come from the registry.
                    fields[0].store(value, Ordering::Relaxed);
                    drop(guard);
                    std::thread::yield_now();
                    fields[1].store(value, Ordering::Relaxed);
                    drop(shared);

                    if value == 1 {
                        barrier.wait();
                    }
                }
            });
        }

        for _ in 0..READERS {
            s.spawn(move || {
                let mut reads = 0;
                while !stop.load(Ordering::Relaxed) {
                    let Ok(guard) = registry.read_guard() else {
                        std::thread::yield_now();
                        continue;
                    };
                    let shared = guard.share();
                    drop(guard);
                    std::thread::yield_now();
                    drop(shared);

                    reads += 1;
                    if reads == 1 {
                        barrier.wait();
                    }
                }
            });
        }

        // Every worker completes an operation before snapshotting begins.
        barrier.wait();
        for _ in 0..snapshots {
            let exclusion = loop {
                if let Some(exclusion) = registry.try_exclude_writers() {
                    break exclusion;
                }
                std::thread::yield_now();
            };

            let snapshot: [usize; WRITERS] = std::array::from_fn(|i| {
                let first = data[i][1].load(Ordering::Relaxed);
                assert_eq!(
                    first,
                    data[i][0].load(Ordering::Relaxed),
                    "writer {i} was still updating during exclusion",
                );
                first
            });

            std::thread::yield_now();

            for (i, fields) in data.iter().enumerate() {
                assert_eq!(
                    [
                        fields[1].load(Ordering::Relaxed),
                        fields[0].load(Ordering::Relaxed),
                    ],
                    [snapshot[i]; 2],
                    "writer {i} changed its payload while exclusion was held",
                );
            }
            drop(exclusion);
            std::thread::yield_now();
        }
    });

    for (i, fields) in data.iter().enumerate() {
        let first = fields[0].load(Ordering::Relaxed);
        assert!(first > 0, "writer {i} made no progress");
        assert_eq!(first, fields[1].load(Ordering::Relaxed));
    }
}
