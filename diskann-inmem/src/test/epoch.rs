/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Directed stress tests for `Registry`.

use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

use rand::{Rng, distr::StandardUniform};

use crate::{
    epoch::Registry,
    tag::{AtomicTag, Tag},
};

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
        if let Ok(_) = self.tag.compare_exchange(
            Tag::AVAILABLE,
            Tag::OWNED,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            unsafe { &mut *self.payload.get() }.write(Box::new(payload));
            f();
            self.tag.store(Tag::PUBLISHED, Ordering::Release);
        }
    }

    unsafe fn try_read(&self) -> Option<&Data> {
        if self.tag.load(Ordering::Acquire).can_read() {
            let payload = unsafe { &*self.payload.get() };
            Some(&*unsafe { payload.assume_init_ref() })
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

        if let Ok(_) = self.tag.compare_exchange(
            Tag::PUBLISHED,
            Tag::RETIRING,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            true
        } else {
            false
        }
    }

    unsafe fn make_available(&self) {
        assert_eq!(self.tag.load(Ordering::Relaxed), Tag::RETIRING);
        unsafe { (&mut *self.payload.get()).assume_init_drop() };

        if let Err(_) = self.tag.compare_exchange(
            Tag::RETIRING,
            Tag::AVAILABLE,
            Ordering::Release,
            Ordering::Relaxed,
        ) {
            panic!("concurrency violation");
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if self.tag.load(Ordering::Relaxed) != Tag::AVAILABLE {
            let payload = self.payload.get_mut();
            unsafe { payload.assume_init_drop() };
        }
    }
}

// We control concurrency, so can safely share this.
unsafe impl Sync for Slot {}

fn make_payload(epoch: u64, index: usize) -> Data {
    [
        index as u32,
        epoch as u32,
        (epoch >> 32) as u32,
        (index as u32) ^ (epoch as u32) ^ ((epoch >> 32) as u32),
    ]
}

fn verify_payload(data: &Data) -> (usize, u64) {
    let checksum = data[0] ^ data[1] ^ data[2];
    assert_eq!(
        data[3], checksum,
        "torn or corrupted read: payload {data:?}, expected checksum {checksum}"
    );
    let index = data[0] as usize;
    let epoch = data[1] as u64 | ((data[2] as u64) << 32);
    (index, epoch)
}

struct Record {
    epoch: u64,
    index: usize,
    data: Data,
}

fn read_job(
    registry: &Registry,
    slots: &[Slot],
    stop_at: u64,
    retire_rate: f64,
    active: &AtomicUsize,
) -> Vec<Record> {
    assert!(retire_rate > 0.0);
    assert!(retire_rate < 1.0);

    let mut records = Vec::new();
    let mut rng = rand::rng();

    loop {
        let mut reads = Vec::<&Data>::new();
        let guard = registry.guard().unwrap();
        if guard.epoch >= stop_at {
            break;
        }

        for (i, slot) in slots.iter().enumerate() {
            if let Some(read) = unsafe { slot.try_read() } {
                reads.push(read);

                let sample: f64 = rng.sample(StandardUniform);
                if sample < retire_rate && slot.retire() {
                    guard.retire(i as u32);
                    active.fetch_sub(1, Ordering::Release);

                    std::thread::yield_now();
                    records.push(Record {
                        epoch: guard.epoch,
                        index: i,
                        data: *read,
                    });
                }
            }
        }
    }

    records
}

fn retire_job(registry: &Registry, slots: &[Slot], stop_at: u64, active: &AtomicUsize) {
    loop {
        let epoch = registry.epoch();
        if epoch >= stop_at {
            return;
        }

        if active.load(Ordering::Acquire) != 0 {
            std::thread::yield_now();
            continue;
        }

        if let Some(drain) = registry.try_advance() {
            for i in drain {
                unsafe { slots[i as usize].make_available() };
            }
        }
    }
}

fn write_job(registry: &Registry, slots: &[Slot], stop_at: u64, active: &AtomicUsize) {
    loop {
        let epoch = registry.epoch();
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

    let stop_at = if cfg!(miri) { 11 } else { 50_000 };
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
            write_epoch <= (record.epoch + 1),
            "read data from the future: write_gen={write_epoch}, read_gen={}",
            record.epoch
        );
    }
}
