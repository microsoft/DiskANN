/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Striped lock table for per-vertex synchronization.

use std::sync::RwLock;

/// A fixed-size table of striped locks for per-vertex synchronization.
///
/// Vertex IDs are mapped to lock stripes via a multiply-shift hash. This provides
/// constant memory overhead regardless of dataset size, at the cost of
/// occasional false contention between vertices that map to the same stripe.
///
/// The stripe count is derived from available hardware parallelism (4× cores,
/// rounded to the next power of two, minimum 64). This keeps allocation lightweight
/// in debug/test while still making false contention negligible for real workloads.
pub(crate) struct StripedLocks {
    stripes: Box<[RwLock<()>]>,
    hash_shift: u32,
}

impl StripedLocks {
    // Fibonacci hashing constant for 64-bit: floor(2^64 / phi)
    const HASH_MULTIPLIER: u64 = 0x9E3779B97F4A7C15;

    pub(crate) fn new() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let count = (cpus * 4).next_power_of_two().max(64);
        let hash_shift = 64 - count.trailing_zeros();

        Self {
            stripes: (0..count)
                .map(|_| RwLock::new(()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hash_shift,
        }
    }

    #[inline]
    fn stripe_index(&self, id: usize) -> usize {
        ((id as u64).wrapping_mul(Self::HASH_MULTIPLIER) >> self.hash_shift) as usize
    }

    #[inline]
    pub(crate) fn read(&self, id: usize) -> std::sync::RwLockReadGuard<'_, ()> {
        let stripe = self.stripe_index(id);
        self.stripes[stripe]
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[inline]
    pub(crate) fn write(&self, id: usize) -> std::sync::RwLockWriteGuard<'_, ()> {
        let stripe = self.stripe_index(id);
        self.stripes[stripe]
            .write()
            .unwrap_or_else(|e| e.into_inner())
    }
}
