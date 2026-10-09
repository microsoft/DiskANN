/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Free space map.
//!
//! The free space map tracks the status for each ID, which can be one of Free,
//! Occupied, or Deleted.
//!
//! Individual read/write/rmw operations in Garnet are atomic, but sequences of
//! these operations are not. To ensure the FSM is accurate, we try to order
//! operations so that concurrency conflicts are benign.
//!
//! Additionally, in order to keep inserts fast, we don't want to repeatedly
//! scan the FSM to find ids to use. For this case we employ thread-safe
//! in-memory views of certain state that is updated atomically: a queue which
//! keeps track of a fixed number of IDs available for reused, and an atomic
//! counter which tracks the next new ID.

use crossbeam::queue::ArrayQueue;
use std::sync::{
    Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use thiserror::Error;

use crate::garnet::{Callbacks, Context, GarnetError, Term};

const BLOCK_SIZE_IDS: usize = 2usize.pow(16);
const BLOCK_SIZE_BYTES: usize = BLOCK_SIZE_IDS / 8;
const FAST_SIZE: usize = 1024;
// Full FSM key will be the prefix + u32 index of the block
const FSM_KEY_PREFIX: u32 = u32::from_be_bytes(*b"_fsm");

#[derive(Debug, Error, PartialEq)]
pub(crate) enum FsmError {
    #[error("Garnet operation failed")]
    Garnet(#[from] GarnetError),
    #[error("requested ID is out of range {0}")]
    IdOutOfRange(u32),
}

/// Guard protecting vector writes against quantization phase changes.
pub(crate) struct ReuseGuard<'a> {
    id: u32,
    barrier: RwLockReadGuard<'a, Barrier>,
}

impl<'a> ReuseGuard<'a> {
    fn new(id: u32, barrier: RwLockReadGuard<'a, Barrier>) -> Self {
        ReuseGuard { id, barrier }
    }

    pub(crate) fn id(&self) -> u32 {
        self.id
    }

    pub(crate) fn should_quantize(&self) -> bool {
        self.barrier.quantization_enabled
    }

    pub(crate) fn max_id_for_backfill(&self) -> u32 {
        self.barrier.max_id_for_backfill
    }
}

struct Barrier {
    max_id_for_backfill: u32,
    quantization_enabled: bool,
}

struct IdMinter {
    next_id: u32,
    max_block: u32,
    buffer: Vec<u8>,
}

/// The free space map manages the user-vector ID pool in Garnet, including IDs
/// reclaimed after deletion and gaps left by imports. The configured start-point
/// ID is reserved and is not tracked.
///
/// Use `next_id()` to allocate and mark an ID as used, `claim_id()` to claim an
/// imported ID, and `mark_free()` to release a deleted vector's ID.
///
/// When quantization backfill is required, set `reuse_enabled` to `false` and
/// `quantization_enabled` to true in the constructor. This will prevent all ID
/// reuse until `enable_reuse()` is called. At the start of backfill,
/// `max_id_for_backfill()` can be used to set the upper bound of the backfill
/// range.
pub(crate) struct FreeSpaceMap {
    /// Garnet callbacks for reading/writing FSM keys
    callbacks: Callbacks,
    reserved_id: u32,
    /// A flag to signal whether there are free IDs in the FSM.
    /// Gaps and deletions set this flag; a scan with no free IDs clears it to
    /// prevent extraneous reads of FSM blocks.
    has_free_ids: AtomicBool,
    /// A queue of reusable IDs to prevent excessive reads of the FSM
    fast_free_list: ArrayQueue<u32>,
    /// Controls minting new IDs and expanding the FSM blocks
    id_minter: RwLock<IdMinter>,
    /// The total number of IDs marked used in the FSM
    total_used: AtomicUsize,
    /// Controls when ID reuse is enabled.
    reuse_enabled: AtomicBool,
    /// Quantization backfill related parameters that must be synchronized.
    barrier: RwLock<Barrier>,
    /// Refill lock, to prevent multiple fast free list refills happening concurrently.
    refill_lock: Mutex<()>,
}

impl FreeSpaceMap {
    pub(crate) fn new(
        ctx: &Context,
        callbacks: Callbacks,
        reserved_id: u32,
        quantization_enabled: bool,
        reuse_enabled: bool,
    ) -> Result<Self, FsmError> {
        let has_free_ids = AtomicBool::new(false);
        let fast_free_list = ArrayQueue::new(FAST_SIZE);
        let total_used = AtomicUsize::new(0);
        let reuse_enabled = AtomicBool::new(reuse_enabled);
        let barrier = RwLock::new(Barrier {
            max_id_for_backfill: u32::MAX,
            quantization_enabled,
        });
        let id_minter = RwLock::new(IdMinter {
            next_id: 0,
            max_block: u32::MAX,
            buffer: vec![0u8; BLOCK_SIZE_BYTES],
        });
        let refill_lock = Mutex::new(());

        let mut this = Self {
            callbacks,
            reserved_id,
            has_free_ids,
            fast_free_list,
            id_minter,
            total_used,
            reuse_enabled,
            barrier,
            refill_lock,
        };

        // Attempt to load state from Garnet.
        let block_key = Self::block_key(0);
        if this
            .callbacks
            .exists_wid(&ctx.term(Term::Metadata), block_key, BLOCK_SIZE_BYTES)
        {
            this.load_state(ctx)?;
        } else {
            // Allocate first block.
            let mut id_minter = this.id_minter.write().unwrap();
            this.expand_to(&mut id_minter, ctx, 0)?;
        }

        Ok(this)
    }

    /// Load all state from Garnet by scanning the FSM blocks.
    fn load_state(&mut self, ctx: &Context) -> Result<(), FsmError> {
        let mut max_block_id = 0;
        while self.callbacks.exists_wid(
            &ctx.term(Term::Metadata),
            Self::block_key(max_block_id),
            BLOCK_SIZE_BYTES,
        ) {
            max_block_id += 1;
        }

        let mut block = vec![0u8; BLOCK_SIZE_BYTES];
        let mut last_used_id = -1i64;
        let mut total_used = 0usize;

        for block_id in (0..max_block_id).rev() {
            let block_key = Self::block_key(block_id);

            if !self
                .callbacks
                .read_single_wid(&ctx.term(Term::Metadata), block_key, &mut block)
            {
                break;
            }

            let mut id = block_id * BLOCK_SIZE_IDS as u32 + (BLOCK_SIZE_IDS as u32 - 1);

            for &byte in block.iter().rev() {
                for bidx in (0..8).rev() {
                    if id != self.reserved_id {
                        let used = bit_used(byte, bidx);
                        if used {
                            last_used_id = last_used_id.max(id as i64);
                            total_used += 1;
                        } else if (id as i64) < last_used_id {
                            let _ = self.fast_free_list.push(id);
                        }
                    }

                    id = id.saturating_sub(1);
                }
            }
        }

        let mut id_minter = self.id_minter.write().unwrap();
        id_minter.max_block = max_block_id - 1;

        id_minter.next_id = (last_used_id + 1) as u32;

        let barrier = self.barrier.get_mut().unwrap();
        if barrier.quantization_enabled {
            barrier.max_id_for_backfill = id_minter.next_id.saturating_sub(1);
        }

        self.total_used.store(total_used, Ordering::Release);

        if !self.fast_free_list.is_empty() {
            self.has_free_ids.store(true, Ordering::Release);
        }

        Ok(())
    }

    /// Mark an ID as free.
    pub(crate) fn mark_free(&self, ctx: &Context, id: u32) -> Result<(), FsmError> {
        // We don't care about the changed status on free.
        self.mark_id(ctx, id, false).map(|_| ())
    }

    /// Mark an ID as occupied.
    fn mark_used(&self, ctx: &Context, id: u32) -> Result<bool, FsmError> {
        self.mark_id(ctx, id, true)
    }

    /// Mark an ID according to value (true = used, false = free), but don't check that it's in range.
    ///
    /// Side effects only happen if the value actually changed. The return value reflects whether
    /// data changed or not.
    ///
    /// This version does not acquire a guard on `id_minter` and is safe to call while that lock is held.
    fn mark_id_unchecked(&self, ctx: &Context, id: u32, used: bool) -> Result<bool, FsmError> {
        let (block_id, byte_idx, bit_idx) = self.indexes_for_id(id);
        let block_key = Self::block_key(block_id);
        let mut changed = false;

        if !self.callbacks.rmw_wid(
            &ctx.term(Term::Metadata),
            block_key,
            BLOCK_SIZE_BYTES,
            |data: &mut [u8]| changed = update_status(used, &mut data[byte_idx], bit_idx),
        ) {
            return Err(FsmError::Garnet(GarnetError::Write));
        }

        if changed {
            if used {
                self.total_used.fetch_add(1, Ordering::AcqRel);
            } else {
                self.total_used.fetch_sub(1, Ordering::AcqRel);
            }
        }

        // NOTE: We don't modify the free list if the id was already free.
        if !used && changed {
            // Push the id onto the fast free list. If the queue is full, ignore it.
            let _ = self.fast_free_list.push(id);
            self.has_free_ids.store(true, Ordering::Release);
        }

        Ok(changed)
    }

    /// Mark an ID according to value (true = used, false = free).
    /// Side effects only happen if the value actually changed. The return value reflects whether
    /// data changed or not.
    fn mark_id(&self, ctx: &Context, id: u32, used: bool) -> Result<bool, FsmError> {
        {
            let id_minter = self.id_minter.read().unwrap();
            if id == self.reserved_id || id >= id_minter.next_id {
                return Err(FsmError::IdOutOfRange(id));
            }
        }

        self.mark_id_unchecked(ctx, id, used)
    }

    /// Return whether a given ID is free.
    pub(crate) fn is_free(&self, ctx: &Context, id: u32) -> Result<bool, FsmError> {
        // Don't hold the lock longer than we have to.
        let max_block = {
            let id_minter = self.id_minter.read().unwrap();
            if id == self.reserved_id || id >= id_minter.next_id {
                return Err(FsmError::IdOutOfRange(id));
            }

            id_minter.max_block
        };

        let (block_id, byte_idx, bit_idx) = self.indexes_for_id(id);
        if block_id > max_block || max_block == u32::MAX {
            return Err(FsmError::Garnet(GarnetError::Read));
        }

        let block_key = Self::block_key(block_id);
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];

        if !self
            .callbacks
            .read_single_wid(&ctx.term(Term::Metadata), block_key, &mut block)
        {
            return Err(FsmError::Garnet(GarnetError::Read));
        }

        let free = !bit_used(block[byte_idx], bit_idx);

        Ok(free)
    }

    /// Return a a new ID.
    /// This may be a a fresh ID larger than all the others, or it may be a reused ID that
    /// previously belonged to a deleted element. The returned ID is marked as used.
    ///
    /// Returns a guard with `id()` and `should_quantize()` accessors. IDs returned
    /// with `should_quantize()` equal true should be quantized.
    pub(crate) fn next_id(&self, ctx: &Context) -> Result<ReuseGuard<'_>, FsmError> {
        // A read barrier is acquired for the whole function. This prevents ID
        // minting during quantization phase changes.
        let barrier = self.barrier.read().unwrap();

        if self.reuse_enabled.load(Ordering::Acquire) && self.has_free_ids.load(Ordering::Acquire) {
            // We retry reusing a freed ID until there are none or we get one and marking it used
            // succeeds in changing the value.
            loop {
                let id = if let Some(id) = self.fast_free_list.pop() {
                    let changed = self.mark_used(ctx, id)?;
                    if !changed {
                        continue;
                    }
                    Some(id)
                } else {
                    // need to scan
                    if self.refill_fast_free_list(ctx)?
                        && let Some(id) = self.fast_free_list.pop()
                    {
                        let changed = self.mark_used(ctx, id)?;
                        if !changed {
                            continue;
                        }
                        Some(id)
                    } else {
                        None
                    }
                };

                if let Some(id) = id {
                    return Ok(ReuseGuard::new(id, barrier));
                }

                break;
            }
        }

        // Mint a new ID and mark it used.
        let mut id_minter = self.id_minter.write().unwrap();
        let mut id = id_minter.next_id;
        if id == self.reserved_id {
            id = id.checked_add(1).ok_or(FsmError::IdOutOfRange(id))?;
        }
        let next_id = id.checked_add(1).ok_or(FsmError::IdOutOfRange(id))?;
        self.expand_to(&mut id_minter, ctx, id)?;
        self.mark_id_unchecked(ctx, id, true)?;
        id_minter.next_id = next_id;

        Ok(ReuseGuard::new(id, barrier))
    }

    /// Guard writes to an existing ID without changing its allocation state.
    pub(crate) fn existing_id(&self, id: u32) -> ReuseGuard<'_> {
        ReuseGuard::new(id, self.barrier.read().unwrap())
    }

    /// Claim an imported ID, creating any missing blocks and advancing the maximum ID.
    ///
    /// Repeated claims do not change the used count. The configured start-point ID
    /// and u32::MAX are rejected; u32::MAX cannot represent its next ID.
    pub(crate) fn claim_id(&self, ctx: &Context, id: u32) -> Result<(), FsmError> {
        if id == self.reserved_id {
            return Err(FsmError::IdOutOfRange(id));
        }
        let next_id = id.checked_add(1).ok_or(FsmError::IdOutOfRange(id))?;
        let mut id_minter = self.id_minter.write().unwrap();
        let grows = id >= id_minter.next_id;

        if grows {
            self.expand_to(&mut id_minter, ctx, id)?;
        }

        let _ = self.mark_id_unchecked(ctx, id, true)?;

        if grows {
            if id > id_minter.next_id {
                self.has_free_ids.store(true, Ordering::Release);
            }
            id_minter.next_id = next_id;
        }

        Ok(())
    }

    /// Return the maximum ID that has been assigned to a vector.
    ///
    /// This ID may be free if that ID has been deleted since the ID was created.
    pub(crate) fn max_id(&self) -> u32 {
        let id_minter = self.id_minter.read().unwrap();
        id_minter.next_id.saturating_sub(1)
    }

    /// Return the number of IDs currently marked used in the FSM.
    pub(crate) fn total_used(&self) -> usize {
        self.total_used.load(Ordering::Acquire)
    }

    /// Return the FSM block number, byte index, and bit index for a given ID.
    /// The block number is the block which stores this ID, the byte index is byte offset
    /// within the block which contains the status bits, and the bit index is the bit index
    /// within that byte (from MSB to LSB) of the first status bit.
    fn indexes_for_id(&self, id: u32) -> (u32, usize, usize) {
        let id = id as usize;
        let block_id = (id / BLOCK_SIZE_IDS) as u32;
        let block_idx = id % BLOCK_SIZE_IDS;
        let byte_idx = block_idx / 8;
        let bit_idx = block_idx % 8;
        (block_id, byte_idx, bit_idx)
    }

    fn block_key(block_id: u32) -> u64 {
        let block_id: u64 = block_id.into();
        block_id << 32 | (FSM_KEY_PREFIX as u64)
    }

    /// Scan the FSM blocks to fill up fast_free_list.
    fn refill_fast_free_list(&self, ctx: &Context) -> Result<bool, FsmError> {
        // NOTE: We take a lock to prevent multiple refills happening simultaneously.
        let _guard = self.refill_lock.lock().unwrap();

        // If we had to wait to acquire the lock, it's possible some else refilled the list first, so check it again.
        if !self.fast_free_list.is_empty() {
            return Ok(true);
        }

        let (max_block, next_id) = {
            let id_minter = self.id_minter.read().unwrap();
            (id_minter.max_block, id_minter.next_id)
        };

        let mut has_free_ids = false;
        let mut id = 0u32;
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];
        'scan: for block_id in 0..=max_block {
            if id >= next_id {
                // Don't look at IDs outside the current range.
                break;
            }

            let block_key = Self::block_key(block_id);
            if !self
                .callbacks
                .read_single_wid(&ctx.term(Term::Metadata), block_key, &mut block)
            {
                return Err(FsmError::Garnet(GarnetError::Read));
            }

            for &byte in &block {
                if id >= next_id {
                    // Don't look at IDs outside the current range.
                    break 'scan;
                }

                if byte == 0xff {
                    id += 8;
                    continue;
                }

                for bidx in 0..8 {
                    if id >= next_id {
                        // Don't look at IDs outside the current range.
                        break 'scan;
                    }

                    if id != self.reserved_id && !bit_used(byte, bidx) {
                        has_free_ids = true;
                        self.has_free_ids.store(true, Ordering::Release);
                        if self.fast_free_list.push(id).is_err() {
                            break 'scan;
                        }
                    }
                    id += 1;
                }
            }
        }

        if !has_free_ids {
            self.has_free_ids.store(false, Ordering::Release);
        }

        Ok(has_free_ids)
    }

    /// Create every missing block through the block containing `id`.
    /// The initialized prefix advances only after each successful block write.
    fn expand_to(
        &self,
        id_minter: &mut RwLockWriteGuard<IdMinter>,
        ctx: &Context,
        id: u32,
    ) -> Result<(), FsmError> {
        let (target_block, _, _) = self.indexes_for_id(id);
        if id_minter.max_block != u32::MAX && target_block <= id_minter.max_block {
            return Ok(());
        }

        let first_block = if id_minter.max_block == u32::MAX {
            0
        } else {
            id_minter.max_block + 1
        };
        let ctx = ctx.term(Term::Metadata);
        for block_id in first_block..=target_block {
            let block_key = Self::block_key(block_id);

            if !self.callbacks.write_wid(&ctx, block_key, &id_minter.buffer) {
                return Err(FsmError::Garnet(GarnetError::Write));
            }

            // Keep the successfully persisted prefix if a later write fails.
            id_minter.max_block = block_id;
        }

        Ok(())
    }

    /// Visit each used id in the FSM, invoking f on each id.
    pub(crate) fn visit_used<F>(&self, ctx: &Context, mut f: F) -> Result<(), FsmError>
    where
        F: FnMut(u32) -> bool,
    {
        let max_block = { self.id_minter.read().unwrap().max_block };
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];

        for block_id in 0..=max_block {
            let block_key = Self::block_key(block_id);
            if !self
                .callbacks
                .read_single_wid(&ctx.term(Term::Metadata), block_key, &mut block)
            {
                return Err(FsmError::Garnet(GarnetError::Read));
            }

            let first_id = block_id * BLOCK_SIZE_IDS as u32;
            for (byte_idx, &byte) in block.iter().enumerate() {
                if byte == 0x00 {
                    continue;
                }

                let byte_id = first_id + (byte_idx * 8) as u32;
                for bidx in 0..8 {
                    if bit_used(byte, bidx) {
                        let id = byte_id + bidx as u32;
                        if id != self.reserved_id && !f(id) {
                            return Ok(());
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Signal that new IDs should be quantized.
    pub(crate) fn enable_quantization(&self) {
        self.enable_quantization_if(|| true);
    }

    /// Prepare a quantization change while under the barrier, then enable it on success.
    /// `prepare` must not acquire the FSM barrier.
    pub(crate) fn enable_quantization_if(&self, prepare: impl FnOnce() -> bool) -> bool {
        let mut guard = self.barrier.write().unwrap();
        if !prepare() {
            return false;
        }
        guard.max_id_for_backfill = self.max_id();
        guard.quantization_enabled = true;
        true
    }

    /// Allow reuse of previously deleted IDs.
    pub(crate) fn enable_reuse(&self) {
        self.reuse_enabled.store(true, Ordering::Release);
    }

    /// Return the max ID for purposes of backfilling quantized vectors.
    pub(crate) fn max_id_for_backfill(&self) -> u32 {
        self.barrier.read().unwrap().max_id_for_backfill
    }
}

/// Return whether the `bidx`th bit is set in byte, where bits are labeled from left to right.
fn bit_used(byte: u8, bidx: usize) -> bool {
    (byte >> (7 - bidx)) & 0x1 == 0x1
}

/// Update the `bidx`th bit to match `used`, returning whether the value changed.
fn update_status(used: bool, byte: &mut u8, bidx: usize) -> bool {
    let mask = 0x1 << (7 - bidx);
    let value = (used as u8) << (7 - bidx);
    let changed = used && *byte & mask == 0 || !used && *byte & mask != 0;
    *byte &= !mask;
    *byte |= value;
    changed
}

#[cfg(test)]
mod tests {
    use crate::{
        fsm::{BLOCK_SIZE_BYTES, BLOCK_SIZE_IDS, FreeSpaceMap, FsmError},
        garnet::{Callbacks, Context, GarnetError, Term},
        test_utils::Store,
    };

    #[test]
    fn basic_next_id() {
        let store = Store::new();
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();

        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 2);
    }

    #[test]
    fn claim_zero_in_empty_map() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), u32::MAX, false, true).unwrap();

        fsm.claim_id(&ctx, 0).unwrap();
        // Second call should be idempotent
        fsm.claim_id(&ctx, 0).unwrap();
        assert!(!fsm.is_free(&ctx, 0).unwrap());
        assert_eq!(fsm.total_used(), 1);
        assert_eq!(fsm.max_id(), 0);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        assert_eq!(fsm.total_used(), 2);
    }

    #[test]
    fn basic_delete() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();

        assert_eq!(fsm.mark_free(&ctx, 1), Err(FsmError::IdOutOfRange(1)));
        for id in 1..=64 {
            assert_eq!(fsm.next_id(&ctx).unwrap().id(), id);
        }

        fsm.mark_free(&ctx, 37).unwrap();
        fsm.mark_free(&ctx, 9).unwrap();
        fsm.mark_free(&ctx, 37).unwrap();
        assert_eq!(fsm.total_used(), 62);
        for id in 1..=64 {
            assert_eq!(fsm.is_free(&ctx, id).unwrap(), id == 9 || id == 37);
        }
        assert_eq!(fsm.is_free(&ctx, 65), Err(FsmError::IdOutOfRange(65)));
    }

    #[test]
    fn basic_id_reuse() {
        let store = Store::new();
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();

        for _ in 0u32..64 {
            let _ = fsm.next_id(&ctx).unwrap();
        }

        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 65);

        // After deleting an ID, it should be returned from `next_id()`.
        fsm.mark_free(&ctx, 37).unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 37);
        // Once all free IDs are used, fresh ones should be returned.
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 66);
    }

    #[test]
    fn basic_recovery() {
        let store = Store::new();
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();

        for _ in 0u32..64 {
            let _ = fsm.next_id(&ctx).unwrap();
        }

        fsm.mark_free(&ctx, 37).unwrap();

        // Loading FSM from store should recover all the state.
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        assert_eq!(fsm.max_id(), 64);
        assert_eq!(fsm.total_used(), 63);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 37);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 65);
    }

    #[test]
    fn backfill_recovery() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, false).unwrap();

        for _ in 0..64 {
            let _ = fsm.next_id(&ctx).unwrap();
        }
        fsm.mark_free(&ctx, 37).unwrap();
        fsm.enable_quantization();
        drop(fsm);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, true, false).unwrap();
        assert_eq!(fsm.max_id_for_backfill(), 64);

        let next_id = fsm.next_id(&ctx).unwrap();
        assert_eq!(next_id.id(), 65);
        assert!(next_id.should_quantize());
        drop(next_id);
        assert_eq!(fsm.max_id_for_backfill(), 64);

        fsm.enable_reuse();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 37);
    }

    #[test]
    fn claim_out_of_order_ids() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        let block_size = BLOCK_SIZE_IDS as u32;
        let high_id = 3 * block_size + 7;

        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        for id in [
            high_id,
            block_size + 5,
            block_size,
            block_size - 1,
            1,
            high_id,
        ] {
            fsm.claim_id(&ctx, id).unwrap();
        }
        assert_eq!(fsm.max_id(), high_id);
        assert_eq!(fsm.total_used(), 5);
        for id in [2, block_size + 1, 2 * block_size, high_id - 1] {
            assert!(fsm.is_free(&ctx, id).unwrap(), "id={id}");
        }
        let mut used = Vec::new();
        fsm.visit_used(&ctx, |id| {
            used.push(id);
            true
        })
        .unwrap();
        assert_eq!(
            used,
            [1, block_size - 1, block_size, block_size + 5, high_id]
        );
        let empty_block = store
            .get(
                ctx.term(Term::Metadata).get(),
                &FreeSpaceMap::block_key(2).to_ne_bytes(),
            )
            .unwrap();
        assert_eq!(empty_block.len(), BLOCK_SIZE_BYTES);
        assert!(empty_block.iter().all(|byte| *byte == 0));
        fsm.mark_free(&ctx, high_id).unwrap();
        assert_eq!(fsm.total_used(), 4);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), high_id);
        assert_eq!(fsm.total_used(), 5);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 2);
        drop(fsm);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        assert_eq!(fsm.max_id(), high_id);
        assert_eq!(fsm.total_used(), 6);
        used.clear();
        fsm.visit_used(&ctx, |id| {
            used.push(id);
            true
        })
        .unwrap();
        assert_eq!(
            used,
            [1, 2, block_size - 1, block_size, block_size + 5, high_id]
        );
    }

    #[test]
    fn start_point_is_not_tracked() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();

        assert_eq!(fsm.claim_id(&ctx, 0), Err(FsmError::IdOutOfRange(0)));
        assert_eq!(fsm.total_used(), 0);
        assert_eq!(fsm.max_id(), 0);
        assert_eq!(fsm.is_free(&ctx, 0), Err(FsmError::IdOutOfRange(0)));
        assert_eq!(fsm.mark_free(&ctx, 0), Err(FsmError::IdOutOfRange(0)));
        fsm.visit_used(&ctx, |id| panic!("unexpected used ID {id}"))
            .unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        drop(fsm);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        assert_eq!(fsm.total_used(), 1);
        fsm.mark_free(&ctx, 1).unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 2);
        assert_eq!(fsm.total_used(), 2);
    }

    #[test]
    fn recovery_ignores_start_point_bit() {
        let store = Store::new();
        let ctx = Context::new(0);
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];
        block[0] = 0b11000000;
        store.set(
            ctx.term(Term::Metadata).get(),
            &FreeSpaceMap::block_key(0).to_ne_bytes(),
            &block,
        );

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        assert_eq!(fsm.total_used(), 1);
        assert_eq!(fsm.max_id(), 1);
        let mut used = Vec::new();
        fsm.visit_used(&ctx, |id| {
            used.push(id);
            true
        })
        .unwrap();
        assert_eq!(used, [1]);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 2);
    }

    #[test]
    fn expansion_write_failure_preserves_claims() {
        unsafe extern "C" fn write(
            context: u64,
            key: *const u8,
            key_len: usize,
            value: *const u8,
            value_len: usize,
        ) -> bool {
            let key_bytes = unsafe { std::slice::from_raw_parts(key, key_len) };
            if key_bytes == FreeSpaceMap::block_key(2).to_ne_bytes() {
                return false;
            }
            unsafe {
                (Store::attach().callbacks().write_callback())(
                    context, key, key_len, value, value_len,
                )
            }
        }

        for reload in [false, true] {
            let store = Store::new();
            let ctx = Context::new(0);
            let callbacks = store.callbacks();
            let failing_callbacks = Callbacks::new(
                callbacks.read_callback(),
                write,
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
                callbacks.filter_callback(),
                callbacks.log_callback(),
            );
            let mut fsm = FreeSpaceMap::new(&ctx, failing_callbacks, 0, false, true).unwrap();
            let high_id = 3 * BLOCK_SIZE_IDS as u32 + 7;
            assert_eq!(
                fsm.claim_id(&ctx, high_id),
                Err(FsmError::Garnet(GarnetError::Write))
            );
            assert_eq!(fsm.max_id(), 0);
            assert_eq!(fsm.total_used(), 0);

            let low_id = BLOCK_SIZE_IDS as u32 + 5;
            fsm.claim_id(&ctx, low_id).unwrap();
            if reload {
                fsm = FreeSpaceMap::new(&ctx, callbacks, 0, false, true).unwrap();
            } else {
                fsm.callbacks = callbacks;
            }
            fsm.claim_id(&ctx, high_id).unwrap();
            assert!(!fsm.is_free(&ctx, low_id).unwrap());
            assert!(!fsm.is_free(&ctx, high_id).unwrap());
            assert!(fsm.is_free(&ctx, 2 * BLOCK_SIZE_IDS as u32).unwrap());
            assert_eq!(fsm.total_used(), 2);
            assert_eq!(fsm.max_id(), high_id);
        }
    }

    #[test]
    fn sparse_claims_respect_backfill_barrier() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, false).unwrap();
        let high_id = 2 * BLOCK_SIZE_IDS as u32;
        fsm.claim_id(&ctx, high_id).unwrap();
        fsm.enable_quantization();

        let next_id = fsm.next_id(&ctx).unwrap();
        assert_eq!(next_id.id(), high_id + 1);
        assert!(next_id.should_quantize());
        drop(next_id);
        assert_eq!(fsm.max_id_for_backfill(), high_id);
        fsm.enable_reuse();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        assert_eq!(fsm.total_used(), 3);
    }

    #[test]
    fn exhausted_id_space_does_not_wrap() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        assert_eq!(
            fsm.claim_id(&ctx, u32::MAX),
            Err(FsmError::IdOutOfRange(u32::MAX))
        );
        assert_eq!(fsm.max_id(), 0);
        assert_eq!(fsm.total_used(), 0);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);

        fsm.id_minter.write().unwrap().next_id = u32::MAX;
        assert!(matches!(
            fsm.next_id(&ctx),
            Err(FsmError::IdOutOfRange(u32::MAX))
        ));
        fsm.mark_free(&ctx, 1).unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
    }

    #[test]
    fn dynamic_expansion() {
        let store = Store::new();
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();

        // Asking for more than BLOCK_SIZE_IDS will force another FSM block to be allocated.
        for id in 1..=BLOCK_SIZE_IDS as u32 + 1 {
            assert_eq!(fsm.next_id(&ctx).unwrap().id(), id);
        }
        assert_eq!(fsm.max_id(), BLOCK_SIZE_IDS as u32 + 1);
        assert_eq!(fsm.total_used(), BLOCK_SIZE_IDS + 1);
        for id in BLOCK_SIZE_IDS as u32 - 1..=BLOCK_SIZE_IDS as u32 + 1 {
            assert!(!fsm.is_free(&ctx, id).unwrap());
        }
        fsm.mark_free(&ctx, BLOCK_SIZE_IDS as u32).unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), BLOCK_SIZE_IDS as u32);
    }

    #[test]
    fn recover_sparse_claims() {
        let store = Store::new();
        let ctx = Context::new(0);
        let block_size = BLOCK_SIZE_IDS as u32;
        let high_id = 2 * block_size + 7;
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, true).unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
        fsm.claim_id(&ctx, high_id).unwrap();
        fsm.claim_id(&ctx, block_size).unwrap();
        drop(fsm);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), 0, false, false).unwrap();
        assert_eq!(fsm.max_id(), high_id);
        assert_eq!(fsm.total_used(), 3);
        let mut visited = Vec::new();
        fsm.visit_used(&ctx, |id| {
            visited.push(id);
            true
        })
        .unwrap();
        assert_eq!(visited, [1, block_size, high_id]);
        assert!(fsm.is_free(&ctx, block_size + 1).unwrap());
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), high_id + 1);

        fsm.enable_reuse();
        let reused = fsm.next_id(&ctx).unwrap().id();
        assert!(reused > 0 && reused < high_id && !visited.contains(&reused));
        assert_eq!(fsm.max_id(), high_id + 1);
        assert_eq!(fsm.total_used(), 5);
    }

    #[test]
    fn reserved_id_is_not_tracked() {
        for reserved_id in [0, 2, u32::MAX] {
            let store = Store::new();
            let ctx = Context::new(0);
            let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), reserved_id, false, true).unwrap();

            assert_eq!(fsm.total_used(), 0);
            assert_eq!(fsm.id_minter.read().unwrap().max_block, 0);
            assert_eq!(
                fsm.claim_id(&ctx, reserved_id),
                Err(FsmError::IdOutOfRange(reserved_id))
            );
            assert_eq!(
                fsm.mark_free(&ctx, reserved_id),
                Err(FsmError::IdOutOfRange(reserved_id))
            );

            let expected_ids: Vec<u32> = (0..6).filter(|&id| id != reserved_id).take(5).collect();
            for &id in &expected_ids {
                assert_eq!(fsm.next_id(&ctx).unwrap().id(), id);
            }
            assert_eq!(fsm.total_used(), expected_ids.len());
            assert_eq!(
                fsm.is_free(&ctx, reserved_id),
                Err(FsmError::IdOutOfRange(reserved_id))
            );

            let mut visited_ids = Vec::new();
            fsm.visit_used(&ctx, |id| {
                visited_ids.push(id);
                true
            })
            .unwrap();
            assert_eq!(visited_ids, expected_ids);

            let deleted_id = expected_ids[1];
            fsm.mark_free(&ctx, deleted_id).unwrap();
            while fsm.fast_free_list.pop().is_some() {}
            assert_eq!(fsm.next_id(&ctx).unwrap().id(), deleted_id);
            assert_eq!(fsm.total_used(), expected_ids.len());

            fsm.mark_free(&ctx, deleted_id).unwrap();
            drop(fsm);
            let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), reserved_id, false, true).unwrap();
            assert_eq!(fsm.total_used(), expected_ids.len() - 1);
            assert_eq!(fsm.next_id(&ctx).unwrap().id(), deleted_id);
            assert_eq!(
                fsm.next_id(&ctx).unwrap().id(),
                expected_ids.last().unwrap() + 1
            );
            assert_eq!(fsm.total_used(), expected_ids.len() + 1);

            if reserved_id < BLOCK_SIZE_IDS as u32 {
                let block = store
                    .get(
                        ctx.term(Term::Metadata).get(),
                        &FreeSpaceMap::block_key(0).to_ne_bytes(),
                    )
                    .unwrap();
                let (_, byte_index, bit_index) = fsm.indexes_for_id(reserved_id);
                assert!(!super::bit_used(block[byte_index], bit_index));
            }
        }
    }

    #[test]
    fn reserved_max_id_does_not_overflow() {
        let store = Store::new();
        let ctx = Context::new(0);
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), u32::MAX, false, true).unwrap();
        fsm.id_minter.write().unwrap().next_id = u32::MAX;

        assert!(matches!(
            fsm.next_id(&ctx),
            Err(FsmError::IdOutOfRange(u32::MAX))
        ));
        assert_eq!(fsm.total_used(), 0);
        assert_eq!(fsm.id_minter.read().unwrap().max_block, 0);
    }
}
