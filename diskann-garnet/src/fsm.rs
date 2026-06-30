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

/// Guard returned by `next_id()` to ensure correctness when reuse is enabled.
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

/// The free space map manages an ID pool in Garnet to track which IDs are being
/// used. This is necessary to reuse deleted vectors IDs, which is needed due to
/// the small ID space of u32.
///
/// Users can call `next_id()` to get an ID which may be a newly minted ID or a
/// reused one from a previously deleted vector. Use `mark_used()` and
/// `mark_deleted()` to flag IDs as used or free respectively.
///
/// When quantization backfill is required, set `reuse_enabled` to `false` and
/// `quantization_enabled` to true in the constructor. This will prevent all ID
/// reuse until `enable_reuse()` is called. At the start of backfill,
/// `max_id_for_backfill()` can be used to set the upper bound of the backfill
/// range.
pub(crate) struct FreeSpaceMap {
    /// Garnet callbacks for reading/writing FSM keys
    callbacks: Callbacks,
    /// A flag to signal whether there are free IDs in the FSM.
    /// This is set after a scan of the FSM, and is used to prevent extraneous reads
    /// of FSM blocks.
    has_free_ids: AtomicBool,
    /// A queue of previously deleted IDs to prevent excessive reads to the FSM
    fast_free_list: ArrayQueue<u32>,
    /// Controls minting new IDs and expanding the FSM blocks
    id_minter: RwLock<IdMinter>,
    /// The total number of IDs marked used in the FSM
    total_used: AtomicUsize,
    /// Controls when ID reuse for deleted IDs is enabled.
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
            .exists_wid(&ctx.term(Term::Metadata), block_key)
        {
            this.load_state(ctx)?;
        } else {
            // Allocate first block.
            let (block_id, _, _) = this.indexes_for_id(0);
            let mut id_minter = this.id_minter.write().unwrap();
            this.expand_to(&mut id_minter, ctx, block_id)?;
        }

        Ok(this)
    }

    /// Load all state from Garnet by scanning the FSM blocks.
    fn load_state(&mut self, ctx: &Context) -> Result<(), FsmError> {
        let mut max_block_id = 0;
        while self
            .callbacks
            .exists_wid(&ctx.term(Term::Metadata), Self::block_key(max_block_id))
        {
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

            let mut id = block_id * BLOCK_SIZE_IDS as u32 + BLOCK_SIZE_IDS as u32 - 1;

            for &byte in block.iter().rev() {
                for bidx in (0..8).rev() {
                    let used = bit_used(byte, bidx);
                    if used {
                        last_used_id = last_used_id.max(id as i64);
                        total_used += 1;
                    } else if (id as i64) < last_used_id {
                        let _ = self.fast_free_list.push(id);
                    }

                    id = id.saturating_sub(1);
                }
            }
        }

        let mut id_minter = self.id_minter.write().unwrap();
        id_minter.max_block = max_block_id - 1;

        id_minter.next_id = (last_used_id + 1) as u32;

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
            if id >= id_minter.next_id {
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
            if id >= id_minter.next_id {
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
        let id = id_minter.next_id;
        id_minter.next_id += 1;
        self.expand_to(&mut id_minter, ctx, id)?;
        self.mark_id_unchecked(ctx, id, true)?;

        Ok(ReuseGuard::new(id, barrier))
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

                    if !bit_used(byte, bidx) {
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

    /// Ensure enough blocks exist in the FSM to hold `id`.
    fn expand_to(
        &self,
        id_minter: &mut RwLockWriteGuard<IdMinter>,
        ctx: &Context,
        id: u32,
    ) -> Result<(), FsmError> {
        let (block_id, _, _) = self.indexes_for_id(id);
        if id_minter.max_block == u32::MAX || block_id == id_minter.max_block + 1 {
            let block_key = Self::block_key(block_id);

            if !self
                .callbacks
                .write_wid(&ctx.term(Term::Metadata), block_key, &id_minter.buffer)
            {
                return Err(FsmError::Garnet(GarnetError::Write));
            }

            if id_minter.max_block == u32::MAX {
                id_minter.max_block = 0;
            } else {
                id_minter.max_block += 1;
            }
        } else if block_id > id_minter.max_block + 1 {
            return Err(FsmError::IdOutOfRange(id));
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
        let mut id = 0u32;

        for block_id in 0..max_block + 1 {
            let block_key = Self::block_key(block_id);
            if !self
                .callbacks
                .read_single_wid(&ctx.term(Term::Metadata), block_key, &mut block)
            {
                return Err(FsmError::Garnet(GarnetError::Read));
            }

            for &byte in &block {
                if byte == 0x00 {
                    id += 8;
                    continue;
                }

                for bidx in 0..8 {
                    if bit_used(byte, bidx) {
                        let keep_going = f(id);
                        if !keep_going {
                            return Ok(());
                        }
                    }
                    id += 1;
                }
            }
        }

        Ok(())
    }

    /// Signal that new IDs should be quantized.
    pub(crate) fn enable_quantization(&self) {
        // Taking the write barrier ensures that we switch quantization phases when
        // 1. No `next_id` is not changing (because `next_id()` operates under a read barrier)
        // 2. All vector data for any pending unquantized inserts has been written. (because
        //     `insert()` updates data under the same read barrier via `ReuseGuard`)
        let mut guard = self.barrier.write().unwrap();
        guard.max_id_for_backfill = self.max_id();
        guard.quantization_enabled = true;
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
    use std::sync::atomic::Ordering;

    use crate::{
        fsm::{BLOCK_SIZE_BYTES, BLOCK_SIZE_IDS, FreeSpaceMap, FsmError},
        garnet::{Context, Term},
        test_utils::Store,
    };

    fn verify_block<F>(store: &Store, block_id: u32, mut f: F)
    where
        F: FnMut(&[u8]),
    {
        let ctx = Context::new(0);
        let key = FreeSpaceMap::block_key(block_id);
        let block = store
            .get(ctx.term(Term::Metadata).get(), bytemuck::bytes_of(&key))
            .unwrap();
        f(&block);
    }

    #[test]
    fn create_fresh() {
        let store = Store;
        let ctx = Context::new(0);

        // A fresh FSM should result in full block of all zeroes.
        let _fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();
        verify_block(&store, 0, |block| {
            assert_eq!(block.len(), BLOCK_SIZE_BYTES);
            assert_eq!(block.iter().filter(|b| **b != 0).count(), 0);
        });
    }

    #[test]
    fn basic_next_id() {
        let store = Store;
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();

        // A fresh FSM should not have free IDs.
        assert!(!fsm.has_free_ids.load(std::sync::atomic::Ordering::Acquire));

        // It should return sequentials IDs starting from 0.
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 0);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 1);
    }

    #[test]
    fn basic_delete() {
        let store = Store;
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();

        // Deleting a ID out of range should return an error.
        let res = fsm.mark_free(&ctx, 0);
        assert!(res.is_err());
        assert_eq!(res.err().unwrap(), FsmError::IdOutOfRange(0));

        for _ in 0u32..64 {
            let _ = fsm.next_id(&ctx).unwrap();
        }

        verify_block(&store, 0, |block| {
            for b in &block[0..8] {
                assert_eq!(*b, 0b11111111);
            }
            for b in &block[8..BLOCK_SIZE_BYTES] {
                assert_eq!(*b, 0b00000000);
            }
        });

        fsm.mark_free(&ctx, 37).unwrap();
        fsm.mark_free(&ctx, 9).unwrap();

        verify_block(&store, 0, |block| {
            assert_eq!(block[0], 0b11111111);
            assert_eq!(block[1], 0b10111111);

            for b in &block[2..4] {
                assert_eq!(*b, 0b11111111);
            }
            assert_eq!(block[4], 0b11111011);
            for b in &block[5..8] {
                assert_eq!(*b, 0b11111111);
            }
            for b in &block[8..BLOCK_SIZE_BYTES] {
                assert_eq!(*b, 0b00000000);
            }
        });
    }

    #[test]
    fn basic_id_reuse() {
        let store = Store;
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();

        for _ in 0u32..64 {
            let _ = fsm.next_id(&ctx).unwrap();
        }

        // Next ID should be 64 since none are free before that.
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 64);

        // After deleting an ID, it should be returned from `next_id()`.
        fsm.mark_free(&ctx, 37).unwrap();
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 37);
        // Once all free IDs are used, fresh ones should be returned.
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 65);
        // And has_free_ids should now be false.
        assert!(!fsm.has_free_ids.load(Ordering::Acquire));
    }

    #[test]
    fn basic_recovery() {
        let store = Store;
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();

        for _ in 0u32..64 {
            let _ = fsm.next_id(&ctx).unwrap();
        }

        fsm.mark_free(&ctx, 37).unwrap();

        // Loading FSM from store should recover all the state.
        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();
        assert_eq!(fsm.id_minter.read().unwrap().max_block, 0);
        assert_eq!(fsm.max_id() + 1, 64);
        assert!(fsm.has_free_ids.load(Ordering::Acquire));
        assert_eq!(fsm.fast_free_list.len(), 1);
        assert_eq!(fsm.next_id(&ctx).unwrap().id(), 37);
    }

    #[test]
    fn dynamic_expansion() {
        let store = Store;
        let ctx = Context::new(0);

        let fsm = FreeSpaceMap::new(&ctx, store.callbacks(), false, true).unwrap();

        // Asking for more than BLOCK_SIZE_IDS will force another FSM block to be allocated.
        for _ in 0u32..BLOCK_SIZE_IDS as u32 + 1 {
            let _ = fsm.next_id(&ctx).unwrap();
        }
    }
}
