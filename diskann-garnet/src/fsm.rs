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
//!
//! The atomic `next_id` also prevents minting duplicate IDs to concurrent
//! inserts. While this operation could be done via RMW and retry on conflict
//! detection, the atomic is much faster and just as accurate since this code is
//! the only place where the FSM is read or written to.

use crossbeam::queue::ArrayQueue;
use std::sync::{
    RwLock,
    atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
};
use thiserror::Error;

use crate::garnet::{Callbacks, Context, GarnetError, Term};

const BLOCK_SIZE_IDS: usize = 2usize.pow(16);
const BLOCK_SIZE_BYTES: usize = BLOCK_SIZE_IDS / 8;
const FAST_SIZE: usize = 1024;
// Full FSM key will be the prefix + u32 index of the block
const FSM_KEY_PREFIX: u32 = u32::from_be_bytes(*b"_fsm");

#[derive(Debug, Error, PartialEq)]
pub enum FsmError {
    #[error("Garnet operation failed")]
    Garnet(#[from] GarnetError),
    #[error("requested ID is out of range {0}")]
    IdOutOfRange(u32),
}

pub struct FreeSpaceMap {
    callbacks: Callbacks,
    has_free_ids: AtomicBool,
    fast_free_list: ArrayQueue<u32>,
    max_block: RwLock<u32>,
    next_id: AtomicU32,
    total_used: AtomicUsize,
}

impl FreeSpaceMap {
    pub fn new(ctx: Context, callbacks: Callbacks) -> Result<Self, FsmError> {
        let has_free_ids = AtomicBool::new(false);
        let fast_free_list = ArrayQueue::new(FAST_SIZE);
        let max_block = RwLock::new(u32::MAX);
        let next_id = AtomicU32::new(0);
        let total_used = AtomicUsize::new(0);

        let mut this = Self {
            callbacks,
            has_free_ids,
            fast_free_list,
            max_block,
            next_id,
            total_used,
        };

        // Attempt to load state from Garnet.
        let block_key = Self::block_key(0);
        if this
            .callbacks
            .exists_wid(ctx.term(Term::Metadata), block_key)
        {
            this.load_state(ctx)?;
        } else {
            // Allocate first block.
            let (block_id, _, _) = this.indexes_for_id(0);
            this.expand_to(ctx, block_id)?;
        }

        Ok(this)
    }

    /// Load all state from Garnet by scanning the FSM blocks.
    fn load_state(&mut self, ctx: Context) -> Result<(), FsmError> {
        let mut max_block_id = 0;
        while self
            .callbacks
            .exists_wid(ctx.term(Term::Metadata), Self::block_key(max_block_id))
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
                .read_single_wid(ctx.term(Term::Metadata), block_key, &mut block)
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

        let mut max_block = self.max_block.write().unwrap();
        *max_block = max_block_id - 1;

        self.next_id
            .store((last_used_id + 1) as u32, Ordering::Release);

        self.total_used.store(total_used, Ordering::Release);

        if !self.fast_free_list.is_empty() {
            self.has_free_ids.store(true, Ordering::Release);
        }

        Ok(())
    }

    /// Mark an ID as free.
    pub fn mark_free(&self, ctx: Context, id: u32) -> Result<(), FsmError> {
        // We don't care about the changed status on free.
        self.mark_id(ctx, id, false).map(|_| ())
    }

    /// Mark an ID as occupied.
    fn mark_used(&self, ctx: Context, id: u32) -> Result<bool, FsmError> {
        self.mark_id(ctx, id, true)
    }

    /// Mark an ID according to value (true = used, false = free).
    /// Side effects only happen if the value actually changed. The return value reflects whether
    /// data changed or not.
    fn mark_id(&self, ctx: Context, id: u32, used: bool) -> Result<bool, FsmError> {
        if id >= self.next_id.load(Ordering::Acquire) {
            return Err(FsmError::IdOutOfRange(id));
        }

        let (block_id, byte_idx, bit_idx) = self.indexes_for_id(id);
        let max_block = *self.max_block.read().unwrap();
        if block_id > max_block || max_block == u32::MAX {
            return Err(FsmError::Garnet(GarnetError::Read));
        }

        let block_key = Self::block_key(block_id);
        let mut changed = false;

        if !self.callbacks.rmw_wid(
            ctx.term(Term::Metadata),
            block_key,
            BLOCK_SIZE_BYTES,
            |data: &mut [u8]| changed = update_status(used, &mut data[byte_idx], bit_idx),
        ) {
            return Err(FsmError::Garnet(GarnetError::Write));
        }

        if changed {
            if used {
                self.total_used.fetch_add(1, Ordering::Release);
            } else {
                self.total_used.fetch_sub(1, Ordering::Release);
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

    pub fn is_free(&self, ctx: Context, id: u32) -> Result<bool, FsmError> {
        if id >= self.next_id.load(Ordering::Acquire) {
            return Err(FsmError::IdOutOfRange(id));
        }

        let (block_id, byte_idx, bit_idx) = self.indexes_for_id(id);
        let max_block = *self.max_block.read().unwrap();
        if block_id > max_block || max_block == u32::MAX {
            return Err(FsmError::Garnet(GarnetError::Read));
        }

        let block_key = Self::block_key(block_id);
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];

        if !self
            .callbacks
            .read_single_wid(ctx.term(Term::Metadata), block_key, &mut block)
        {
            return Err(FsmError::Garnet(GarnetError::Read));
        }

        let free = !bit_used(block[byte_idx], bit_idx);

        Ok(free)
    }

    /// Return a a new ID.
    /// This may be a a fresh ID larger than all the others, or it may be a reused ID that
    /// previously belonged to a deleted element. The returned ID is marked as used.
    pub fn next_id(&self, ctx: Context) -> Result<u32, FsmError> {
        if self.has_free_ids.load(Ordering::Acquire) {
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
                    return Ok(id);
                }

                break;
            }
        }

        // Retry minting a new ID until we are successfully able to mark it used.
        let mut changed = false;
        let mut id = 0u32;
        while !changed {
            id = self.next_id.fetch_add(1, Ordering::AcqRel);

            // Make sure we have enough FSM blocks for this id.
            let (block_id, _, _) = self.indexes_for_id(id);
            let max_block = { *self.max_block.read().unwrap() };
            if block_id > max_block {
                self.expand_to(ctx, id)?;
            }

            changed = self.mark_used(ctx, id)?;
        }

        Ok(id)
    }

    pub fn max_id(&self) -> u32 {
        self.next_id.load(Ordering::Acquire).saturating_sub(1)
    }

    pub fn total_used(&self) -> usize {
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
    fn refill_fast_free_list(&self, ctx: Context) -> Result<bool, FsmError> {
        // NOTE: We take a write lock to prevent multiple refills happening simultaneously.
        #[allow(clippy::readonly_write_lock)]
        let max_block = self.max_block.write().unwrap();

        // If we had to wait to acquire the lock, it's possible some else refilled the list first, so check it again.
        if !self.fast_free_list.is_empty() {
            return Ok(true);
        }

        let mut has_free_ids = false;
        let mut id = 0u32;
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];
        'scan: for block_id in 0..*max_block {
            let block_key = Self::block_key(block_id);
            if !self
                .callbacks
                .read_single_wid(ctx.term(Term::Metadata), block_key, &mut block)
            {
                return Err(FsmError::Garnet(GarnetError::Read));
            }

            for &byte in &block {
                if byte == 0xff {
                    id += 8;
                    continue;
                }

                for bidx in 0..8 {
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
    fn expand_to(&self, ctx: Context, id: u32) -> Result<(), FsmError> {
        let (block_id, _, _) = self.indexes_for_id(id);
        let mut max_block = self.max_block.write().unwrap();
        if *max_block == u32::MAX || block_id == *max_block + 1 {
            let block_key = Self::block_key(block_id);
            let block_bytes = vec![0u8; BLOCK_SIZE_BYTES];

            if !self
                .callbacks
                .write_wid(ctx.term(Term::Metadata), block_key, &block_bytes)
            {
                return Err(FsmError::Garnet(GarnetError::Write));
            }

            if *max_block == u32::MAX {
                *max_block = 0;
            } else {
                *max_block += 1;
            }
        } else if block_id > *max_block + 1 {
            return Err(FsmError::IdOutOfRange(id));
        }

        Ok(())
    }

    /// Visit each used id in the FSM, invoking f on each id.
    pub fn visit_used<F>(&self, ctx: Context, mut f: F) -> Result<(), FsmError>
    where
        F: FnMut(u32) -> bool,
    {
        let max_block = { *self.max_block.read().unwrap() };
        let mut block = vec![0u8; BLOCK_SIZE_BYTES];
        let mut id = 0u32;

        for block_id in 0..max_block {
            let block_key = Self::block_key(block_id);
            if !self
                .callbacks
                .read_single_wid(ctx.term(Term::Metadata), block_key, &mut block)
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
        let key = FreeSpaceMap::block_key(block_id);
        let block = store
            .get(Context(0).term(Term::Metadata).0, bytemuck::bytes_of(&key))
            .unwrap();
        f(&block);
    }

    #[test]
    fn create_fresh() {
        let store = Store;

        // A fresh FSM should result in full block of all zeroes.
        let _fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();
        verify_block(&store, 0, |block| {
            assert_eq!(block.len(), BLOCK_SIZE_BYTES);
            assert_eq!(block.iter().filter(|b| **b != 0).count(), 0);
        });
    }

    #[test]
    fn basic_next_id() {
        let store = Store;

        let fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();

        // A fresh FSM should not have free IDs.
        assert!(!fsm.has_free_ids.load(std::sync::atomic::Ordering::Acquire));

        // It should return sequentials IDs starting from 0.
        assert_eq!(fsm.next_id(Context(0)).unwrap(), 0);
        assert_eq!(fsm.next_id(Context(0)).unwrap(), 1);
    }

    #[test]
    fn basic_delete() {
        let store = Store;

        let fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();

        // Deleting a ID out of range should return an error.
        let res = fsm.mark_free(Context(0), 0);
        assert!(res.is_err());
        assert_eq!(res.err().unwrap(), FsmError::IdOutOfRange(0));

        for _ in 0u32..64 {
            let _ = fsm.next_id(Context(0)).unwrap();
        }

        verify_block(&store, 0, |block| {
            for b in &block[0..8] {
                assert_eq!(*b, 0b11111111);
            }
            for b in &block[8..BLOCK_SIZE_BYTES] {
                assert_eq!(*b, 0b00000000);
            }
        });

        fsm.mark_free(Context(0), 37).unwrap();
        fsm.mark_free(Context(0), 9).unwrap();

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

        let fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();

        for _ in 0u32..64 {
            let _ = fsm.next_id(Context(0)).unwrap();
        }

        // Next ID should be 64 since none are free before that.
        assert_eq!(fsm.next_id(Context(0)).unwrap(), 64);

        // After deleting an ID, it should be returned from `next_id()`.
        fsm.mark_free(Context(0), 37).unwrap();
        assert_eq!(fsm.next_id(Context(0)).unwrap(), 37);
        // Once all free IDs are used, fresh ones should be returned.
        assert_eq!(fsm.next_id(Context(0)).unwrap(), 65);
        // And has_free_ids should now be false.
        assert!(!fsm.has_free_ids.load(Ordering::Acquire));
    }

    #[test]
    fn basic_recovery() {
        let store = Store;

        let fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();

        for _ in 0u32..64 {
            let _ = fsm.next_id(Context(0)).unwrap();
        }

        fsm.mark_free(Context(0), 37).unwrap();

        // Loading FSM from store should recover all the state.
        let fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();
        assert_eq!(*fsm.max_block.read().unwrap(), 0);
        assert_eq!(fsm.next_id.load(Ordering::Acquire), 64);
        assert!(fsm.has_free_ids.load(Ordering::Acquire));
        assert_eq!(fsm.fast_free_list.len(), 1);
        assert_eq!(fsm.next_id(Context(0)).unwrap(), 37);
    }

    #[test]
    fn dynamic_expansion() {
        let store = Store;

        let fsm = FreeSpaceMap::new(Context(0), store.callbacks()).unwrap();

        // Asking for more than BLOCK_SIZE_IDS will force another FSM block to be allocated.
        for _ in 0u32..BLOCK_SIZE_IDS as u32 + 1 {
            let _ = fsm.next_id(Context(0)).unwrap();
        }
    }
}
