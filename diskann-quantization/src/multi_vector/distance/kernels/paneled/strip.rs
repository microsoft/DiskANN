// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The default [`Scratch`]: one buffer of `R` rows A-major, carved into `N`-column
//! [`Block`]s. Every fill re-lends the same memory, so the leaves must store rather
//! than accumulate — MaxSim collapses each tile into the running max before the next
//! one starts. Borrows its buffer, so the allocator stays at the call site.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::{Scratch, SlotsAt};
use crate::alloc::{AllocatorCore, Poly};

/// Marker for element types where all-zero is a valid value.
pub(crate) trait ZeroInit: Copy {}
impl ZeroInit for i32 {}
impl ZeroInit for f32 {}

/// The accumulator buffer, sized to the widest B-tile the plan allows.
pub(crate) struct Strip<'a, T, const R: usize, const N: usize> {
    buf: &'a mut [T],
}

/// One B-panel's slot: `R` rows by `N` columns, A-major.
///
/// Slots are packed, so column `c` of the strip sits at `c * R` whichever slot it falls
/// in — the live columns stay one contiguous run, which is what lets a
/// [`Drain`](super::Drain) fold a whole tile in one pass.
pub(crate) struct Block<'a, T, const R: usize, const N: usize> {
    ptr: *mut T,
    _lifetime: PhantomData<&'a mut [T]>,
}

/// Splits slots off the front of a [`Strip`], so disjointness is the borrow checker's
/// job rather than an invariant to uphold by hand.
pub(crate) struct BlockSlots<'a, T, const R: usize, const N: usize>(&'a mut [T]);

impl<'a, T, const R: usize, const N: usize> Iterator for BlockSlots<'a, T, R, N> {
    type Item = Block<'a, T, R, N>;

    fn next(&mut self) -> Option<Block<'a, T, R, N>> {
        // `take` re-lends the buffer for `'a` rather than the `&mut self` borrow, which
        // is what lets a slot outlive the cursor call that produced it.
        let (slot, rest) = core::mem::take(&mut self.0).split_at_mut_checked(R * N)?;
        self.0 = rest;
        // The split is what proves the slots disjoint, so the pointer inherits that
        // guarantee rather than resting on an invariant upheld by hand.
        Some(Block {
            ptr: slot.as_mut_ptr(),
            _lifetime: PhantomData,
        })
    }
}

impl<'a, T: ZeroInit, const R: usize, const N: usize> Strip<'a, T, R, N> {
    /// Zeroing is what makes the `&mut [T]` sound; the kernels overwrite every live
    /// column before it is read.
    pub(crate) fn from_uninit<A: AllocatorCore + std::fmt::Debug>(
        poly: &'a mut Poly<[MaybeUninit<T>], A>,
        len: usize,
    ) -> Self {
        let ptr = poly.as_mut_ptr().cast::<T>();
        // SAFETY: the poly owns `len` `T`-sized slots; `T: ZeroInit` ⇒ all-zero is a
        // valid `T`, so zeroing initializes every element and the slice is sound.
        let buf = unsafe {
            core::ptr::write_bytes(ptr, 0, len);
            core::slice::from_raw_parts_mut(ptr, len)
        };
        Self { buf }
    }
}

impl<'s, T, const R: usize, const N: usize> SlotsAt<'s> for Strip<'_, T, R, N> {
    type Block = Block<'s, T, R, N>;
    type Slots = BlockSlots<'s, T, R, N>;
}

impl<T, const R: usize, const N: usize> Scratch for Strip<'_, T, R, N> {
    /// A fresh cursor over the whole buffer, so calling it again is the per-tile rewind.
    fn slots(&mut self) -> BlockSlots<'_, T, R, N> {
        BlockSlots(&mut *self.buf)
    }
}

impl<T, const R: usize, const N: usize> Block<'_, T, R, N> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T, const R: usize, const N: usize> Strip<'_, T, R, N> {
    /// The accumulator's live prefix: `R` elements per live B row. Capacity beyond that
    /// belongs to a wider tile, so a consumer handed the whole buffer would fold the
    /// *previous* tile's values back in.
    ///
    /// The caller states `live_rows` because only it knows the B extent; the stride is
    /// the strip's own.
    ///
    /// # Panics
    ///
    /// If the live extent outruns the strip — a planning bug, since the strip is sized
    /// to the widest B-tile the plan allows.
    pub(crate) fn columns(&self, live_rows: usize) -> &[T] {
        &self.buf[..live_rows * R]
    }
}
