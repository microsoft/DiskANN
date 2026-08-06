// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The default [`Scratch`]: one A-panel × one B-tile, `R` rows A-major, carved into
//! `N`-column [`Block`]s. Borrows its buffer, so the allocator stays at the call site.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::{Scratch, ScratchAt};
use crate::alloc::{AllocatorCore, Poly};

/// Marker for element types where all-zero is a valid value.
pub(crate) trait ZeroInit: Copy {}
impl ZeroInit for i32 {}
impl ZeroInit for f32 {}

/// The accumulator buffer, sized to the widest B-tile the plan allows.
pub(crate) struct Strip<'a, T, const R: usize, const N: usize>(&'a mut [T]);

/// One B-panel's slot: `R` rows by `N` columns, A-major.
///
/// Slots are packed, so column `c` of the strip sits at `c * R` whichever slot it falls
/// in — the live columns stay one contiguous run, which is what lets a
/// [`Drain`](super::Drain) fold a whole tile in one pass.
///
/// Thin: `R * N` is already in the type, and the `split_at_mut` that carves the slot is
/// where the length still means something, so the handle keeps only the pointer.
pub(crate) struct Block<'a, T, const R: usize, const N: usize> {
    ptr: *mut T,
    _lifetime: PhantomData<&'a mut [T]>,
}

/// Splits slots off the front of a [`Strip`], so disjointness is the borrow checker's
/// job rather than an invariant to uphold by hand.
pub(crate) struct BlockSlots<'a, T, const R: usize, const N: usize>(&'a mut [T]);

impl<'a, T, const R: usize, const N: usize> super::Slots for BlockSlots<'a, T, R, N> {
    type Block = Block<'a, T, R, N>;

    /// # Panics
    ///
    /// If the strip holds fewer slots than the B-tile has panels — a planning bug, and
    /// the reason this can be infallible everywhere else.
    fn next(&mut self) -> Block<'a, T, R, N> {
        // `take` re-lends the buffer for `'a` rather than the `&mut self` borrow, which
        // is what lets a slot outlive the cursor call that produced it.
        let (slot, rest) = core::mem::take(&mut self.0).split_at_mut(R * N);
        self.0 = rest;
        // `split_at_mut` is what proves the slots disjoint, so the pointer inherits that
        // guarantee rather than resting on an invariant upheld by hand.
        Block {
            ptr: slot.as_mut_ptr(),
            _lifetime: PhantomData,
        }
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
        Self(unsafe {
            core::ptr::write_bytes(ptr, 0, len);
            core::slice::from_raw_parts_mut(ptr, len)
        })
    }
}

impl<'a, T, const R: usize, const N: usize> ScratchAt<'a> for Strip<'_, T, R, N> {
    type Block = Block<'a, T, R, N>;
    type Slots = BlockSlots<'a, T, R, N>;
}

impl<T, const R: usize, const N: usize> Scratch for Strip<'_, T, R, N> {
    fn slots(&mut self) -> BlockSlots<'_, T, R, N> {
        BlockSlots(&mut *self.0)
    }
}

impl<T, const R: usize, const N: usize> Block<'_, T, R, N> {
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T, const R: usize, const N: usize> Strip<'_, T, R, N> {
    /// The live prefix, `cols` columns of `R`. The rest is capacity left over from a
    /// wider tile, so the caller states what it wants and gets a bounds check rather
    /// than a promise.
    ///
    /// # Panics
    ///
    /// If `cols * R` exceeds the strip — a planning bug, since the strip is sized to the
    /// widest B-tile the plan allows.
    pub(crate) fn columns(&self, cols: usize) -> &[T] {
        &self.0[..cols * R]
    }
}
