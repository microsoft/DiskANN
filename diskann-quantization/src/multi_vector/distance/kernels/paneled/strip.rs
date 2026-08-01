// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The default [`Scratch`]: one A-panel × one B-tile, `R` rows A-major, carved into
//! `N`-column [`Block`]s. Borrows its buffer, so the allocator stays at the call site.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::{Scratch, ScratchAt, TailIterator};
use crate::alloc::{AllocatorCore, Poly};
use crate::bits::{Dynamic, Length, Static};

/// Marker for element types where all-zero is a valid value.
pub(crate) trait ZeroInit: Copy {}
impl ZeroInit for i32 {}
impl ZeroInit for f32 {}

/// Owns the accumulator's live region as a checked slice — the anchor the unchecked
/// [`Block`]s below are derived from.
pub(crate) struct Strip<'a, T, const R: usize, const N: usize>(&'a mut [T]);

/// One B-panel's slot: `R` rows by `L` columns, where `L.value() <= N`. A full slot is
/// `Static<N>` (a ZST, so the whole handle is one pointer); the trailing slot is
/// `Dynamic` in `1..N`, and its distinct type selects the short
/// [`Accumulate`](super::Accumulate) impl.
pub(crate) struct Block<'a, T, const R: usize, const N: usize, L: Length> {
    ptr: *mut T,
    cols: L,
    _lifetime: PhantomData<&'a mut [T]>,
}

/// A finished [`Strip`], narrowed to its live columns.
pub(crate) struct StripRef<'a, T, const R: usize>(&'a [T]);

/// Carves a [`Strip`] into per-B-panel slots. The trailer comes off the same cursor.
///
/// # Safety invariants
///
/// `ptr` is valid for writes of `R * (full * N + tail_cols)` `T` for `'a`, and only
/// ever advances — so every slot handed out is disjoint from every other.
pub(crate) struct BlockIter<'a, T, const R: usize, const N: usize> {
    ptr: *mut T,
    full: usize,
    tail_cols: usize,
    _lifetime: PhantomData<&'a mut [T]>,
}

impl<'a, T, const R: usize, const N: usize, L: Length> Block<'a, T, R, N, L> {
    /// # Safety
    ///
    /// `ptr` must be valid for writes of `R * cols.value()` `T` for `'a`, must not
    /// alias any other live `Block`, and `cols.value()` must be at most `N`.
    unsafe fn new(ptr: *mut T, cols: L) -> Self {
        debug_assert!(cols.value() <= N, "block wider than its panel");
        Self {
            ptr,
            cols,
            _lifetime: PhantomData,
        }
    }
}

impl<'a, T, const R: usize, const N: usize> Iterator for BlockIter<'a, T, R, N> {
    type Item = Block<'a, T, R, N, Static<N>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.full == 0 {
            return None;
        }
        let ptr = self.ptr;
        // SAFETY: the invariant covers `full` more slots of `R * N`, so the bump stays
        // inside the allocation and the yielded slot is disjoint from all later ones.
        self.ptr = unsafe { self.ptr.add(R * N) };
        self.full -= 1;
        // SAFETY: as above — `ptr` covers exactly `R * N` writable `T`.
        Some(unsafe { Block::new(ptr, Static) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.full, Some(self.full))
    }
}

impl<T, const R: usize, const N: usize> ExactSizeIterator for BlockIter<'_, T, R, N> {}

impl<'a, T, const R: usize, const N: usize> TailIterator for BlockIter<'a, T, R, N> {
    type Tail = Block<'a, T, R, N, Dynamic>;

    fn tail(self) -> Option<Self::Tail> {
        debug_assert_eq!(self.full, 0, "tail taken before the blocks are exhausted");
        // SAFETY: the blocks are exhausted, so the cursor sits on the trailer, which
        // the invariant covers for `R * tail_cols` writable `T`.
        (self.tail_cols > 0).then(|| unsafe { Block::new(self.ptr, Dynamic(self.tail_cols)) })
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

impl<T, const R: usize, const N: usize> Strip<'_, T, R, N> {
    fn cols_capacity(&self) -> usize {
        self.0.len() / R
    }
}

impl<'a, T, const R: usize, const N: usize> ScratchAt<'a> for Strip<'_, T, R, N> {
    type Block = Block<'a, T, R, N, Static<N>>;
    type Short = Block<'a, T, R, N, Dynamic>;
    type Blocks = BlockIter<'a, T, R, N>;
    type Ref = StripRef<'a, T, R>;
}

impl<T, const R: usize, const N: usize> Scratch for Strip<'_, T, R, N> {
    fn blocks(&mut self, cols: usize) -> BlockIter<'_, T, R, N> {
        debug_assert!(
            cols <= self.cols_capacity(),
            "strip must hold the whole B-tile"
        );
        // The checked slice is what establishes `BlockIter`'s invariant: `R * cols`
        // elements are live, and the `&mut self` borrow keeps them exclusive for `'_`.
        BlockIter {
            ptr: self.0.as_mut_ptr(),
            full: cols / N,
            tail_cols: cols % N,
            _lifetime: PhantomData,
        }
    }

    fn as_ref(&self, cols: usize) -> StripRef<'_, T, R> {
        StripRef(&self.0[..R * cols])
    }
}

impl<T, const R: usize, const N: usize, L: Length> Block<'_, T, R, N, L> {
    pub(crate) fn cols(&self) -> usize {
        self.cols.value()
    }
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T, const R: usize> StripRef<'_, T, R> {
    pub(crate) fn cols(&self) -> usize {
        self.0.len() / R
    }
    pub(crate) fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
}
