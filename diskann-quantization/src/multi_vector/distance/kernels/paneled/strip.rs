// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The default [`Scratch`]: one A-panel × one B-tile, `R` rows A-major, carved into
//! `N`-column [`Block`]s. Borrows its buffer, so the allocator stays at the call site.

use core::mem::MaybeUninit;

use super::{Scratch, ScratchAt};
use crate::alloc::{AllocatorCore, Poly};

/// Marker for element types where all-zero is a valid value.
pub(crate) trait ZeroInit: Copy {}
impl ZeroInit for i32 {}
impl ZeroInit for f32 {}

pub(crate) struct Strip<'a, T, const R: usize, const N: usize>(&'a mut [T]);

/// One B-panel's slot: `R` rows by `N` columns (fewer inside a [`Short`]).
pub(crate) struct Block<'a, T, const R: usize, const N: usize>(&'a mut [T]);

/// A finished [`Strip`], narrowed to its live columns.
pub(crate) struct StripRef<'a, T, const R: usize>(&'a [T]);

/// A short trailing slot: same payload, different type, so the runtime-width path is
/// a separate [`Accumulate`](super::Accumulate) impl.
#[derive(Clone, Copy)]
pub(crate) struct Short<P>(pub(crate) P);

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
    type Block = Block<'a, T, R, N>;
    type Short = Short<Block<'a, T, R, N>>;
    type Ref = StripRef<'a, T, R>;
}

impl<T, const R: usize, const N: usize> Scratch for Strip<'_, T, R, N> {
    fn split(
        &mut self,
        cols: usize,
    ) -> (
        impl Iterator<Item = Block<'_, T, R, N>>,
        Option<Short<Block<'_, T, R, N>>>,
    ) {
        debug_assert!(
            cols <= self.cols_capacity(),
            "strip must hold the whole B-tile"
        );
        let (live, _) = self.0.split_at_mut(R * cols);
        let (head, rest) = live.split_at_mut(R * (cols - cols % N));
        (
            head.chunks_mut(R * N).map(Block),
            (!rest.is_empty()).then_some(Short(Block(rest))),
        )
    }

    fn as_ref(&self, cols: usize) -> StripRef<'_, T, R> {
        StripRef(&self.0[..R * cols])
    }
}

impl<T, const R: usize, const N: usize> Block<'_, T, R, N> {
    pub(crate) fn cols(&self) -> usize {
        self.0.len() / R
    }
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
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
