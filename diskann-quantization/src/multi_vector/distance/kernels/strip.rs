/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The accumulator a fill writes and a drain reads.
//!
//! Storage is a flat run of `AR`-element chunks. One chunk is what the rest of the module
//! calls a *column*. Read as memory the strip is therefore row-major with the axes
//! flipped: an `n × AR` matrix whose rows are the accumulator's columns.
//!
//! ```text
//! AR = 4, BR = 2:
//!
//!   memory ->  [ a0 a1 a2 a3 ][ a0 a1 a2 a3 ][ a0 a1 a2 a3 ][ a0 a1 a2 a3 ]
//!                 column 0       column 1       column 2       column 3
//!              \___________ slot 0 _________/\___________ slot 1 _________/
//! ```
//!
//! Column `c` occupies `[c * AR, (c + 1) * AR)` and slot `p` covers columns `p * BR ..`,
//! so [`Strip::columns`] can hand a drain a run that straddles slot boundaries. What the
//! two axes *mean* is the drain's business, not the strip's. See
//! [`RawMax`](super::float::RawMax).

use core::mem;

use super::{Scratch, SlotsAt};

/// Accumulator for one A-panel against one whole B-tile, carved by its [`Scratch`]
/// impl into one [`Slot`] per B-panel.
///
/// Holds no cursor: [`Scratch::slots`] restarts from the front on every fill, which is
/// what lets the same memory be re-lent across tiles without clearing.
pub(super) struct Strip<'a, T, const AR: usize, const BR: usize> {
    buf: &'a mut [[T; AR]],
}

impl<'a, T, const AR: usize, const BR: usize> Strip<'a, T, AR, BR> {
    /// Trailing elements beyond the last whole column are never touched.
    pub(super) fn new(buf: &'a mut [T]) -> Self {
        Self {
            buf: buf.as_chunks_mut::<AR>().0,
        }
    }

    /// The first `live` columns, the region the fill just performed wrote.
    ///
    /// # Panics
    ///
    /// Panics if `live` exceeds the strip's column capacity.
    pub(super) fn columns(&self, live: usize) -> &[[T; AR]] {
        &self.buf[..live]
    }
}

/// The `BR` consecutive columns of a [`Strip`] that one leaf call accumulates into.
pub(super) struct Slot<'a, T, const AR: usize, const BR: usize> {
    buf: &'a mut [[T; AR]; BR],
}

impl<T, const AR: usize, const BR: usize> Slot<'_, T, AR, BR> {
    pub(super) fn columns(&mut self) -> &mut [[T; AR]; BR] {
        &mut *self.buf
    }
}

/// Cuts a [`Strip`] into disjoint [`Slot`]s, stopping short of a trailing partial tile.
///
/// Disjointness and that stopping rule are both structural, from `split_first_chunk_mut`.
pub(super) struct Slots<'a, T, const AR: usize, const BR: usize> {
    rest: &'a mut [[T; AR]],
}

impl<'a, T, const AR: usize, const BR: usize> Iterator for Slots<'a, T, AR, BR> {
    type Item = Slot<'a, T, AR, BR>;

    fn next(&mut self) -> Option<Self::Item> {
        // A reborrow through `&mut self` cannot reach `'a`, so the remainder is moved out
        // and put back. Too short a remainder leaves it empty, which is where it ends.
        let (buf, rest) = mem::take(&mut self.rest).split_first_chunk_mut::<BR>()?;
        self.rest = rest;
        Some(Slot { buf })
    }
}

impl<'s, T, const AR: usize, const BR: usize> SlotsAt<'s> for Strip<'_, T, AR, BR> {
    type Slot = Slot<'s, T, AR, BR>;
    type Slots = Slots<'s, T, AR, BR>;
}

impl<T, const AR: usize, const BR: usize> Scratch for Strip<'_, T, AR, BR> {
    fn slots(&mut self) -> Slots<'_, T, AR, BR> {
        Slots {
            rest: &mut *self.buf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_partition_the_strip_and_stop_short_of_a_partial_tile() {
        let mut buf = [0u32; 4 * 2 * 3 + 5];
        let mut strip = Strip::<u32, 4, 2>::new(&mut buf);

        for (n, mut slot) in strip.slots().enumerate() {
            slot.columns().as_flattened_mut().fill(n as u32 + 1);
        }
        assert_eq!(strip.slots().count(), 3);

        // The 5 trailing elements are shorter than a slot, so no slot covers them.
        assert_eq!(&buf[..8], &[1; 8]);
        assert_eq!(&buf[16..24], &[3; 8]);
        assert_eq!(&buf[24..], &[0; 5]);
    }

    #[test]
    fn columns_cut_ignores_slot_boundaries() {
        let mut buf: [u32; 24] = core::array::from_fn(|i| i as u32);
        let strip = Strip::<u32, 4, 2>::new(&mut buf);

        // Three columns straddle the first slot (2 columns) into the second.
        assert_eq!(
            strip.columns(3),
            &[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        );
        assert_eq!(strip.columns(0), &[] as &[[u32; 4]]);
    }
}
