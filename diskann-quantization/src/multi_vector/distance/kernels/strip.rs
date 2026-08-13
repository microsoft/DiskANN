// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! The accumulator a fill writes and a drain reads.
//!
//! Rows are query rows and one column is one doc — note that a doc arrives as a *row* of
//! the input and lands as a *column* here. Column-major over the whole strip: column `c`
//! occupies `[c * AR, (c + 1) * AR)`, and slot `p` covers columns `p * BR ..`, so a drain
//! can address a run of columns without knowing which slot produced them.

use core::slice::ChunksExactMut;

use super::{Scratch, SlotsAt};

/// Accumulator for one A-panel against one whole B-tile.
///
/// Holds no cursor: [`Scratch::slots`] restarts from the front on every fill, which is
/// what lets the same memory be re-lent across tiles without clearing.
pub(super) struct Strip<'a, T, const AR: usize, const BR: usize> {
    buf: &'a mut [T],
}

impl<'a, T, const AR: usize, const BR: usize> Strip<'a, T, AR, BR> {
    /// Trailing elements beyond the last whole slot are never touched.
    pub(super) fn new(buf: &'a mut [T]) -> Self {
        Self { buf }
    }

    /// The first `live` columns — the only region the fill just performed wrote.
    ///
    /// # Panics
    ///
    /// Panics if `live` exceeds the strip's column capacity.
    pub(super) fn columns(&self, live: usize) -> &[T] {
        &self.buf[..live * AR]
    }
}

/// One `AR × BR` accumulator tile, column-major.
pub(super) struct Slot<'a, T, const AR: usize, const BR: usize> {
    buf: &'a mut [T],
}

impl<T, const AR: usize, const BR: usize> Slot<'_, T, AR, BR> {
    /// Exactly `AR * BR` elements: [`Slots`] only ever cuts whole tiles.
    pub(super) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut *self.buf
    }
}

/// Cuts a strip into disjoint slots, stopping short of a trailing partial tile.
///
/// Disjointness and that stopping rule are both structural, from `chunks_exact_mut`.
pub(super) struct Slots<'a, T, const AR: usize, const BR: usize>(ChunksExactMut<'a, T>);

impl<'a, T, const AR: usize, const BR: usize> Iterator for Slots<'a, T, AR, BR> {
    type Item = Slot<'a, T, AR, BR>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|buf| Slot { buf })
    }
}

impl<'s, T, const AR: usize, const BR: usize> SlotsAt<'s> for Strip<'_, T, AR, BR> {
    type Slot = Slot<'s, T, AR, BR>;
    type Slots = Slots<'s, T, AR, BR>;
}

impl<T, const AR: usize, const BR: usize> Scratch for Strip<'_, T, AR, BR> {
    fn slots(&mut self) -> Slots<'_, T, AR, BR> {
        Slots(self.buf.chunks_exact_mut(AR * BR))
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
            slot.as_mut_slice().fill(n as u32 + 1);
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
        assert_eq!(strip.columns(3), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert_eq!(strip.columns(0), &[] as &[u32]);
    }
}
