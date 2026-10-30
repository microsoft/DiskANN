/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Split and reassignment policies, and the assignment machinery they share.

use std::fmt::Debug;

use diskann_utils::views::rowmajor::Matrix;
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use crate::{ANNError, ivf::region::AsMatRef};

pub(crate) mod kmeans;
pub(crate) mod reassign;

pub(crate) use kmeans::{FirstPoints, KMeansSplit};
pub(crate) use reassign::Linked;

////////////
// Errors //
////////////

#[derive(Debug, thiserror::Error)]
pub(crate) enum SplitError<L> {
    #[error("Split could not retrieve list {0:?} from the region")]
    UnableToRetrieve(L),
    #[error("Parents is empty")]
    EmptyParents,
}

impl<L: Debug + Send + Sync + 'static> From<SplitError<L>> for ANNError {
    #[track_caller]
    fn from(error: SplitError<L>) -> Self {
        Self::new(error)
    }
}

////////////
// Assign //
////////////

/// Chooses a destination for every point of one list, reading points as `V`.
pub(crate) trait Assign<V> {
    /// Assign points to candidates and call `emit` exactly once per member, with the
    /// destination of one of `candidates`.
    fn assign<I>(
        &self,
        members: &[I],
        points: V,
        candidates: &[(usize, &[f32])],
        emit: impl FnMut(I, usize),
    ) where
        I: Copy;
}

/// Assigns each point to its nearest candidate by squared L2.
///
/// The earliest candidate wins ties, including infinite distances.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NearestCentroid;

impl<'a, V> Assign<V> for NearestCentroid
where
    V: AsMatRef<'a, f32>,
{
    fn assign<I: Copy>(
        &self,
        members: &[I],
        points: V,
        candidates: &[(usize, &[f32])],
        mut emit: impl FnMut(I, usize),
    ) {
        // With no candidates there is nothing to assign to.
        let Some((&first, rest)) = candidates.split_first() else {
            return;
        };

        // Naive loop over the members
        for (&id, point) in members.iter().zip(points.as_view().rows()) {
            emit(id, nearest(point, first, rest));
        }
    }
}

/// The candidate nearest to `point`, starting from `default`; the earliest wins ties,
/// including infinite distances.
#[inline(always)]
fn nearest<T>(point: &[T], default: (usize, &[T]), candidates: &[(usize, &[T])]) -> usize
where
    SquaredL2: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T]>,
{
    let mut best = default.0;
    let mut min: f32 = SquaredL2::evaluate(point, default.1);

    for &(dst, centroid) in candidates.iter() {
        let distance: f32 = SquaredL2::evaluate(point, centroid);
        if distance < min {
            min = distance;
            best = dst;
        }
    }

    best
}

/////////////
// Buckets //
/////////////

/// Items recorded in consecutive source buckets, each destined for a key below `count`.
///
/// Invariants:
/// - `items` and `keys` have equal lengths, and every key is less than `count`.
/// - `offsets` starts at zero, is non-decreasing, and ends at most at `items.len()`.
#[derive(Debug, Clone)]
struct Buckets<I> {
    count: usize,
    items: Vec<I>,
    keys: Vec<usize>,
    offsets: Vec<usize>,
}

impl<I: Copy> Buckets<I> {
    /// Buckets for keys below `count`.
    fn new(count: usize) -> Self {
        Self {
            count,
            items: Vec::new(),
            keys: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Add `item`, destined for `key`, to the open source bucket.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not less than the key count.
    fn push(&mut self, item: I, key: usize) {
        assert!(key < self.count, "bucket key {key} is out of range");

        self.items.push(item);
        self.keys.push(key);
    }

    /// Close the open source bucket and return its items.
    fn seal(&mut self) -> &[I] {
        let start = self.offsets[self.offsets.len() - 1];
        self.offsets.push(self.items.len());

        &self.items[start..]
    }

    /// Stable grouping of every recorded item, sealed or not, by key.
    fn group(&self) -> Groups<I> {
        debug_assert_eq!(self.items.len(), self.keys.len());

        let mut offsets = vec![0; self.count + 1];
        for &key in &self.keys {
            offsets[key + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let mut next = offsets.clone();
        let mut items = self.items.clone();

        for (&item, &key) in self.items.iter().zip(&self.keys) {
            // SAFETY: Counting indexed `offsets[key + 1]`, so `key < next.len()`. Key `key`
            // occurs exactly `offsets[key + 1] - offsets[key]` times, so its cursor stays
            // below `offsets[key + 1] <= items.len()`.
            unsafe {
                let slot = next.get_unchecked_mut(key);
                *items.get_unchecked_mut(*slot) = item;
                *slot += 1;
            }
        }

        Groups { items, offsets }
    }
}

/// Items grouped by key; see [`Buckets::group`].
#[derive(Debug)]
struct Groups<I> {
    items: Vec<I>,
    offsets: Vec<usize>,
}

impl<I> Groups<I> {
    /// Items destined for `key`, in recording order.
    ///
    /// # Panics
    ///
    /// Panics if `key` is not less than the key count.
    fn get(&self, key: usize) -> &[I] {
        &self.items[self.offsets[key]..self.offsets[key + 1]]
    }
}

#[cfg(test)]
mod tests {
    use super::Buckets;

    #[test]
    fn buckets_group_stably_by_key() {
        let mut moved = Buckets::new(4);
        for (item, key) in [(10, 2), (11, 0), (12, 2)] {
            moved.push(item, key);
        }
        assert_eq!(moved.seal(), [10, 11, 12]);
        assert!(moved.seal().is_empty());
        moved.push(13, 2);
        assert_eq!(moved.seal(), [13]);
        moved.push(14, 1);

        // Unsealed items are grouped too.
        let incoming = moved.group();
        assert_eq!(incoming.get(0), [11]);
        assert_eq!(incoming.get(1), [14]);
        assert_eq!(incoming.get(2), [10, 12, 13]);
        assert!(incoming.get(3).is_empty());
    }

    #[test]
    #[should_panic(expected = "bucket key 4 is out of range")]
    fn buckets_reject_out_of_range_keys() {
        Buckets::new(4).push(10_u32, 4);
    }
}
