/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Split and reassignment policies, and the machinery they share.

use std::fmt::Debug;

use diskann_quantization::Parallelism;
use diskann_utils::views::rowmajor::Matrix;
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{ANNError, ivf::region::AsMatRef};

pub(crate) mod kmeans;
pub(crate) mod reassign;

pub(crate) use kmeans::{FirstPoints, KMeansSplit};
pub(crate) use reassign::Linked;

////////////
// Errors //
////////////

/// Errors reported by the split and reassignment policies.
#[derive(Debug, thiserror::Error)]
pub(crate) enum SplitError<L> {
    /// The list was not filled into the region.
    #[error("Split could not retrieve list {0:?} from the region")]
    UnableToRetrieve(L),
    /// A split was requested with no parents.
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

/// Chooses a destination for every point in `V`.
pub(crate) trait Assign<V, T>: Sync {
    /// Call `emit(member, key)` once per member, choosing `key` from the `(key, centroid)`
    /// pairs of `candidates`. With no candidates, nothing is emitted.
    fn assign<I>(
        &self,
        members: &[I],
        points: V,
        candidates: &[(usize, &[T])],
        emit: impl FnMut(I, usize),
    ) where
        I: Copy;
}

/// Naive double-loop assignment of each point to its nearest candidate by squared L2.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NearestCentroid;

impl<'a, V, T> Assign<V, T> for NearestCentroid
where
    V: AsMatRef<'a, f32>,
    SquaredL2: for<'c, 'd> PureDistanceFunction<&'c [f32], &'d [T]>,
{
    fn assign<I: Copy>(
        &self,
        members: &[I],
        points: V,
        candidates: &[(usize, &[T])],
        mut emit: impl FnMut(I, usize),
    ) {
        // With no candidates there is nothing to assign to.
        let Some((&first, rest)) = candidates.split_first() else {
            return;
        };

        // Naive loop over the members
        for (&id, point) in members.iter().zip(points.as_view().rows()) {
            let mut best = first.0;
            let mut min: f32 = SquaredL2::evaluate(point, first.1);

            for &(dst, centroid) in rest.iter() {
                let distance: f32 = SquaredL2::evaluate(point, centroid);
                if distance < min {
                    min = distance;
                    best = dst;
                }
            }

            emit(id, best);
        }
    }
}

/////////////////
// Parallelism //
/////////////////

/// A simple trait to run iterator plus collection operations based on [`Parallelism`].
trait TryMap {
    /// Return `f(i)` for every task `i` in `0..tasks`, in task order, or the error of the
    /// first task in that order that fails.
    fn try_map<T, E, F>(self, tasks: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize) -> Result<T, E> + Sync + Send;
}

impl TryMap for Parallelism {
    fn try_map<T, E, F>(self, tasks: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize) -> Result<T, E> + Sync + Send,
    {
        match self {
            // Collect every result first: collecting straight into a `Result` would report
            // whichever error Rayon found first, not the first in task order.
            Parallelism::Rayon => {
                let results: Vec<_> = (0..tasks).into_par_iter().map(f).collect();
                results.into_iter().collect()
            }
            Parallelism::Sequential | _ => (0..tasks).map(f).collect(),
        }
    }
}

/////////////
// Buckets //
/////////////

/// Items recorded in consecutive source buckets, each mapped to a key below `count`.
///
/// # Invariants
///
/// * `items` and `keys` have equal lengths, and every key is less than `count`.
/// * `offsets` starts at zero, is non-decreasing, and ends at most at `items.len()`.
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

    /// Extend the current bucket with `(item, key)` pairs from an interator.
    ///
    /// # Panics
    ///
    /// Panics if any `key` is not less than the key count.
    fn extend<Itr>(&mut self, itr: Itr)
    where
        Itr: Iterator<Item = (I, usize)>,
    {
        for (item, key) in itr {
            self.push(item, key);
        }
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
