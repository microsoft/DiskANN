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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::views::rowmajor;
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use super::*;

    //-----------------//
    // NearestCentroid //
    //-----------------//

    fn matrix(rows: &[&[f32]]) -> rowmajor::Owned<f32> {
        let ncols = rows.first().map_or(0, |row| row.len());
        rowmajor::Owned::try_from_data(rows.concat().into(), rows.len(), ncols).unwrap()
    }

    fn assign(
        members: &[u32],
        points: rowmajor::Ref<'_, f32>,
        candidates: &[(usize, &[f32])],
    ) -> Vec<(u32, usize)> {
        let mut emitted = Vec::new();
        NearestCentroid.assign(members, points, candidates, |member, key| {
            emitted.push((member, key))
        });
        emitted
    }

    #[test]
    fn nearest_centroid_emits_each_member_with_its_nearest_key() {
        // Member IDs and keys are not positions.
        let candidates: [(usize, &[f32]); 2] = [(7, &[0.0, 0.0]), (3, &[10.0, 10.0])];
        let points = matrix(&[&[1.0, 0.0], &[9.0, 9.0], &[4.0, 4.0]]);
        assert_eq!(
            assign(&[40, 41, 42], points.as_view(), &candidates),
            [(40, 7), (41, 3), (42, 7)]
        );
    }

    #[test]
    fn nearest_centroid_breaks_ties_toward_the_earlier_candidate() {
        // Point 5 is as near 0 as 10.
        let (low, high): (&[f32], &[f32]) = (&[0.0], &[10.0]);
        let points = matrix(&[&[5.0]]);
        for (candidates, expected) in [([(0, low), (1, high)], 0), ([(1, high), (0, low)], 1)] {
            assert_eq!(assign(&[1], points.as_view(), &candidates), [(1, expected)]);
        }
    }

    #[test]
    fn nearest_centroid_without_candidates_emits_nothing() {
        let points = matrix(&[&[1.0]]);
        let candidates: [(usize, &[f32]); 0] = [];
        NearestCentroid.assign(&[1u32], points.as_view(), &candidates, |_, _| {
            panic!("there is nothing to emit")
        });
    }

    /// Compare assignment with a scalar oracle, including first-candidate tie breaking.
    ///
    /// Small integer coordinates keep distances and ties exact across summation orders.
    fn nearest_centroid_fuzz_impl(dim: usize, ntrials: usize, rng: &mut StdRng) {
        let coordinate = Uniform::new_inclusive(-8i8, 8).unwrap();
        let count = Uniform::new_inclusive(0, 12).unwrap();
        for _ in 0..ntrials {
            let (npoints, ncandidates) = (count.sample(rng), count.sample(rng));
            let mut random = |rows: usize| -> Vec<f32> {
                let values = (0..rows * dim).map(|_| f32::from(coordinate.sample(rng)));
                values.collect()
            };
            let (points, centroids) = (random(npoints), random(ncandidates));

            let members: Vec<u32> = (100..).take(npoints).collect();
            let candidates: Vec<(usize, &[f32])> =
                (10..).step_by(3).zip(centroids.chunks(dim)).collect();
            let points = rowmajor::Owned::try_from_data(points.into(), npoints, dim).unwrap();

            let emitted = assign(&members, points.as_view(), &candidates);

            let squared = |x: &[f32], y: &[f32]| -> f32 {
                x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum()
            };
            let expected: Vec<(u32, usize)> = members
                .iter()
                .zip(points.rows())
                .filter_map(|(&member, point)| {
                    let nearest = candidates
                        .iter()
                        .min_by(|a, b| squared(point, a.1).total_cmp(&squared(point, b.1)))?;
                    Some((member, nearest.0))
                })
                .collect();
            assert_eq!(emitted, expected);
        }
    }

    #[test]
    fn nearest_centroid_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x510e527fade682d1);
        for dim in [1, 3, 8, 17] {
            nearest_centroid_fuzz_impl(dim, 25, &mut rng);
        }
    }

    //--------//
    // TryMap //
    //--------//

    const PARALLELISMS: [Parallelism; 2] = [Parallelism::Sequential, Parallelism::Rayon];

    #[test]
    fn try_map_returns_results_in_task_order() {
        for parallelism in PARALLELISMS {
            let squares = parallelism.try_map(100, |i| Ok::<_, ()>(i * i));
            assert_eq!(
                squares,
                Ok((0..100).map(|i| i * i).collect()),
                "{parallelism:?}"
            );
        }
    }

    #[test]
    fn try_map_reports_the_first_failing_task_in_order() {
        // Task 5 is slow to fail, so under Rayon tasks 17 and 40 fail first; the error of
        // task 5 is still the one reported.
        for parallelism in PARALLELISMS {
            let result = parallelism.try_map(100, |i| match i {
                5 => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Err(i)
                }
                17 | 40 => Err(i),
                _ => Ok(i),
            });
            assert_eq!(result, Err(5), "{parallelism:?}");
        }
    }

    #[test]
    fn try_map_without_tasks_runs_nothing() {
        for parallelism in PARALLELISMS {
            let result =
                parallelism.try_map(0, |_| -> Result<(), ()> { panic!("no task should run") });
            assert_eq!(result, Ok(vec![]), "{parallelism:?}");
        }
    }

    //---------//
    // Buckets //
    //---------//

    /// Check every key's complete group, including empty groups.
    #[track_caller]
    fn assert_groups(buckets: &Buckets<u32>, expected: &[Vec<u32>]) {
        assert_eq!(buckets.count, expected.len());
        let groups = buckets.group();
        for (key, items) in expected.iter().enumerate() {
            assert_eq!(groups.get(key), items, "key {key}");
        }
    }

    #[test]
    fn buckets_seal_returns_the_open_bucket() {
        let mut buckets = Buckets::<u32>::new(3);
        assert!(buckets.seal().is_empty(), "a bucket can be sealed empty");

        buckets.push(10, 2);
        buckets.push(11, 0);
        assert_eq!(buckets.seal(), [10, 11]);

        buckets.extend([(12, 1), (13, 1)].into_iter());
        assert_eq!(buckets.seal(), [12, 13]);
        assert!(buckets.seal().is_empty());
    }

    #[test]
    fn buckets_group_is_stable_and_includes_the_open_bucket() {
        let mut buckets = Buckets::<u32>::new(4);
        buckets.extend([(1, 2), (2, 0), (3, 2)].into_iter());
        buckets.seal();
        buckets.extend([(4, 0), (5, 2)].into_iter());

        assert_groups(&buckets, &[vec![2, 4], vec![], vec![1, 3, 5], vec![]]);
    }

    #[test]
    #[should_panic(expected = "bucket key 3 is out of range")]
    fn buckets_push_rejects_out_of_range_keys() {
        Buckets::<u32>::new(3).push(0, 3);
    }

    /// Compare randomly sealed pushes/extends with a flat log, preserving order within
    /// each key.
    fn buckets_group_fuzz_impl(keys: usize, steps: usize, rng: &mut StdRng) {
        let key = Uniform::new(0, keys).unwrap();
        let mut buckets = Buckets::new(keys);
        let mut recorded: Vec<(u32, usize)> = Vec::new();
        // Where the open bucket starts in `recorded`.
        let mut open = 0;

        for _ in 0..steps {
            let next = recorded.len() as u32;
            match rng.random_range(0..4) {
                0 => {
                    let expected: Vec<u32> = recorded[open..].iter().map(|e| e.0).collect();
                    assert_eq!(buckets.seal(), expected);
                    open = recorded.len();
                }
                1 => {
                    let end = next + rng.random_range(0..5);
                    let run: Vec<(u32, usize)> =
                        (next..end).map(|item| (item, key.sample(rng))).collect();
                    buckets.extend(run.iter().copied());
                    recorded.extend(run);
                }
                _ => {
                    let item = (next, key.sample(rng));
                    buckets.push(item.0, item.1);
                    recorded.push(item);
                }
            }
        }

        let expected: Vec<Vec<u32>> = (0..keys)
            .map(|key| {
                recorded
                    .iter()
                    .filter(|e| e.1 == key)
                    .map(|e| e.0)
                    .collect()
            })
            .collect();
        assert_groups(&buckets, &expected);
    }

    #[test]
    fn buckets_group_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x9b05688c2b3e6c1f);
        for keys in [1, 2, 7, 64] {
            buckets_group_fuzz_impl(keys, 200, &mut rng);
        }
    }
}
