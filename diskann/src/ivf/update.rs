/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Declarative partition updates consumed by [`Apply`].
//!
//! Each distinct operation is its own type: [`Bootstrap`], [`Appends`], [`Splits`], and
//! [`Reassignments`]. [`SplitInsert`] composes the last three for an insert batch that
//! splits lists.
//!
//! Only the index constructs updates. It validates everything it reads from the
//! provider or its caller, and its planner guarantees the invariants documented on
//! each type by construction. Providers receive updates through [`Apply`] and can only
//! read them.
//!
//! Point memberships are stored grouped: group keys are ascending and unique, and each
//! group's point ids are ascending in one contiguous slice. Providers can therefore
//! write each list once and process distinct groups in parallel.
//!
//! [`Apply`]: crate::ivf::dynamic::Apply

use std::cmp::Ordering;

use diskann_utils::views::{Matrix, MatrixView};

use crate::{ivf::grouped::Grouped, utils::VectorId};

///////////////////
// CentroidBlock //
///////////////////

/// Freshly reserved centroids with full-precision vectors, one row per id.
///
/// A block is never empty, its ids are fresh and unique, its rows share one non-zero
/// dimension, and every value is finite.
#[derive(Debug, PartialEq)]
pub struct CentroidBlock<L> {
    ids: Box<[L]>,
    vectors: Matrix<f32>,
}

impl<L: VectorId> CentroidBlock<L> {
    /// Pair `ids[i]` with row `i` of `vectors`.
    pub(super) fn new(ids: Vec<L>, vectors: Matrix<f32>) -> Self {
        debug_assert_eq!(ids.len(), vectors.nrows());
        Self {
            ids: ids.into(),
            vectors,
        }
    }

    /// Number of centroids.
    #[expect(clippy::len_without_is_empty, reason = "a block is never empty")]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Dimension of every centroid.
    pub fn dim(&self) -> usize {
        self.vectors.ncols()
    }

    /// Centroid ids in row order.
    pub fn ids(&self) -> &[L] {
        &self.ids
    }

    /// Centroid vectors; row `i` belongs to `self.ids()[i]`.
    pub fn vectors(&self) -> MatrixView<'_, f32> {
        self.vectors.as_view()
    }

    /// Iterate `(id, vector)` pairs in row order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (L, &[f32])> {
        self.ids.iter().copied().zip(self.vectors.row_iter())
    }

    /// Release the ids and vectors, for example to adopt the matrix without copying.
    pub fn into_parts(self) -> (Box<[L]>, Matrix<f32>) {
        (self.ids, self.vectors)
    }
}

///////////////
// Bootstrap //
///////////////

/// The initial centroids of an empty index. Installs no points.
#[derive(Debug, PartialEq)]
pub struct Bootstrap<L> {
    centroids: CentroidBlock<L>,
}

impl<L> Bootstrap<L> {
    /// Install `centroids` as the first live lists.
    pub(super) fn new(centroids: CentroidBlock<L>) -> Self {
        Self { centroids }
    }

    /// The initial centroids.
    pub fn centroids(&self) -> &CentroidBlock<L> {
        &self.centroids
    }

    /// Release the initial centroids.
    pub fn into_centroids(self) -> CentroidBlock<L> {
        self.centroids
    }
}

/////////////
// Appends //
/////////////

/// Newly staged points placed into lists, grouped by destination list.
///
/// Every point staged for the operation appears exactly once.
#[derive(Debug, PartialEq, Eq)]
pub struct Appends<Id, L> {
    by_list: Grouped<L, Id>,
}

impl<Id: VectorId, L: VectorId> Appends<Id, L> {
    /// Place `ids[i]` into `lists[i]`.
    pub(super) fn new(ids: &[Id], lists: &[L]) -> Self {
        debug_assert_eq!(ids.len(), lists.len());
        let pairs = lists.iter().copied().zip(ids.iter().copied()).collect();
        Self {
            by_list: Grouped::from_pairs(pairs),
        }
    }

    /// Number of appended points.
    pub fn len(&self) -> usize {
        self.by_list.len()
    }

    /// Whether no points are appended.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Destination lists, ascending.
    pub fn lists(&self) -> &[L] {
        self.by_list.keys()
    }

    /// Every appended id, grouped by destination list.
    pub fn ids(&self) -> &[Id] {
        self.by_list.values()
    }

    /// Iterate `(list, ids)` groups in ascending list order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (L, &[Id])> {
        self.by_list.iter()
    }
}

//////////////
// Transfer //
//////////////

/// The source and destination of reassigned points. The two never match.
///
/// Ordered by destination, then source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Transfer<L> {
    source: L,
    destination: L,
}

impl<L: Copy> Transfer<L> {
    /// The list the points leave.
    pub fn source(&self) -> L {
        self.source
    }

    /// The list the points join.
    pub fn destination(&self) -> L {
        self.destination
    }
}

impl<L: Ord> Ord for Transfer<L> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.destination
            .cmp(&other.destination)
            .then_with(|| self.source.cmp(&other.source))
    }
}

impl<L: Ord> PartialOrd for Transfer<L> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

////////////
// Splits //
////////////

/// Parents retired by binary splits, their two children each, and where every
/// existing member of each parent goes.
///
/// Parents are ascending and unique. Rows `2 * i` and `2 * i + 1` of
/// [`Splits::children`] replace parent `i`; children are never parents and may end up
/// empty. Every member a parent had when the split was planned is evacuated exactly
/// once, never into a parent. Newly staged points routed to a parent are placed by
/// [`Appends`] instead.
#[derive(Debug, PartialEq)]
pub struct Splits<Id, L> {
    parents: Box<[L]>,
    children: CentroidBlock<L>,
    /// Keyed by `(parent, destination)`.
    evacuations: Grouped<(L, L), Id>,
}

impl<Id: VectorId, L: VectorId> Splits<Id, L> {
    /// Retire `parents`, replacing parent `i` with rows `2i` and `2i + 1` of `children`.
    ///
    /// Each item of `evacuations` is `(parent, members, lists)`: `members[j]` moves to
    /// `lists[j]`.
    pub(super) fn new<'a>(
        parents: Vec<L>,
        children: CentroidBlock<L>,
        evacuations: impl IntoIterator<Item = (L, &'a [Id], &'a [L])>,
    ) -> Self {
        debug_assert_eq!(children.len(), 2 * parents.len());
        let pairs = evacuations
            .into_iter()
            .flat_map(|(parent, members, lists)| {
                debug_assert_eq!(members.len(), lists.len());
                lists
                    .iter()
                    .zip(members)
                    .map(move |(&list, &id)| ((parent, list), id))
            })
            .collect();
        Self {
            parents: parents.into(),
            children,
            evacuations: Grouped::from_pairs(pairs),
        }
    }

    /// Number of splits.
    #[expect(clippy::len_without_is_empty, reason = "there is always a split")]
    pub fn len(&self) -> usize {
        self.parents.len()
    }

    /// Retired parent lists, ascending.
    pub fn parents(&self) -> &[L] {
        &self.parents
    }

    /// Every child centroid; rows `2 * i` and `2 * i + 1` replace `self.parents()[i]`.
    pub fn children(&self) -> &CentroidBlock<L> {
        &self.children
    }

    /// Number of existing points evacuated from all parents.
    pub fn num_evacuated(&self) -> usize {
        self.evacuations.len()
    }

    /// Iterate the individual splits in ascending parent order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Split<'_, Id, L>> {
        (0..self.parents.len()).map(|index| Split {
            splits: self,
            index,
        })
    }
}

/// One parent of [`Splits`] and the two children replacing it.
#[derive(Debug, Clone, Copy)]
pub struct Split<'a, Id, L> {
    splits: &'a Splits<Id, L>,
    index: usize,
}

impl<'a, Id: VectorId, L: VectorId> Split<'a, Id, L> {
    /// The retired parent list.
    pub fn parent(self) -> L {
        self.splits.parents[self.index]
    }

    /// The two new child lists.
    pub fn children(self) -> [L; 2] {
        let ids = self.splits.children.ids();
        [ids[2 * self.index], ids[2 * self.index + 1]]
    }

    /// The two child centroids, in the order of [`Self::children`].
    pub fn centroids(self) -> [&'a [f32]; 2] {
        let vectors = &self.splits.children.vectors;
        [vectors.row(2 * self.index), vectors.row(2 * self.index + 1)]
    }

    /// Where the parent's existing members go, as `(destination, ids)` groups in
    /// ascending destination order.
    pub fn evacuations(self) -> impl ExactSizeIterator<Item = (L, &'a [Id])> {
        let evacuations = &self.splits.evacuations;
        let parent = self.parent();
        let start = evacuations
            .keys()
            .partition_point(|&(from, _)| from < parent);
        let end = evacuations
            .keys()
            .partition_point(|&(from, _)| from <= parent);
        evacuations
            .groups(start..end)
            .map(|((_, destination), ids)| (destination, ids))
    }
}

///////////////////
// Reassignments //
///////////////////

/// Existing points moved between lists that survive the update.
///
/// Grouped by [`Transfer`] in ascending destination-then-source order; ids ascend
/// within each transfer.
#[derive(Debug, PartialEq, Eq)]
pub struct Reassignments<Id, L> {
    by_transfer: Grouped<Transfer<L>, Id>,
}

impl<Id: VectorId, L: VectorId> Reassignments<Id, L> {
    /// Each item of `moves` is `(source, members, lists)`: `members[j]` moves to
    /// `lists[j]`, or stays when that is `source`.
    pub(super) fn new<'a>(moves: impl IntoIterator<Item = (L, &'a [Id], &'a [L])>) -> Self {
        let pairs = moves
            .into_iter()
            .flat_map(|(source, members, lists)| {
                debug_assert_eq!(members.len(), lists.len());
                lists
                    .iter()
                    .zip(members)
                    .filter(move |&(&destination, _)| destination != source)
                    .map(move |(&destination, &id)| {
                        (
                            Transfer {
                                source,
                                destination,
                            },
                            id,
                        )
                    })
            })
            .collect();
        Self {
            by_transfer: Grouped::from_pairs(pairs),
        }
    }

    /// Number of reassigned points.
    pub fn len(&self) -> usize {
        self.by_transfer.len()
    }

    /// Whether no points are reassigned.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate `(transfer, ids)` groups in ascending destination-then-source order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Transfer<L>, &[Id])> {
        self.by_transfer.iter()
    }
}

/////////////////
// SplitInsert //
/////////////////

/// An insert batch that splits lists: the new points, the splits they trigger, and
/// the reassignment of existing points around the splits.
///
/// Every point appears in exactly one part. Nothing is appended or reassigned into a
/// split parent, and no reassignment leaves a parent (its evacuation empties it) or a
/// new child.
#[derive(Debug, PartialEq)]
pub struct SplitInsert<Id, L> {
    appends: Appends<Id, L>,
    splits: Splits<Id, L>,
    reassignments: Reassignments<Id, L>,
}

impl<Id, L> SplitInsert<Id, L> {
    /// Combine the parts of one split insert.
    pub(super) fn new(
        appends: Appends<Id, L>,
        splits: Splits<Id, L>,
        reassignments: Reassignments<Id, L>,
    ) -> Self {
        Self {
            appends,
            splits,
            reassignments,
        }
    }

    /// Newly staged points and their final lists.
    pub fn appends(&self) -> &Appends<Id, L> {
        &self.appends
    }

    /// The splits and their parents' evacuations.
    pub fn splits(&self) -> &Splits<Id, L> {
        &self.splits
    }

    /// Existing points moved between surviving lists.
    pub fn reassignments(&self) -> &Reassignments<Id, L> {
        &self.reassignments
    }

    /// Release the parts.
    pub fn into_parts(self) -> (Appends<Id, L>, Splits<Id, L>, Reassignments<Id, L>) {
        (self.appends, self.splits, self.reassignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block whose values are `0, 1, 2, ...` in row-major order.
    fn block(ids: &[u32], dim: usize) -> CentroidBlock<u32> {
        let data = (0..ids.len() * dim).map(|x| x as f32).collect();
        CentroidBlock::new(
            ids.to_vec(),
            Matrix::try_from(data, ids.len(), dim).unwrap(),
        )
    }

    fn owned<'a, K, V: Copy + 'a>(groups: impl Iterator<Item = (K, &'a [V])>) -> Vec<(K, Vec<V>)> {
        groups.map(|(key, values)| (key, values.to_vec())).collect()
    }

    #[test]
    fn centroid_block_pairs_ids_with_rows() {
        let bootstrap = Bootstrap::new(block(&[4, 2], 3));
        let block = bootstrap.centroids();
        assert_eq!((block.len(), block.dim()), (2, 3));
        assert_eq!(block.ids(), &[4, 2]);
        assert_eq!(block.vectors().row(1), &[3.0, 4.0, 5.0]);
        assert_eq!(
            owned(block.iter()),
            vec![(4, vec![0.0, 1.0, 2.0]), (2, vec![3.0, 4.0, 5.0])]
        );

        let (ids, vectors) = bootstrap.into_centroids().into_parts();
        assert_eq!(&*ids, &[4, 2]);
        assert_eq!(vectors.nrows(), 2);
    }

    #[test]
    fn appends_group_ids_by_list() {
        let appends = Appends::<u32, u32>::new(&[5, 1, 3, 2], &[7, 9, 7, 9]);
        assert_eq!(appends.len(), 4);
        assert_eq!(appends.lists(), &[7, 9]);
        assert_eq!(appends.ids(), &[3, 5, 1, 2]);
        assert_eq!(
            owned(appends.iter()),
            vec![(7, vec![3, 5]), (9, vec![1, 2])]
        );
        assert!(Appends::<u32, u32>::new(&[], &[]).is_empty());
    }

    #[test]
    fn splits_expose_each_split() {
        let splits = Splits::<u32, u32>::new(
            vec![10, 20],
            block(&[30, 31, 40, 41], 2),
            [(10, &[3, 1, 2][..], &[31, 30, 40][..]), (20, &[], &[])],
        );
        assert_eq!(splits.len(), 2);
        assert_eq!(splits.parents(), &[10, 20]);
        assert_eq!(splits.children().ids(), &[30, 31, 40, 41]);
        assert_eq!(splits.num_evacuated(), 3);

        let split: Vec<_> = splits.iter().collect();
        assert_eq!(split[0].parent(), 10);
        assert_eq!(split[0].children(), [30, 31]);
        assert_eq!(split[0].centroids(), [&[0.0f32, 1.0][..], &[2.0, 3.0][..]]);
        assert_eq!(
            owned(split[0].evacuations()),
            vec![(30, vec![1]), (31, vec![3]), (40, vec![2])]
        );
        assert_eq!(split[1].children(), [40, 41]);
        assert_eq!(split[1].evacuations().len(), 0);
    }

    #[test]
    fn reassignments_skip_members_that_stay() {
        let reassignments = Reassignments::<u32, u32>::new([
            (7, &[1, 2, 3][..], &[7, 9, 8][..]),
            (8, &[4, 5][..], &[9, 8][..]),
        ]);
        assert_eq!(reassignments.len(), 3);
        let transfers: Vec<_> = reassignments
            .iter()
            .map(|(t, ids)| (t.source(), t.destination(), ids.to_vec()))
            .collect();
        assert_eq!(
            transfers,
            vec![(7, 8, vec![3]), (7, 9, vec![2]), (8, 9, vec![4])]
        );
        assert!(Reassignments::<u32, u32>::new([(7, &[1][..], &[7][..])]).is_empty());
    }

    #[test]
    fn split_insert_releases_its_parts() {
        let update = SplitInsert::new(
            Appends::<u32, u32>::new(&[100], &[30]),
            Splits::new(vec![10], block(&[30, 31], 1), [(10, &[1][..], &[31][..])]),
            Reassignments::new([(7, &[2][..], &[30][..])]),
        );
        assert_eq!(update.appends().len(), 1);
        assert_eq!(update.splits().num_evacuated(), 1);
        assert_eq!(update.reassignments().len(), 1);
        let (appends, splits, reassignments) = update.into_parts();
        assert_eq!(
            (appends.lists(), splits.parents(), reassignments.len()),
            (&[30][..], &[10][..], 1)
        );
    }
}
