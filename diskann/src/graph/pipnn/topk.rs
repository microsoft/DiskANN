/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Top-k selection for the PiPNN ranking kernels.
//!
//! The nearest set of a point holds the k nearest candidates found so far, nearest
//! first. An unfilled slot holds [`Candidate::EMPTY`]. The k-th distance of a
//! nearest set is the distance in its last slot. It stays positive infinity until
//! the set is full. To offer a candidate to a nearest set is to insert the
//! candidate if it is nearer than the k-th distance. NaN and positive infinity are
//! never nearer, so they never enter a nearest set.
//!
//! [`select_top_k_ids`] serves partition assignment. It selects the nearest set of
//! each distance row and writes its IDs. [`select_top_k_symmetric`] serves leaf
//! construction. It runs a pair scan: it reads each unordered point pair once and
//! offers the pair to the nearest sets of both points. The caller sets k through
//! the width of the output. For common values of k, both functions use fixed-size
//! nearest sets, so the compiler can unroll insertion.
//!
//! Both functions compare [`LANES`] distances at a time with the k-th distance and
//! insert only the nearer ones. After a nearest set fills, most groups have no
//! nearer distance, so one SIMD comparison replaces `LANES` scalar comparisons.
//! The SIMD loops run inside `run2` or `run3` of architecture `A`, which compiles
//! them with the target features of `A`.

use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_wide::{SIMDPartialOrd, SIMDVector};

use super::simd::{LANES, Simd};

/// The ID of an output slot that holds no candidate.
pub(super) const UNASSIGNED: u32 = u32::MAX;

/// One slot of a nearest set: a candidate ID and its distance.
///
/// The ID is a position in the kernel input. Partition assignment uses leader
/// columns. Leaf construction uses point positions in the leaf.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Candidate {
    pub(super) local_idx: u32,
    pub(super) distance: f32,
}

impl Candidate {
    /// An unfilled slot. Every distance that can enter a nearest set is nearer.
    pub(super) const EMPTY: Self = Self::new(UNASSIGNED, f32::INFINITY);

    pub(super) const fn new(local_idx: u32, distance: f32) -> Self {
        Self {
            local_idx,
            distance,
        }
    }

    /// Return `true` if this slot holds a candidate.
    pub(super) const fn is_assigned(self) -> bool {
        self.local_idx != UNASSIGNED
    }
}

impl Default for Candidate {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Write the IDs of the k nearest columns of each distance row, nearest first.
///
/// k is the column count of `output`, and `output` must have one row per distance
/// row. Debug builds check this. Equal distances can select either column. If a
/// row has fewer than k distances that are not NaN or positive infinity, the
/// remaining slots hold [`UNASSIGNED`]. `scratch` is reusable storage for values
/// of k without a fixed-size nearest set.
pub(super) fn select_top_k_ids<A: Simd>(
    arch: A,
    distances: MatrixView<'_, f32>,
    output: MutMatrixView<'_, u32>,
    scratch: &mut Vec<Candidate>,
) {
    debug_assert_eq!(
        distances.nrows(),
        output.nrows(),
        "top-k IDs need one output row per distance row"
    );
    // Fixed-size nearest sets serve the common partition fanouts 1, 2, 3, 8, and
    // 10. On AVX2 with 1000-leader rows, a slice took about 3x the ranking time of
    // a fixed-size set at k = 8 and k = 10. That was 16-25% of `assign_leaders`
    // time at 128 dimensions. A branch-free insertion was slower than both.
    match output.ncols() {
        0 => {}
        1 => select_ids_with(arch, distances, output, &mut [Candidate::EMPTY; 1]),
        2 => select_ids_with(arch, distances, output, &mut [Candidate::EMPTY; 2]),
        3 => select_ids_with(arch, distances, output, &mut [Candidate::EMPTY; 3]),
        8 => select_ids_with(arch, distances, output, &mut [Candidate::EMPTY; 8]),
        10 => select_ids_with(arch, distances, output, &mut [Candidate::EMPTY; 10]),
        k => {
            scratch.resize(k, Candidate::EMPTY);
            select_ids_with(arch, distances, output, scratch.as_mut_slice());
        }
    }
}

/// Fill the nearest set of each point with its k nearest other points.
///
/// `distances` must be a symmetric matrix with one row and one column per point,
/// and `output` has one row per point. Debug builds check the shape. k is the
/// column count of `output`. Only the
/// strict lower triangle of `distances` is read: each pair is read once and
/// offered to both points. Candidate IDs are point positions. Equal distances can
/// select either point. Slots that no pair fills hold [`Candidate::EMPTY`].
/// `kth_distances` is reusable storage. It holds the k-th distance of each point
/// after the scan.
pub(super) fn select_top_k_symmetric<A: Simd>(
    arch: A,
    distances: MatrixView<'_, f32>,
    output: MutMatrixView<'_, Candidate>,
    kth_distances: &mut Vec<f32>,
) {
    let points = output.nrows();
    debug_assert!(
        distances.nrows() == points && distances.ncols() == points,
        "a symmetric scan needs one distance row and column per point"
    );
    let k = output.ncols();
    let slots = output.into_inner();
    // Fixed-size nearest sets serve the common `leaf_k` values 1, 2, and 3. On AVX2
    // at k = 3, slices took 1.1x to 2.5x the ranking time of fixed-size sets.
    match k {
        0 => {}
        1 => scan_pairs(arch, distances, slots.as_chunks_mut::<1>().0, kth_distances),
        2 => scan_pairs(arch, distances, slots.as_chunks_mut::<2>().0, kth_distances),
        3 => scan_pairs(arch, distances, slots.as_chunks_mut::<3>().0, kth_distances),
        k => {
            let mut neighborhoods: Vec<&mut [Candidate]> = slots.chunks_exact_mut(k).collect();
            scan_pairs(arch, distances, &mut neighborhoods, kth_distances);
        }
    }
}

/// Select the nearest set of each distance row in `nearest`, then copy its IDs to
/// the matching output row. All rows reuse the storage of `nearest`.
fn select_ids_with<A, Nearest>(
    arch: A,
    distances: MatrixView<'_, f32>,
    mut output: MutMatrixView<'_, u32>,
    nearest: &mut Nearest,
) where
    A: Simd,
    Nearest: AsMut<[Candidate]> + ?Sized,
{
    // `row_iter` panics on a matrix without columns. `row` returns an empty row,
    // so every slot of that row becomes `UNASSIGNED`.
    for (row, ids) in output.row_iter_mut().enumerate() {
        select_nearest(arch, nearest, distances.row(row));
        for (id, candidate) in ids.iter_mut().zip(nearest.as_mut().iter()) {
            *id = candidate.local_idx;
        }
    }
}

/// Replace the contents of a non-empty nearest set with the k nearest entries of
/// `distances`.
///
/// Each candidate ID is a position in `distances`.
#[inline]
fn select_nearest<A, Nearest>(arch: A, nearest: &mut Nearest, distances: &[f32])
where
    A: Simd,
    Nearest: AsMut<[Candidate]> + ?Sized,
{
    arch.run2(
        #[inline(always)]
        move |distances: &[f32], nearest: &mut Nearest| {
            let nearest = nearest.as_mut();
            nearest.fill(Candidate::EMPTY);
            let (groups, tail) = distances.as_chunks::<LANES>();
            let mut kth_distance = f32::INFINITY;
            for (group, group_distances) in groups.iter().enumerate() {
                let (first, values) = (group * LANES, load_group(arch, group_distances));
                kth_distance =
                    offer_group(arch, nearest, first, group_distances, values, kth_distance);
            }
            offer_each(nearest, groups.len() * LANES, tail, kth_distance);
        },
        distances,
        nearest,
    );
}

/// Run a pair scan over every point, and leave the k-th distance of each point in
/// `kth_distances`.
fn scan_pairs<A, Nearest>(
    arch: A,
    distances: MatrixView<'_, f32>,
    neighborhoods: &mut [Nearest],
    kth_distances: &mut Vec<f32>,
) where
    A: Simd,
    Nearest: AsMut<[Candidate]>,
{
    // The k-th distances live in their own array. One SIMD load then reads the
    // k-th distances of `LANES` targets, and one comparison checks `LANES` pairs.
    neighborhoods
        .iter_mut()
        .for_each(|nearest| nearest.as_mut().fill(Candidate::EMPTY));
    kth_distances.clear();
    kth_distances.resize(neighborhoods.len(), f32::INFINITY);
    // Point 0 has no earlier point to pair with.
    for source in 1..neighborhoods.len() {
        let row = distances.row(source);
        offer_pairs(arch, source, row, neighborhoods, kth_distances);
    }
}

/// Offer the pair of point `source` and each earlier point `target` to the nearest
/// sets of both points.
///
/// `row[target]` is the distance between `source` and `target`. Only the entries
/// before `source` are read. For each point `p`, `kth_distances[p]` must be the
/// k-th distance of `neighborhoods[p]`. This function keeps that true.
#[inline]
fn offer_pairs<A, Nearest>(
    arch: A,
    source: usize,
    row: &[f32],
    neighborhoods: &mut [Nearest],
    kth_distances: &mut [f32],
) where
    A: Simd,
    Nearest: AsMut<[Candidate]>,
{
    arch.run3(
        #[inline(always)]
        move |distances: &[f32], neighborhoods: &mut [Nearest], kth_distances: &mut [f32]| {
            let mut kth_distance = kth_distances[source];
            let (groups, tail) = distances.as_chunks::<LANES>();
            for (group, group_distances) in groups.iter().enumerate() {
                let first = group * LANES;
                // The source and its targets have different nearest sets, so both
                // directions use the same loaded vector.
                let values = load_group(arch, group_distances);
                let nearest = neighborhoods[source].as_mut();
                kth_distance =
                    offer_group(arch, nearest, first, group_distances, values, kth_distance);
                // Each lane compares with the k-th distance of its own target. The
                // source has no effect on this decision. Each lane also updates a
                // different target, so the mask stays correct while lanes insert.
                // `offer_group` must check again, because all its lanes share one set.
                let target_kth = load_group(arch, &kth_distances.as_chunks::<LANES>().0[group]);
                let mut eligible = A::active_lanes(values.lt_simd(target_kth));
                while eligible != 0 {
                    let lane = eligible.trailing_zeros() as usize;
                    eligible &= eligible - 1;
                    let candidate = Candidate::new(source as u32, group_distances[lane]);
                    let target = first + lane;
                    kth_distances[target] =
                        insert_sorted(neighborhoods[target].as_mut(), candidate);
                }
            }
            // The tail has fewer than `LANES` pairs. Offer each pair to both points
            // without SIMD.
            let first = groups.len() * LANES;
            for (offset, &distance) in tail.iter().enumerate() {
                let target = first + offset;
                if distance < kth_distance {
                    let candidate = Candidate::new(target as u32, distance);
                    kth_distance = insert_sorted(neighborhoods[source].as_mut(), candidate);
                }
                if distance < kth_distances[target] {
                    let candidate = Candidate::new(source as u32, distance);
                    kth_distances[target] =
                        insert_sorted(neighborhoods[target].as_mut(), candidate);
                }
            }
            kth_distances[source] = kth_distance;
        },
        &row[..source],
        neighborhoods,
        kth_distances,
    );
}

/// Offer the candidates of one group to a nearest set, and return the new k-th
/// distance.
///
/// `first_candidate` is the candidate ID of `distances[0]`, and `values` holds
/// the same distances as `distances`.
#[inline(always)]
fn offer_group<A: Simd>(
    arch: A,
    nearest: &mut [Candidate],
    first_candidate: usize,
    distances: &[f32; LANES],
    values: A::Vector,
    mut kth_distance: f32,
) -> f32 {
    // An unfilled nearest set accepts every distance that is not NaN or positive
    // infinity, so a scalar scan skips the mask. On AVX2, removing this path cost
    // 3% of leaf ranking time at k = 3 and 20% for 100-leader rows at k = 3.
    if kth_distance == f32::INFINITY {
        return offer_each(nearest, first_candidate, distances, kth_distance);
    }
    let mut eligible = A::active_lanes(values.lt_simd(A::Vector::splat(arch, kth_distance)));
    while eligible != 0 {
        let lane = eligible.trailing_zeros() as usize;
        eligible &= eligible - 1;
        // An earlier insertion in this group can lower the k-th distance.
        if distances[lane] < kth_distance {
            let candidate = Candidate::new((first_candidate + lane) as u32, distances[lane]);
            kth_distance = insert_sorted(nearest, candidate);
        }
    }
    kth_distance
}

/// Offer each entry of `distances` to a nearest set, and return the new k-th
/// distance.
///
/// `first_candidate` is the candidate ID of `distances[0]`.
#[inline(always)]
fn offer_each(
    nearest: &mut [Candidate],
    first_candidate: usize,
    distances: &[f32],
    mut kth_distance: f32,
) -> f32 {
    for (offset, &distance) in distances.iter().enumerate() {
        if distance < kth_distance {
            let candidate = Candidate::new((first_candidate + offset) as u32, distance);
            kth_distance = insert_sorted(nearest, candidate);
        }
    }
    kth_distance
}

/// Load one group of distances without copying it.
///
/// A load from a copy (`from_array(*group)`) can leave the copy on the stack in a
/// large loop. Each group then pays a store and a dependent reload: on AVX2 that
/// cost 11% of leaf ranking time.
#[inline(always)]
fn load_group<A: Simd>(arch: A, group: &[f32; LANES]) -> A::Vector {
    // A vector with more lanes than `LANES` would read past `group`. Both sides are
    // constants, so release builds remove this check.
    assert_eq!(A::Vector::LANES, group.len());
    // SAFETY: the assertion above proves that the load reads exactly the
    // `group.len()` readable values of `group`.
    unsafe { A::Vector::load_simd(arch, group.as_ptr()) }
}

/// Insert a candidate into a nearest set in nearest-first order, and return the
/// new k-th distance.
///
/// The set must not be empty, and the candidate must be nearer than the k-th
/// distance. A candidate goes after existing candidates with an equal distance.
#[inline(always)]
fn insert_sorted(nearest: &mut [Candidate], candidate: Candidate) -> f32 {
    let last = nearest.len() - 1;
    let mut slot = last;
    while slot > 0 && candidate.distance < nearest[slot - 1].distance {
        nearest[slot] = nearest[slot - 1];
        slot -= 1;
    }
    nearest[slot] = candidate;
    nearest[last].distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::test_support::{ArchCheck, for_each_arch};
    use diskann_wide::ARCH;
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
    use rstest::rstest;

    /// Split flat storage into nearest sets of `k` slots each.
    fn runtime_neighborhoods(storage: &mut [Candidate], k: usize) -> Vec<&mut [Candidate]> {
        storage.chunks_exact_mut(k).collect()
    }

    #[rstest]
    #[case::first_scan(0, 3)]
    #[case::more_points(2, 5)]
    #[case::fewer_points(5, 2)]
    #[case::same_points(3, 3)]
    #[case::no_points(3, 0)]
    fn starting_a_pair_scan_clears_neighborhoods_and_sizes_kth_distances(
        #[case] previous_points: usize,
        #[case] points: usize,
    ) {
        // Given: reusable storage contains assigned candidates and finite k-th distances.
        let mut output = vec![[Candidate::new(0, -12.0), Candidate::new(1, -6.0)]; points];
        let mut kth_distances = vec![-6.0; previous_points];
        // No pair can enter a nearest set, so every slot must come from the new scan.
        let distances = vec![f32::INFINITY; points * points];

        select_top_k_symmetric(
            ARCH,
            MatrixView::try_from(distances.as_slice(), points, points).unwrap(),
            MutMatrixView::try_from(output.as_flattened_mut(), points, 2).unwrap(),
            &mut kth_distances,
        );

        // Then: no old candidate or k-th distance belongs to the new scan.
        assert_eq!(output, vec![[Candidate::EMPTY; 2]; points]);
        assert_eq!(kth_distances, vec![f32::INFINITY; points]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "one distance row and column per point")]
    fn a_symmetric_scan_debug_checks_the_distance_matrix_size() {
        let distances = [0.0; 4];
        let mut output = [Candidate::EMPTY; 3];

        select_top_k_symmetric(
            ARCH,
            MatrixView::try_from(&distances[..], 2, 2).unwrap(),
            MutMatrixView::try_from(&mut output[..], 3, 1).unwrap(),
            &mut Vec::new(),
        );
    }

    #[rstest]
    #[case::empty_input(&[], [Candidate::EMPTY; 3])]
    #[case::only_nan_and_infinity(&[f32::INFINITY, f32::NAN], [Candidate::EMPTY; 3])]
    #[case::fewer_candidates_than_slots(
        &[11.0, -4.0],
        [Candidate::new(1, -4.0), Candidate::new(0, 11.0), Candidate::EMPTY],
    )]
    #[case::already_nearest_first(
        &[-7.0, 2.0, 8.0, 13.0],
        [Candidate::new(0, -7.0), Candidate::new(1, 2.0), Candidate::new(2, 8.0)],
    )]
    #[case::replacements_and_reordering(
        &[8.0, -7.0, 13.0, 2.0],
        [Candidate::new(1, -7.0), Candidate::new(3, 2.0), Candidate::new(0, 8.0)],
    )]
    #[case::nan_and_infinity_leave_gaps_in_ids(
        &[f32::NAN, 6.0, f32::INFINITY, -2.0, 1.0],
        [Candidate::new(3, -2.0), Candidate::new(4, 1.0), Candidate::new(1, 6.0)],
    )]
    fn selection_returns_nearest_input_positions_and_clears_unused_slots(
        #[case] distances: &[f32],
        #[case] expected: [Candidate; 3],
    ) {
        // Given: output still holds results from a different input.
        let mut output = [
            Candidate::new(2, -20.0),
            Candidate::new(4, -10.0),
            Candidate::new(9, -1.0),
        ];

        select_nearest(ARCH, &mut output, distances);

        assert_eq!(output, expected);
    }

    #[test]
    fn a_reused_nearest_set_does_not_keep_its_old_kth_distance() {
        let mut output = [Candidate::EMPTY; 2];
        select_nearest(ARCH, &mut output, &[-20.0, -10.0]);
        assert_eq!(output, [Candidate::new(0, -20.0), Candidate::new(1, -10.0)]);

        // Every new distance exceeds the previous selection's k-th distance.
        select_nearest(ARCH, &mut output, &[16.0, 3.0, 9.0]);

        assert_eq!(output, [Candidate::new(1, 3.0), Candidate::new(2, 9.0)]);
    }

    #[test]
    fn id_selection_matches_a_full_sort() {
        // Production selects the architecture at run time, so the grid runs on each one.
        struct SelectionGrid;

        impl ArchCheck for SelectionGrid {
            fn check<A: Simd>(&self, arch: A) {
                let lanes = A::Vector::LANES;
                let arch_name = std::any::type_name::<A>();
                // An empty distance row has no matrix form here. The `empty_input` case of
                // the selection test covers it.
                for count in [
                    1,
                    lanes - 1,
                    lanes,
                    lanes + 1,
                    2 * lanes,
                    2 * lanes + 3,
                    3 * lanes + 2,
                ] {
                    // Each pair offers a nearer distance before a farther one. Later pairs
                    // improve on earlier pairs, so a full nearest set must keep lowering
                    // its k-th distance inside a group.
                    let mut distances: Vec<_> = (0..count).map(|i| -(i as f32) - 1.0).collect();
                    for pair in distances.chunks_exact_mut(2) {
                        pair.swap(0, 1);
                    }
                    // k = 1, 2, 3, 8, and 10 use fixed-size nearest sets. The others use
                    // slices.
                    for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, lanes + 1] {
                        // Sorting the whole input is independent of the insertion and
                        // k-th distance logic.
                        let mut order: Vec<_> = (0..count).collect();
                        order.sort_by(|&left, &right| distances[left].total_cmp(&distances[right]));
                        let mut expected: Vec<_> = order
                            .into_iter()
                            .take(k)
                            .map(|index| index as u32)
                            .collect();
                        expected.resize(k, UNASSIGNED);
                        let mut output = vec![0; k];

                        select_top_k_ids(
                            arch,
                            MatrixView::try_from(distances.as_slice(), 1, count).unwrap(),
                            MutMatrixView::try_from(output.as_mut_slice(), 1, k).unwrap(),
                            &mut Vec::new(),
                        );

                        assert_eq!(output, expected, "{arch_name}, count={count}, k={k}");
                    }
                }
            }
        }

        for_each_arch(&SelectionGrid);
    }

    // The nightly Miri step selects this test by name. Rename it there too.
    #[test]
    fn selection_returns_distinct_candidates_when_distances_tie() {
        let best_index = 2 * LANES + 2;
        let mut distances = vec![8.0; best_index + 1];
        distances[best_index] = -2.0;
        let mut output = [Candidate::EMPTY; 4];

        select_nearest(ARCH, &mut output, &distances);

        assert_eq!(output[0], Candidate::new(best_index as u32, -2.0));
        // Any three distinct input positions with distance 8 are valid; tie order is free.
        let ties = &output[1..];
        assert!(
            ties.iter()
                .all(|c| c.local_idx < best_index as u32 && c.distance == 8.0),
            "invalid tied candidates: {ties:?}"
        );
        let mut ids: Vec<_> = ties.iter().map(|c| c.local_idx).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 3, "a candidate must not occupy multiple slots");
    }

    #[rstest]
    #[case::negative_zero(-0.0)]
    #[case::positive_zero(0.0)]
    #[case::lowest_finite(f32::MIN)]
    #[case::highest_finite(f32::MAX)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    fn a_selected_distance_keeps_its_position_and_bit_pattern(#[case] distance: f32) {
        // The SIMD comparison differs by architecture, so each one must keep the bits.
        struct FloatBits {
            distance: f32,
        }

        impl ArchCheck for FloatBits {
            fn check<A: Simd>(&self, arch: A) {
                let lanes = A::Vector::LANES;
                let arch_name = std::any::type_name::<A>();
                // The first and last lanes of a group, the first lane of the next group,
                // and the scalar tail.
                for index in [0, lanes - 1, lanes, 2 * lanes + 1] {
                    let mut distances = vec![f32::NAN; 2 * lanes + 2];
                    distances[index] = self.distance;
                    let mut output = [Candidate::EMPTY; 2];

                    select_nearest(arch, output.as_mut_slice(), &distances);

                    let context = format!("{arch_name}, index={index}");
                    assert_eq!(output[0].local_idx, index as u32, "{context}");
                    assert_eq!(
                        output[0].distance.to_bits(),
                        self.distance.to_bits(),
                        "{context}"
                    );
                    assert_eq!(output[1], Candidate::EMPTY, "{context}");
                }
            }
        }

        for_each_arch(&FloatBits { distance });
    }

    #[test]
    fn successive_pair_rows_accumulate_each_points_nearest_neighbors() {
        fn check<Nearest: AsMut<[Candidate]>>(neighborhoods: &mut [Nearest]) {
            let mut kth_distances = vec![f32::INFINITY; 4];
            // Given: pairs 0-1=9, 0-2=2, and 1-2=7; point 3 has no offered pairs.
            offer_pairs(ARCH, 1, &[9.0], neighborhoods, &mut kth_distances);
            offer_pairs(ARCH, 2, &[2.0, 7.0], neighborhoods, &mut kth_distances);
            let before_last_point = [
                [Candidate::new(2, 2.0), Candidate::new(1, 9.0)],
                [Candidate::new(2, 7.0), Candidate::new(0, 9.0)],
                [Candidate::new(0, 2.0), Candidate::new(1, 7.0)],
                [Candidate::EMPTY; 2],
            ];
            for (nearest, expected) in neighborhoods.iter_mut().zip(&before_last_point) {
                assert_eq!(nearest.as_mut(), expected);
            }
            assert_eq!(kth_distances, [9.0, 9.0, 7.0, f32::INFINITY]);

            // When: point 3 is at distances 6, 1, and 4 from points 0, 1, and 2.
            offer_pairs(ARCH, 3, &[6.0, 1.0, 4.0], neighborhoods, &mut kth_distances);

            // Then: each endpoint keeps its own nearest two, and its k-th distance is
            // the farther one.
            let expected = [
                [Candidate::new(2, 2.0), Candidate::new(3, 6.0)],
                [Candidate::new(3, 1.0), Candidate::new(2, 7.0)],
                [Candidate::new(0, 2.0), Candidate::new(3, 4.0)],
                [Candidate::new(1, 1.0), Candidate::new(2, 4.0)],
            ];
            for (nearest, expected) in neighborhoods.iter_mut().zip(&expected) {
                assert_eq!(nearest.as_mut(), expected);
            }
            assert_eq!(kth_distances, [6.0, 7.0, 4.0, 4.0]);
        }

        check(&mut [[Candidate::EMPTY; 2]; 4]);
        let mut storage = [Candidate::EMPTY; 8];
        check(&mut runtime_neighborhoods(&mut storage, 2));
    }

    #[test]
    fn pair_scans_match_a_full_sort_for_each_k_and_point_count() {
        // Production selects the architecture at run time, so the grid runs on each one.
        struct PairScanGrid;

        impl ArchCheck for PairScanGrid {
            fn check<A: Simd>(&self, arch: A) {
                let lanes = A::Vector::LANES;
                let arch_name = std::any::type_name::<A>();
                for point_count in [
                    lanes,
                    lanes + 1,
                    lanes + 2,
                    2 * lanes,
                    2 * lanes + 1,
                    2 * lanes + 2,
                ] {
                    // Given: each pair has a distinct integer distance in shuffled order, so
                    // rows see neighbors in an order unrelated to their IDs and no pairs tie.
                    // The final pair row has point_count - 1 distances, straddling vector
                    // boundaries.
                    let mut pair_distances: Vec<f32> = (0..point_count * (point_count - 1) / 2)
                        .map(|value| value as f32)
                        .collect();
                    pair_distances.shuffle(&mut StdRng::seed_from_u64(point_count as u64));
                    // Pair (high, low) with low < high has index high * (high - 1) / 2 + low.
                    let distances: Vec<f32> = (0..point_count * point_count)
                        .map(|index| {
                            let (point, other) = (index / point_count, index % point_count);
                            let (high, low) = (point.max(other), point.min(other));
                            if high == low {
                                0.0
                            } else {
                                pair_distances[high * (high - 1) / 2 + low]
                            }
                        })
                        .collect();
                    let distances =
                        MatrixView::try_from(distances.as_slice(), point_count, point_count)
                            .unwrap();
                    // k = 1, 2, and 3 use fixed-size nearest sets. The others use slices.
                    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17] {
                        let mut output = vec![Candidate::EMPTY; point_count * k];
                        let mut kth_distances = Vec::new();
                        let mut neighborhoods =
                            MutMatrixView::try_from(output.as_mut_slice(), point_count, k).unwrap();

                        // When: offer every non-self pair once, through the production
                        // width dispatch.
                        select_top_k_symmetric(
                            arch,
                            distances,
                            neighborhoods.as_mut_view(),
                            &mut kth_distances,
                        );

                        // Then: independently sort each complete row, excluding only the
                        // point itself.
                        for (point, &kth_distance) in kth_distances.iter().enumerate() {
                            let mut expected: Vec<_> = distances
                                .row(point)
                                .iter()
                                .enumerate()
                                .filter(|(other, _)| *other != point)
                                .map(|(other, &distance)| Candidate::new(other as u32, distance))
                                .collect();
                            expected
                                .sort_by(|left, right| left.distance.total_cmp(&right.distance));
                            expected.truncate(k);
                            expected.resize(k, Candidate::EMPTY);

                            assert_eq!(
                                neighborhoods.row(point),
                                expected,
                                "{arch_name}, point={point}, point_count={point_count}, k={k}"
                            );
                            assert_eq!(
                                kth_distance,
                                expected[k - 1].distance,
                                "{arch_name}, k-th distance of point={point}, point_count={point_count}, k={k}"
                            );
                        }
                    }
                }
            }
        }

        for_each_arch(&PairScanGrid);
    }

    #[test]
    fn the_target_accepts_a_pair_that_the_source_rejects() {
        let source = 2 * LANES + 2;
        // Targets sit inside a group, at both edges of a group, and in the scalar tail.
        for target in [1, LANES - 1, LANES, 2 * LANES + 1] {
            let mut storage = vec![Candidate::EMPTY; source + 1];
            let mut neighborhoods = runtime_neighborhoods(&mut storage, 1);
            let mut kth_distances = vec![f32::INFINITY; source + 1];
            let mut distances = vec![f32::INFINITY; source];
            distances[0] = -6.0;
            distances[target] = 4.0;

            // The source fills its only slot with point 0 before reaching this target.
            offer_pairs(
                ARCH,
                source,
                &distances,
                &mut neighborhoods,
                &mut kth_distances,
            );

            let mut expected = vec![Candidate::EMPTY; source + 1];
            expected[0] = Candidate::new(source as u32, -6.0);
            expected[target] = Candidate::new(source as u32, 4.0);
            expected[source] = Candidate::new(0, -6.0);
            let expected_kth_distances: Vec<_> = expected.iter().map(|c| c.distance).collect();
            drop(neighborhoods);
            assert_eq!(storage, expected, "target={target}");
            assert_eq!(kth_distances, expected_kth_distances, "target={target}");
        }
    }

    #[test]
    fn the_source_accepts_a_pair_that_the_target_rejects() {
        let source = 2 * LANES + 2;
        // Targets sit inside a group, at both edges of a group, and in the scalar tail.
        for target in [1, LANES - 1, LANES, 2 * LANES + 1] {
            let mut storage = vec![Candidate::EMPTY; source + 1];
            let mut neighborhoods = runtime_neighborhoods(&mut storage, 1);
            let mut kth_distances = vec![f32::INFINITY; source + 1];
            let mut previous_distances = vec![f32::INFINITY; target];
            previous_distances[0] = 1.0;
            offer_pairs(
                ARCH,
                target,
                &previous_distances,
                &mut neighborhoods,
                &mut kth_distances,
            );
            let mut distances = vec![f32::INFINITY; source];
            distances[target] = 5.0;

            // The target already has a closer neighbor; the source still needs one.
            offer_pairs(
                ARCH,
                source,
                &distances,
                &mut neighborhoods,
                &mut kth_distances,
            );

            let mut expected = vec![Candidate::EMPTY; source + 1];
            expected[0] = Candidate::new(target as u32, 1.0);
            expected[target] = Candidate::new(0, 1.0);
            expected[source] = Candidate::new(target as u32, 5.0);
            let expected_kth_distances: Vec<_> = expected.iter().map(|c| c.distance).collect();
            drop(neighborhoods);
            assert_eq!(storage, expected, "target={target}");
            assert_eq!(kth_distances, expected_kth_distances, "target={target}");
        }
    }

    // The nightly Miri step selects the `finite` case by its generated name,
    // `case_1_finite`. Keep this case first, or change the filter there too.
    #[rstest]
    #[case::finite(12.0)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    #[case::negative_zero(-0.0)]
    fn a_single_pair_updates_exactly_its_two_endpoints(#[case] distance: f32) {
        let source = 2 * LANES + 2;
        // Targets sit at both edges of a group and in the scalar tail.
        for target in [LANES - 1, LANES, 2 * LANES + 1] {
            let mut storage = vec![Candidate::EMPTY; (source + 1) * 2];
            let mut neighborhoods = runtime_neighborhoods(&mut storage, 2);
            let mut kth_distances = vec![f32::INFINITY; source + 1];
            let mut distances = vec![f32::INFINITY; source];
            distances[target] = distance;

            offer_pairs(
                ARCH,
                source,
                &distances,
                &mut neighborhoods,
                &mut kth_distances,
            );

            drop(neighborhoods);
            let mut expected = vec![Candidate::EMPTY; (source + 1) * 2];
            expected[source * 2] = Candidate::new(target as u32, distance);
            expected[target * 2] = Candidate::new(source as u32, distance);
            assert_eq!(storage, expected, "target={target}");
            assert_eq!(
                storage[source * 2].distance.to_bits(),
                distance.to_bits(),
                "target={target}"
            );
            assert_eq!(
                storage[target * 2].distance.to_bits(),
                distance.to_bits(),
                "target={target}"
            );
            // One neighbor leaves each two-slot nearest set open to another candidate.
            assert_eq!(
                kth_distances,
                vec![f32::INFINITY; source + 1],
                "target={target}"
            );
        }
    }

    #[rstest]
    #[case::no_pairs(&[])]
    #[case::nan_pairs(&[f32::NAN; 2 * LANES + 1])]
    #[case::infinite_pairs(&[f32::INFINITY; 2 * LANES + 1])]
    fn nan_infinite_or_missing_pairs_leave_nearest_sets_unchanged(#[case] distances: &[f32]) {
        let source = distances.len();
        let mut storage = vec![Candidate::EMPTY; 2 * LANES + 2];
        let mut neighborhoods = runtime_neighborhoods(&mut storage, 1);
        let mut kth_distances = vec![f32::INFINITY; 2 * LANES + 2];
        offer_pairs(ARCH, 1, &[3.0], &mut neighborhoods, &mut kth_distances);
        let previous_output: Vec<_> = neighborhoods
            .iter()
            .map(|nearest| nearest.to_vec())
            .collect();
        let previous_kth_distances = kth_distances.clone();

        // An empty row makes point 0 the source; it has no earlier points.
        offer_pairs(
            ARCH,
            source,
            distances,
            &mut neighborhoods,
            &mut kth_distances,
        );

        let output: Vec<_> = neighborhoods
            .iter()
            .map(|nearest| nearest.to_vec())
            .collect();
        assert_eq!(output, previous_output);
        assert_eq!(kth_distances, previous_kth_distances);
    }

    #[test]
    fn zero_width_outputs_select_nothing() {
        let distances = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut kth_distances = vec![-1.0; 2];

        select_top_k_symmetric(
            ARCH,
            MatrixView::try_from(&distances[..], 3, 3).unwrap(),
            MutMatrixView::try_from(&mut [][..], 3, 0).unwrap(),
            &mut kth_distances,
        );
        select_top_k_ids(
            ARCH,
            MatrixView::try_from(&distances[..], 3, 3).unwrap(),
            MutMatrixView::try_from(&mut [][..], 3, 0).unwrap(),
            &mut Vec::new(),
        );

        // A scan without slots leaves the reusable k-th distances untouched.
        assert_eq!(kth_distances, [-1.0; 2]);
    }

    #[test]
    fn distance_rows_without_columns_leave_every_slot_unassigned() {
        let mut output = [0; 4];

        select_top_k_ids(
            ARCH,
            MatrixView::try_from(&[][..], 2, 0).unwrap(),
            MutMatrixView::try_from(&mut output[..], 2, 2).unwrap(),
            &mut Vec::new(),
        );

        assert_eq!(output, [UNASSIGNED; 4]);
    }
}
