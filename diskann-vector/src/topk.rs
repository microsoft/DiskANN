/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sorted-array TopK selection without matrix or distance-metric dependencies.
//!
//! Nearest sets are sorted by nondecreasing distance, with [`Candidate::EMPTY`]
//! in unused slots. Offers accept only distances strictly below the current
//! k-th distance: NaN and positive infinity are ignored, negative infinity is
//! accepted, and new candidates follow existing candidates at equal distances.
//! SIMD helpers must run under the vector architecture's target features.

use diskann_wide::{Architecture, BitMask, Const, SIMDMask, SIMDPartialOrd, SIMDVector};

/// The ID of an output slot that holds no candidate.
pub const UNASSIGNED: u32 = u32::MAX;

/// One slot of a nearest set: an input-local candidate ID and its distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub local_idx: u32,
    pub distance: f32,
}

impl Candidate {
    /// An unfilled slot. Every distance that can enter a nearest set is nearer.
    pub const EMPTY: Self = Self::new(UNASSIGNED, f32::INFINITY);

    pub const fn new(local_idx: u32, distance: f32) -> Self {
        Self {
            local_idx,
            distance,
        }
    }

    /// Return `true` if this slot holds a candidate.
    pub const fn is_assigned(self) -> bool {
        self.local_idx != UNASSIGNED
    }
}

impl Default for Candidate {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// The number of distances in a shared-threshold SIMD group.
///
/// 16 lanes fill one AVX-512 register. AVX2 uses two registers for a group,
/// and Neon uses four. Native lane-wise consumers need not use this width.
pub const LANES: usize = 16;

/// Insert a candidate into a nearest set and return the new k-th distance.
///
/// The set must be nonempty, sorted by nondecreasing distance, and the candidate
/// must be strictly nearer than its last slot. Its ID must not be [`UNASSIGNED`].
/// A candidate goes after existing candidates with an equal distance.
/// Arrays and slices both retain their concrete type at the call site.
#[inline(always)]
pub fn insert_sorted<Nearest>(nearest: &mut Nearest, candidate: Candidate) -> f32
where
    Nearest: AsMut<[Candidate]> + ?Sized,
{
    let nearest = nearest.as_mut();
    let last = nearest.len() - 1;
    let mut slot = last;
    while slot > 0 && candidate.distance < nearest[slot - 1].distance {
        nearest[slot] = nearest[slot - 1];
        slot -= 1;
    }
    nearest[slot] = candidate;
    nearest[last].distance
}

/// Offer each distance to one nearest set and return the new k-th distance.
///
/// The set must be nonempty and sorted. `kth_distance` must be its last distance.
/// `first_candidate` is the ID of `distances[0]`; all resulting IDs must be
/// representable as `u32` and less than [`UNASSIGNED`].
#[inline(always)]
pub fn offer_each(
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

/// Offer a group sharing one nearest set and return its new k-th distance.
///
/// Preconditions are those of [`offer_each`], and `values` must hold the same
/// distances as `distances`. Eligible lanes are visited in increasing order,
/// checking the updated threshold again after each insertion.
#[inline(always)]
pub fn offer_group<V>(
    arch: V::Arch,
    nearest: &mut [Candidate],
    first_candidate: usize,
    distances: &[f32; LANES],
    values: V,
    mut kth_distance: f32,
) -> f32
where
    V: SIMDVector<Scalar = f32, ConstLanes = Const<LANES>> + SIMDPartialOrd,
{
    // An unfilled nearest set accepts every distance that is not NaN or positive
    // infinity, so a scalar scan skips the mask. On AVX2, removing this path cost
    // 3% of leaf ranking time at k = 3 and 20% for 100-leader rows at k = 3.
    if kth_distance == f32::INFINITY {
        return offer_each(nearest, first_candidate, distances, kth_distance);
    }
    let mask: BitMask<LANES, V::Arch> = values.lt_simd(V::splat(arch, kth_distance)).into();
    let mut eligible = mask.to_underlying();
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

/// Load one group of distances without copying it.
///
/// Loading from a copy can leave a store and dependent reload in a large loop.
#[inline(always)]
pub fn load_group<A: Architecture>(arch: A, group: &[f32; LANES]) -> A::f32x16 {
    const { assert!(A::f32x16::LANES == LANES) };
    // SAFETY: the assertion above proves that the load reads exactly the
    // `group.len()` readable values of `group`.
    unsafe { A::f32x16::load_simd(arch, group.as_ptr()) }
}

/// Offer one candidate to each eligible row using already loaded distances.
///
/// Lane `i` updates `neighborhoods[i]` and `kth_distances[i]`. Each eligible row
/// must be nonempty and sorted, its threshold must equal its last distance, and
/// `values` and `thresholds` must match the corresponding slices. The mask stays
/// valid while inserting: unlike [`offer_group`], lanes update different rows.
/// `candidate_id` must not be [`UNASSIGNED`], and each set mask bit must index all
/// three slices. Lanes are visited in increasing order.
#[inline(always)]
pub fn offer_rows<V, Nearest>(
    distances: &[f32],
    values: V,
    thresholds: V,
    candidate_id: u32,
    neighborhoods: &mut [Nearest],
    kth_distances: &mut [f32],
) where
    V: SIMDVector<Scalar = f32, ConstLanes = Const<LANES>> + SIMDPartialOrd,
    Nearest: AsMut<[Candidate]>,
{
    let mask: BitMask<LANES, V::Arch> = values.lt_simd(thresholds).into();
    let mut eligible = mask.to_underlying();
    while eligible != 0 {
        let lane = eligible.trailing_zeros() as usize;
        eligible &= eligible - 1;
        let candidate = Candidate::new(candidate_id, distances[lane]);
        kth_distances[lane] = insert_sorted(neighborhoods[lane].as_mut(), candidate);
    }
}

/// Visit eligible lanes of an existing native group in increasing lane order.
///
/// A caller comparing different nearest rows computes the transformed distances,
/// per-row thresholds, and valid-row mask before calling this helper. Each
/// callback can insert into its own row and update that row's threshold.
/// Unlike [`offer_group`], this helper does not recheck a shared threshold.
///
/// Only nonempty masks materialize the vector into a fixed-size array. Bits
/// outside the group are ignored. Native groups of 4, 8, and 16 lanes retain
/// their original widths.
#[inline(always)]
pub fn for_each_eligible<V, const N: usize>(
    values: V,
    mut mask: u64,
    mut offer: impl FnMut(usize, f32),
) where
    V: SIMDVector<Scalar = f32, ConstLanes = Const<N>>,
{
    assert!(N > 0 && N <= u64::BITS as usize);
    mask &= u64::MAX >> (u64::BITS as usize - N);
    if mask == 0 {
        return;
    }
    let mut distances = [0.0; N];
    // SAFETY: `ConstLanes = Const<N>` guarantees the store writes exactly the
    // `N` elements of the local array.
    unsafe { values.store_simd(distances.as_mut_ptr()) };
    while mask != 0 {
        let lane = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        offer(lane, distances[lane]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_wide::arch::Scalar;

    // Stable sorting is independent of the insertion implementation and retains
    // input order for equal distances, including differently signed zeros.
    fn oracle(distances: &[f32], first: usize, k: usize) -> Vec<Candidate> {
        let mut expected: Vec<_> = distances
            .iter()
            .enumerate()
            .filter(|&(_, &distance)| distance < f32::INFINITY)
            .map(|(index, &distance)| Candidate::new((first + index) as u32, distance))
            .collect();
        expected.sort_by(|left, right| left.distance.partial_cmp(&right.distance).unwrap());
        expected.truncate(k);
        expected.resize(k, Candidate::EMPTY);
        expected
    }

    fn assert_same_bits(actual: &[Candidate], expected: &[Candidate]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.local_idx, expected.local_idx);
            assert_eq!(actual.distance.to_bits(), expected.distance.to_bits());
        }
    }

    fn check_shared_groups<A: Architecture>(arch: A) {
        arch.run(
            #[inline(always)]
            move || {
                for count in [0, 1, 15, 16, 17, 31, 32, 35, 65] {
                    let distances: Vec<_> = (0..count)
                        .map(|i| match i % 13 {
                            0 => f32::NAN,
                            1 => f32::INFINITY,
                            2 => f32::NEG_INFINITY,
                            3 => -0.0,
                            4 => 0.0,
                            _ => ((count - i) % 11) as f32 - 5.0,
                        })
                        .collect();
                    for k in [1, 2, 3, 4, 7, 8, 10, 16, 70] {
                        let mut nearest = vec![Candidate::EMPTY; k];
                        let mut kth = f32::INFINITY;
                        let (groups, tail) = distances.as_chunks::<LANES>();
                        for (group, values) in groups.iter().enumerate() {
                            kth = offer_group(
                                arch,
                                &mut nearest,
                                37 + group * LANES,
                                values,
                                load_group(arch, values),
                                kth,
                            );
                        }
                        kth = offer_each(&mut nearest, 37 + groups.len() * LANES, tail, kth);
                        assert_same_bits(&nearest, &oracle(&distances, 37, k));
                        assert_eq!(kth.to_bits(), nearest[k - 1].distance.to_bits());
                    }
                }

                // All initial-mask lanes qualify, but the first insertion lowers
                // the shared threshold. Later lanes must recheck it.
                let mut nearest = [Candidate::new(100, 100.0)];
                let distances = std::array::from_fn(|lane| lane as f32 + 1.0);
                let kth = offer_group(
                    arch,
                    &mut nearest,
                    200,
                    &distances,
                    load_group(arch, &distances),
                    100.0,
                );
                assert_eq!(nearest, [Candidate::new(200, 1.0)]);
                assert_eq!(kth, 1.0);
            },
        );
    }

    fn check_native_group<V, const N: usize>(arch: V::Arch)
    where
        V: SIMDVector<Scalar = f32, ConstLanes = Const<N>> + SIMDPartialOrd,
        u64: From<<<V::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        let distances: [f32; N] = std::array::from_fn(|lane| match lane % 4 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            _ => -0.0,
        });
        // SAFETY: the input has exactly the vector's `N` lanes.
        let values = unsafe { V::load_simd(arch, distances.as_ptr()) };
        let thresholds = V::splat(arch, f32::INFINITY);
        // The final query is padding even if its distance would qualify.
        let valid = (1u64 << (N - 1)) - 1;
        let mask = u64::from(values.lt_simd(thresholds).bitmask().to_underlying()) & valid;
        let mut rows = vec![[Candidate::EMPTY; 2]; N];
        let mut kth = vec![f32::INFINITY; N];
        let mut visited = Vec::new();
        for_each_eligible(values, mask, |lane, distance| {
            visited.push(lane);
            kth[lane] = insert_sorted(&mut rows[lane], Candidate::new(73, distance));
        });
        let expected_lanes: Vec<_> = (0..N - 1)
            .filter(|&lane| distances[lane] < f32::INFINITY)
            .collect();
        assert_eq!(visited, expected_lanes);
        for lane in 0..N {
            if expected_lanes.contains(&lane) {
                assert_same_bits(
                    &rows[lane],
                    &[Candidate::new(73, distances[lane]), Candidate::EMPTY],
                );
            } else {
                assert_eq!(rows[lane], [Candidate::EMPTY; 2]);
            }
        }
        assert_eq!(kth, vec![f32::INFINITY; N]);
        for_each_eligible(values, 0, |_, _| panic!("an empty mask must not visit"));
        for_each_eligible(values, 1u64 << N, |_, _| {
            panic!("out-of-group bits must not visit")
        });

        let distances: [f32; N] = std::array::from_fn(|lane| lane as f32);
        // SAFETY: the input has exactly the vector's `N` lanes.
        let values = unsafe { V::load_simd(arch, distances.as_ptr()) };
        let mut visited = Vec::new();
        for_each_eligible(values, u64::MAX, |lane, distance| {
            visited.push((lane, distance));
        });
        assert_eq!(
            visited,
            (0..N).map(|lane| (lane, lane as f32)).collect::<Vec<_>>()
        );
    }

    fn check_arch<A: Architecture>(arch: A)
    where
        u64: From<<<A::mask_f32x4 as SIMDMask>::BitMask as SIMDMask>::Underlying>
            + From<<<A::mask_f32x8 as SIMDMask>::BitMask as SIMDMask>::Underlying>
            + From<<<A::mask_f32x16 as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    {
        check_shared_groups(arch);
        arch.run(
            #[inline(always)]
            move || {
                check_native_group::<A::f32x4, 4>(arch);
                check_native_group::<A::f32x8, 8>(arch);
                check_native_group::<A::f32x16, 16>(arch);
            },
        );
    }

    #[test]
    fn topk_matches_sort_and_visits_native_lanes_on_available_architectures() {
        check_arch(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            use diskann_wide::arch::x86_64::{V3, V4};
            if let Some(arch) = V3::new_checked() {
                check_arch(arch);
            }
            if let Some(arch) = V4::new_checked_miri() {
                check_arch(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            use diskann_wide::arch::aarch64::Neon;
            if let Some(arch) = Neon::new_checked() {
                check_arch(arch);
            }
        }
    }

    #[test]
    fn topk_insert_retains_array_and_slice_support_and_stable_ties() {
        let mut nearest = [Candidate::EMPTY; 3];
        insert_sorted(&mut nearest, Candidate::new(5, -0.0));
        insert_sorted(nearest.as_mut_slice(), Candidate::new(6, 0.0));
        insert_sorted(&mut nearest, Candidate::new(7, -1.0));
        assert_same_bits(
            &nearest,
            &[
                Candidate::new(7, -1.0),
                Candidate::new(5, -0.0),
                Candidate::new(6, 0.0),
            ],
        );
    }

    #[test]
    fn topk_different_rows_keep_independent_thresholds() {
        let distances = [9.0, 3.0, -4.0, f32::NEG_INFINITY];
        let mut rows = [
            [Candidate::new(0, 2.0), Candidate::new(1, 8.0)],
            [Candidate::new(0, 1.0), Candidate::new(1, 8.0)],
            [Candidate::new(0, -1.0), Candidate::EMPTY],
            [Candidate::EMPTY; 2],
        ];
        let mut kth = [8.0, 8.0, f32::INFINITY, f32::INFINITY];
        let before = rows;
        let mut values = [f32::INFINITY; LANES];
        values[..4].copy_from_slice(&distances);
        let mut thresholds = [f32::INFINITY; LANES];
        thresholds[..4].copy_from_slice(&kth);
        offer_rows(
            &distances,
            <Scalar as Architecture>::f32x16::from_array(Scalar, values),
            <Scalar as Architecture>::f32x16::from_array(Scalar, thresholds),
            2,
            &mut rows,
            &mut kth,
        );
        assert_eq!(rows[0], before[0]);
        for lane in 1..4 {
            let distances = [
                before[lane][0].distance,
                before[lane][1].distance,
                distances[lane],
            ];
            assert_same_bits(&rows[lane], &oracle(&distances, 0, 2));
        }
        assert_eq!(kth, [8.0, 3.0, -1.0, f32::INFINITY]);
    }
}
