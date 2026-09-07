/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! SIMD schema for PiPNN numerical kernels.

use diskann_wide::{Architecture, Const, SIMDFloat, SIMDMask, SIMDVector, SupportedLaneCount};

/// Default SIMD representation used by both PiPNN ranking kernels.
///
/// This alias is the single build-time width selection.
type DefaultVector<A> = <A as Architecture>::f32x16;

/// Operations required by PiPNN SIMD vectors.
pub(super) trait PiPNNSIMDVector: SIMDVector<Scalar = f32> + SIMDFloat {
    /// Return one bit for each selected lane.
    fn active_lanes(mask: Self::Mask) -> u64;
}

impl<F, const N: usize> PiPNNSIMDVector for F
where
    F: SIMDVector<Scalar = f32, ConstLanes = Const<N>> + SIMDFloat,
    Const<N>: SupportedLaneCount,
    u64: From<<<F::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
{
    #[inline(always)]
    fn active_lanes(mask: Self::Mask) -> u64 {
        u64::from(mask.bitmask().to_underlying())
    }
}

/// PiPNN SIMD representation for one architecture.
pub(super) trait PiPNNSIMDSchema: Architecture {
    /// SIMD vector used by both ranking kernels.
    type Vector: PiPNNSIMDVector<Arch = Self>;
}

impl<A> PiPNNSIMDSchema for A
where
    A: Architecture,
    DefaultVector<A>: PiPNNSIMDVector<Arch = A>,
{
    type Vector = DefaultVector<A>;
}

/// One group of distances and their column indexes in the supplied slice.
#[derive(Clone, Copy)]
pub(super) enum DistanceBlock<'a, F: PiPNNSIMDVector> {
    Simd {
        first_idx: usize,
        values: F,
        lanes: &'a [f32],
    },
    Scalar {
        idx: usize,
        distance: f32,
    },
}

/// Iterate over distance groups without a callback on the ranking hot path.
///
/// Scalar lanes borrow the input slice, so sharing a block across updates needs
/// no temporary array. Tail entries follow in column order. The caller keeps the
/// selected architecture's execution scope active.
#[inline(always)]
pub(super) fn distance_blocks<A: PiPNNSIMDSchema>(
    arch: A,
    distances: &[f32],
) -> impl Iterator<Item = DistanceBlock<'_, A::Vector>> {
    let simd_end = distances.len() - distances.len() % A::Vector::LANES;
    distances[..simd_end]
        .chunks_exact(A::Vector::LANES)
        .enumerate()
        .map(move |(group, lanes)| DistanceBlock::Simd {
            first_idx: group * A::Vector::LANES,
            // SAFETY: chunks_exact yields one complete SIMD group.
            values: unsafe { A::Vector::load_simd(arch, lanes.as_ptr()) },
            lanes,
        })
        .chain(
            distances[simd_end..]
                .iter()
                .enumerate()
                .map(move |(tail, &distance)| DistanceBlock::Scalar {
                    idx: simd_end + tail,
                    distance,
                }),
        )
}
