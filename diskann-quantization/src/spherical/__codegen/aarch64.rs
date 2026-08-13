/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Explicitly instantiate the AArch64 Neon spherical inner-product paths.
use diskann_wide::arch::aarch64::Neon;

use crate::{
    alloc::{AllocatorError, GlobalAllocator},
    spherical::{
        iface::{AsData, AsFull, AsQuery, DistanceComputer, Reify},
        vectors,
    },
};

/// Instantiate the Neon inner-product implementation for
/// `&[f32] × USlice<'_, 1>` in the full-precision-query-to-data path.
#[inline(never)]
pub fn onebit_neon_ip_full_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<1>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon inner-product implementation for
/// `&[f32] × USlice<'_, 2>` in the full-precision-query-to-data path.
#[inline(never)]
pub fn twobit_neon_ip_full_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<2>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon inner-product implementation for
/// `&[f32] × USlice<'_, 4>` in the full-precision-query-to-data path.
#[inline(never)]
pub fn fourbit_neon_ip_full_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<4>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon inner-product implementation for
/// `USlice<'_, 2> × USlice<'_, 2>` in the data-to-data path.
#[inline(never)]
pub fn twobit_neon_ip_data_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<2>, AsData<2>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon inner-product implementation for the two-bit
/// query-to-data path.
///
/// `dispatch_map!(2, AsQuery<2>, Neon);`
#[inline(never)]
pub fn twobit_neon_ip_query_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2>, AsData<2>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon inner-product implementation for
/// `USlice<'_, 4> × USlice<'_, 4>` in the data-to-data path.
#[inline(never)]
pub fn fourbit_neon_ip_data_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<4>, AsData<4>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon inner-product implementation for the four-bit
/// query-to-data path.
///
/// `dispatch_map!(4, AsQuery<4>, Neon);`
#[inline(never)]
pub fn fourbit_neon_ip_query_data(
    arch: Neon,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4>, AsData<4>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon SquaredL2 implementation for
/// `USlice<'_, 2> × USlice<'_, 2>` in the data-to-data path.
#[inline(never)]
pub fn twobit_neon_l2_data_data(
    arch: Neon,
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<2>, AsData<2>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon SquaredL2 implementation for the two-bit
/// query-to-data path.
///
/// `dispatch_map!(2, AsQuery<2>, Neon);`
#[inline(never)]
pub fn twobit_neon_l2_query_data(
    arch: Neon,
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2>, AsData<2>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon SquaredL2 implementation for
/// `USlice<'_, 4> × USlice<'_, 4>` in the data-to-data path.
#[inline(never)]
pub fn fourbit_neon_l2_data_data(
    arch: Neon,
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<4>, AsData<4>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}

/// Instantiate the Neon SquaredL2 implementation for the four-bit
/// query-to-data path.
///
/// `dispatch_map!(4, AsQuery<4>, Neon);`
#[inline(never)]
pub fn fourbit_neon_l2_query_data(
    arch: Neon,
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4>, AsData<4>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );

    DistanceComputer::new(reify, GlobalAllocator)
}
