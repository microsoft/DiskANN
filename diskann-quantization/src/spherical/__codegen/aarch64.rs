/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Explicitly instantiate the AArch64 Neon spherical inner-product paths.
use diskann_wide::arch::aarch64::Neon;

use crate::{
    alloc::{AllocatorError, GlobalAllocator},
    spherical::{
        iface::{AsData, AsQuery, DistanceComputer, Reify},
        vectors,
    },
};

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
