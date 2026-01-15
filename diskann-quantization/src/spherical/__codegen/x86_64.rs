/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The strategy here is to generate the concrete instantiations of [`DistanceComputer`]
//! that we care about.
//!
//! This will ensure that the relevant `evaluate` methods get compiled.

use diskann_wide::arch::x86_64::{V3, V4};

use crate::{
    alloc::{AllocatorError, GlobalAllocator},
    bits::{BitTranspose, Dense},
    spherical::{
        iface::{AsData, AsFull, AsQuery, DistanceComputer, Reify},
        vectors,
    },
};

///////////
// 1-bit //
///////////

//----//
// V3 //
//----//

#[inline(never)]
pub fn onebit_v3_l2_data_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<1>, AsData<1>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_ip_data_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<1>, AsData<1>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_cosine_data_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<1>, AsData<1>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_l2_query_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4, BitTranspose>, AsData<1>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_ip_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4, BitTranspose>, AsData<1>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_cosine_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4, BitTranspose>, AsData<1>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_l2_full_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<1>>::new(vectors::CompensatedSquaredL2::new(dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_ip_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<1>>::new(vectors::CompensatedIP::new(shift, dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn onebit_v3_cosine_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<1>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

///////////
// 2-bit //
///////////

//----//
// V4 //
//----//

#[inline(never)]
pub fn twobit_v4_l2_data_data(arch: V4, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<2>, AsData<2>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v4_ip_data_data(
    arch: V4,
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

#[inline(never)]
pub fn twobit_v4_cosine_data_data(
    arch: V4,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<2>, AsData<2>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v4_l2_query_data(arch: V4, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2, Dense>, AsData<2>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v4_ip_query_data(
    arch: V4,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2, Dense>, AsData<2>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v4_cosine_query_data(
    arch: V4,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2, Dense>, AsData<2>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

//----//
// V3 //
//----//

#[inline(never)]
pub fn twobit_v3_l2_data_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<2>, AsData<2>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_ip_data_data(
    arch: V3,
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

#[inline(never)]
pub fn twobit_v3_cosine_data_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<2>, AsData<2>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_l2_query_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2, Dense>, AsData<2>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_ip_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2, Dense>, AsData<2>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_cosine_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<2, Dense>, AsData<2>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_l2_full_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<2>>::new(vectors::CompensatedSquaredL2::new(dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_ip_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<2>>::new(vectors::CompensatedIP::new(shift, dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn twobit_v3_cosine_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<2>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

///////////
// 4-bit //
///////////

#[inline(never)]
pub fn fourbit_v3_l2_data_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<4>, AsData<4>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_ip_data_data(
    arch: V3,
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

#[inline(never)]
pub fn fourbit_v3_cosine_data_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<4>, AsData<4>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_l2_query_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4, Dense>, AsData<4>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_ip_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4, Dense>, AsData<4>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_cosine_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<4, Dense>, AsData<4>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_l2_full_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<4>>::new(vectors::CompensatedSquaredL2::new(dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_ip_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<4>>::new(vectors::CompensatedIP::new(shift, dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn fourbit_v3_cosine_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<4>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

///////////
// 8-bit //
///////////

#[inline(never)]
pub fn eightbit_v3_l2_data_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<8>, AsData<8>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_ip_data_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<8>, AsData<8>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_cosine_data_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsData<8>, AsData<8>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_l2_query_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<8, Dense>, AsData<8>>::new(
        vectors::CompensatedSquaredL2::new(dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_ip_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<8, Dense>, AsData<8>>::new(
        vectors::CompensatedIP::new(shift, dim),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_cosine_query_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsQuery<8, Dense>, AsData<8>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_l2_full_data(arch: V3, dim: usize) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<8>>::new(vectors::CompensatedSquaredL2::new(dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_ip_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify =
        Reify::<_, _, AsFull, AsData<8>>::new(vectors::CompensatedIP::new(shift, dim), dim, arch);
    DistanceComputer::new(reify, GlobalAllocator)
}

#[inline(never)]
pub fn eightbit_v3_cosine_full_data(
    arch: V3,
    shift: &[f32],
    dim: usize,
) -> Result<DistanceComputer, AllocatorError> {
    let reify = Reify::<_, _, AsFull, AsData<4>>::new(
        vectors::CompensatedCosine::new(vectors::CompensatedIP::new(shift, dim)),
        dim,
        arch,
    );
    DistanceComputer::new(reify, GlobalAllocator)
}
