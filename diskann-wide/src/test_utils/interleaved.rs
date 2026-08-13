/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::common::{USizeConvertTo, bytes};
use crate::{InterleavedLoadStore, SIMDVector, constant::Const, traits::ArrayType};

fn vectors<T, const LANES: usize, const N: usize>() -> [[T; LANES]; N]
where
    T: Copy + Default,
    usize: USizeConvertTo<T>,
{
    core::array::from_fn(|stream| {
        core::array::from_fn(|lane| (stream * LANES + lane).test_convert())
    })
}

fn interleave<T: Copy, const LANES: usize, const N: usize>(vectors: &[[T; LANES]; N]) -> Vec<T> {
    (0..LANES)
        .flat_map(|lane| (0..N).map(move |stream| vectors[stream][lane]))
        .collect()
}

pub(crate) fn test_deinterleaved_load<T, const LANES: usize, const N: usize, V>(arch: V::Arch)
where
    T: Copy + Default + PartialEq + bytemuck::Pod + std::fmt::Debug,
    [T; LANES]: bytemuck::Pod,
    usize: USizeConvertTo<T>,
    Const<LANES>: ArrayType<T, Type = [T; LANES]>,
    V: SIMDVector<Scalar = T, ConstLanes = Const<LANES>> + InterleavedLoadStore<N>,
{
    let expected = vectors::<T, LANES, N>();
    let input = interleave(&expected);
    let input_bytes = bytes(&input);
    let mut storage = vec![0_u8; input_bytes.len() * 2 + 1];

    for offset in 0..=input_bytes.len() {
        storage.fill(0);
        storage[offset..offset + input_bytes.len()].copy_from_slice(input_bytes);

        // SAFETY: The copied window contains exactly `N * LANES` readable values.
        let actual =
            unsafe { V::load_deinterleaved(arch, storage.as_ptr().add(offset).cast::<T>()) };
        assert_eq!(actual.map(SIMDVector::to_array), expected);
    }
}

pub(crate) fn test_interleaved_store<T, const LANES: usize, const N: usize, V>(arch: V::Arch)
where
    T: Copy + Default + PartialEq + bytemuck::Pod + std::fmt::Debug,
    [T; LANES]: bytemuck::Pod,
    usize: USizeConvertTo<T>,
    Const<LANES>: ArrayType<T, Type = [T; LANES]>,
    V: SIMDVector<Scalar = T, ConstLanes = Const<LANES>> + InterleavedLoadStore<N>,
{
    let source = vectors::<T, LANES, N>();
    let vectors = source.map(|vector| V::from_array(arch, vector));
    let expected = interleave(&source);
    let expected_bytes = bytes(&expected);
    let mut storage = vec![0_u8; expected_bytes.len() * 2 + 1];

    for offset in 0..=expected_bytes.len() {
        storage.fill(0);
        // SAFETY: The output window contains exactly `N * LANES` writable values.
        unsafe { V::store_interleaved(vectors, storage.as_mut_ptr().add(offset).cast::<T>()) };

        assert!(storage[..offset].iter().all(|&byte| byte == 0));
        assert_eq!(
            &storage[offset..offset + expected_bytes.len()],
            expected_bytes
        );
        assert!(
            storage[offset + expected_bytes.len()..]
                .iter()
                .all(|&byte| byte == 0)
        );
    }
}
