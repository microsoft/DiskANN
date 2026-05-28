/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::common::{USizeConvertTo, bytes, bytes_mut, iota_slice};
use crate::{
    bitmask,
    constant::Const,
    traits::{ArrayType, BitMaskType, SIMDMask, SIMDVector},
};

// Common utilities shared between unit tests.
// Since the pattern of tests is the same for many different implementations, we keep
// testing utilities together in one place to ensure uniformity of testing methodology.

// Test the loading functions of `SIMDVector`.
// Require that the input type is constructible from an unsigned, 64-bit integer to enable
// initialization and comparison of the loaded result.
pub(crate) fn test_load_simd<T, const N: usize, V>(arch: V::Arch)
where
    T: Default + std::marker::Copy + std::cmp::PartialEq + std::fmt::Debug + bytemuck::Pod,
    usize: USizeConvertTo<T>,
    Const<N>: ArrayType<T, Type = [T; N]>,
    bitmask::BitMask<N, V::Arch>: SIMDMask<Arch = V::Arch>,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>>,
{
    let mut reference = [T::default(); N];
    iota_slice(&mut reference);

    // Test full-width loads at every byte offset to verify unaligned load correctness.
    // A `u8` buffer of `2 * N * elsize` bytes lets us slide the reference data to every
    // byte position and confirm `load_simd` behaves like `ptr::read_unaligned`.
    let elsize: usize = std::mem::size_of::<T>();
    let mut input = vec![0u8; elsize * 2 * N];

    for i in 0..=N * elsize {
        input.fill(0);
        input[i..i + elsize * N].copy_from_slice(bytes(&reference));

        // SAFETY: `i + N * elsize <= 2 * N * elsize`, so the read is in bounds.
        let v = unsafe { V::load_simd(arch, input.as_ptr().add(i).cast::<T>()) };
        let arr = v.to_array();
        assert_eq!(arr, reference);
    }

    // Test predicated loads at every byte offset.
    //
    // `reference` is `N + 1` elements so the window can slide by up to one full element
    // while still having room for a full-width masked load. The pointer is placed so that
    // reading beyond `kept` elements would exceed the allocation, letting miri catch
    // over-reads.
    let mut reference = vec![T::default(); N + 1];
    iota_slice(&mut reference);

    for keep_first in 0..=N + 5 {
        let kept = keep_first.min(N);

        for sub in 0..(elsize + 1) {
            let offset = elsize * (N - kept + 1) - sub;

            let mut expected = [T::default(); N];
            bytes_mut(&mut expected[..kept])
                .copy_from_slice(&bytes(&reference)[offset..offset + kept * elsize]);

            // SAFETY: `offset + kept * elsize <= (N + 1) * elsize`, so `ptr` through
            // `ptr + kept` elements is within the allocation.
            let ptr = unsafe { reference.as_ptr().byte_add(offset) };

            // SAFETY:
            // * `ptr` is valid for `kept` elements.
            // * Each API must not read beyond `keep_first` elements.
            let v = unsafe { V::load_simd_first(arch, ptr, keep_first) };
            assert_eq!(
                v.to_array(),
                expected,
                "Failed `load_simd_first` for keep_first = {}, sub = {}",
                keep_first,
                sub
            );

            // SAFETY: Same as `V::load_simd_first`.
            let v = unsafe {
                V::load_simd_masked_logical(arch, ptr, V::Mask::keep_first(arch, keep_first))
            };
            assert_eq!(
                v.to_array(),
                expected,
                "Failed `load_simd_masked_logical` for keep_first = {}, sub = {}",
                keep_first,
                sub
            );

            // SAFETY: Same as `V::load_simd_first`.
            let v = unsafe {
                V::load_simd_masked(
                    arch,
                    ptr,
                    <Const<N> as BitMaskType<V::Arch>>::Type::keep_first(arch, keep_first),
                )
            };
            assert_eq!(
                v.to_array(),
                expected,
                "Failed `load_simd_masked` for keep_first = {}, sub = {}",
                keep_first,
                sub
            );
        }
    }
}
