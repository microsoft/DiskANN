/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::common::{USizeConvertTo, iota_slice};
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
    T: Default + std::marker::Copy + std::cmp::PartialEq + std::fmt::Debug,
    usize: USizeConvertTo<T>,
    Const<N>: ArrayType<T, Type = [T; N]>,
    bitmask::BitMask<N, V::Arch>: SIMDMask<Arch = V::Arch>,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>>,
{
    // Test loads for all alignments.
    // Our strategy to to create an array of twice the underlying vector width and perform a
    // full-width load on each offset.
    let mut input = vec![T::default(); 2 * N];
    iota_slice(input.as_mut_slice());

    for i in 0..N {
        // SAFETY: By construction `input` is built to hold `2 * N` elements.
        // The maximum offset we apply is `N`, so all reads are valid.
        //
        // `load_simd` does not have alignment requirements stricter than `N`.
        let v = unsafe { V::load_simd(arch, input.as_ptr().add(i)) };
        let arr = v.to_array();
        // Ensure we read the correct window of the input array.
        for (j, value) in arr.iter().enumerate().take(N) {
            assert_eq!(*value, (i + j).test_convert());
        }
    }

    // Test `load_first`.
    // We will use a sliding window over an array so `miri` can check for out-of-bounds
    // reads.
    let mut input = [T::default(); N];
    iota_slice(input.as_mut_slice());

    // Set the loop bounds from 0 to one greater than the number of lanes.
    // Set up each load so that reading beyond the requested number of lanes will generate
    // an out-of-bounds read.
    for keep_first in 0..=N + 5 {
        let offset = N - keep_first.min(N);

        // SAFETY: `offset` less than or equal to `N`, the memory between `input.as_ptr()`
        // and `ptr` is valid and is contained within a single allocated object.
        let ptr = unsafe { input.as_ptr().add(offset) };

        // A helper lambda to provide uniform checking for the various ways we can invoke
        // predicated loads.
        let check = |arr: [T; N]| {
            for (i, value) in arr.iter().enumerate().take(N) {
                if i < keep_first {
                    assert_eq!(*value, (N - keep_first.min(N) + i).test_convert());
                } else {
                    assert_eq!(*value, 0.test_convert());
                }
            }
        };

        // Need miri to ensure the safety of this load.

        // Load using the `load_simd_first` API.
        // SAFETY: `ptr` points to valid memory and the contract of `V::load_simd_first`
        // must guarantee that nothing beyond the `keep_first` elements is accessed.
        let v = unsafe { V::load_simd_first(arch, ptr, keep_first) };
        let arr = v.to_array();
        println!("Array From First = {:?}", arr);
        check(arr);

        // Check by constructing a logical mask.
        // SAFETY: `ptr` points to valid memory and the contract of `V::load_simd_first`
        // must guarantee that nothing beyond the `keep_first` elements is accessed.
        let v = unsafe {
            V::load_simd_masked_logical(arch, ptr, V::Mask::keep_first(arch, keep_first))
        };
        let arr = v.to_array();
        println!("Array Logical Mask = {:?}", arr);
        check(arr);

        // Check by constructing a bit-mask mask.
        // SAFETY: `ptr` points to valid memory and the contract of `V::load_simd_first`
        // must guarantee that nothing beyond the `keep_first` elements is accessed.
        let v = unsafe {
            V::load_simd_masked(
                arch,
                ptr,
                <Const<N> as BitMaskType<V::Arch>>::Type::keep_first(arch, keep_first),
            )
        };
        let arr = v.to_array();
        println!("Array Logical Mask = {:?}", arr);
        check(arr);
    }
}
