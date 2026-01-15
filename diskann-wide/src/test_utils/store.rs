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

// Test the store operations in `SIMDVector`.
// Require that the input type is constructible from an unsigned, 64-bit integer to enable
// initialization and comparison of the loaded result.
pub(crate) fn test_store_simd<T, const N: usize, V>(arch: V::Arch)
where
    T: Default + std::marker::Copy + std::cmp::PartialEq + std::fmt::Debug + std::ops::AddAssign,
    usize: USizeConvertTo<T>,
    Const<N>: ArrayType<T, Type = [T; N]>,
    bitmask::BitMask<N, V::Arch>: SIMDMask<Arch = V::Arch>,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>>,
{
    // Test stores for all alignments.
    // Our strategy to to create an array of twice the underlying vector width and perform a
    // full-width stores on each offset.
    let mut output = vec![T::default(); 2 * N];

    let mut input = [T::default(); N];
    iota_slice(input.as_mut_slice());
    // Add 1 to each value so the first entry is non-zero.
    for i in input.iter_mut() {
        *i += 1.test_convert();
    }
    let v = V::from_array(arch, input);

    for i in 0..N {
        output.fill(T::default());

        // SAFETY: By construction `input` is built to hold `2 * N` elements.
        // The maximum offset we apply is `N`, so all stores are valid.
        //
        // `store_simd` does not have alignment requirements stricter than `N`.
        unsafe { v.store_simd(output.as_mut_ptr().add(i)) };
        // Ensure we read the correct window of the input array.
        for (j, value) in output.iter().enumerate() {
            if j < i {
                assert_eq!(
                    *value,
                    T::default(),
                    "values before the write pointer should not be accessed"
                );
            } else if j < i + N {
                assert_eq!(
                    *value,
                    (j - i + 1).test_convert(),
                    "expected index {} to be set to {}",
                    j,
                    (j - i + 1)
                );
            } else {
                assert_eq!(
                    *value,
                    T::default(),
                    "values after the write section should not be accessed"
                );
            }
        }
    }

    // Set up each store so that writing beyond the requested number of lanes will generate
    // an out-of-bounds write.
    for keep_first in 0..=N + 5 {
        let offset = N - keep_first.min(N);

        // A helper lambda to provide uniform checking for the various ways we can invoke
        // predicated loads.
        let check = |arr: [T; N]| {
            for (i, value) in arr.iter().enumerate() {
                if i < offset {
                    assert_eq!(*value, 0.test_convert());
                } else {
                    assert_eq!(*value, (i - offset + 1).test_convert());
                }
            }
        };

        // Need miri to ensure the safety of these stores.

        let mut output = [T::default(); N];
        // SAFETY: `offset` is less than or equal to `N`, so the memory between
        // `output.as_mut_ptr()` and `ptr` is valid and is contained within a single
        // allocated object.
        let ptr = unsafe { output.as_mut_ptr().add(offset) };

        // Store using the `store_simd_first` API.
        // SAFETY: `ptr` points to valid memory and the contract of `V::store_simd_first`
        // must guarantee that nothing beyond the `keep_first` elements is accessed.
        unsafe { v.store_simd_first(ptr, keep_first) };
        check(output);

        let mut output = [T::default(); N];
        // SAFETY: `offset` is less than or equal to `N`, so the memory between
        // `output.as_mut_ptr()` and `ptr` is valid and is contained within a single
        // allocated object.
        let ptr = unsafe { output.as_mut_ptr().add(offset) };

        // Check by constructing a logical mask.
        // SAFETY: `ptr` points to valid memory and the contract of `V::store_simd_first`
        // must guarantee that nothing beyond the `keep_first` elements is accessed.
        unsafe { v.store_simd_masked_logical(ptr, V::Mask::keep_first(arch, keep_first)) };
        check(output);

        let mut output = [T::default(); N];
        // SAFETY: `offset` is less than or equal to `N`, so the memory between
        // `output.as_mut_ptr()` and `ptr` is valid and is contained within a single
        // allocated object.
        let ptr = unsafe { output.as_mut_ptr().add(offset) };

        // Check by constructing a bit-mask mask.
        // SAFETY: `ptr` points to valid memory and the contract of `V::store_simd_first`
        // must guarantee that nothing beyond the `keep_first` elements is accessed.
        unsafe {
            v.store_simd_masked(
                ptr,
                <Const<N> as BitMaskType<V::Arch>>::Type::keep_first(arch, keep_first),
            )
        };
        check(output);
    }
}
