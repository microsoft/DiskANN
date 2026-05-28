/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::common::{USizeConvertTo, bytes, iota_slice};
use crate::{
    bitmask,
    constant::Const,
    traits::{ArrayType, BitMaskType, SIMDMask, SIMDVector},
};

// Common utilities shared between unit tests.
// Since the pattern of tests is the same for many different implementations, we keep
// testing utilities together in one place to ensure uniformity of testing methodology.

fn check(output: &[u8], target: &[u8], offset: usize, message: &dyn std::fmt::Display) {
    let iszero = |x: &u8| *x == 0;

    assert!(
        output[..offset].iter().all(iszero),
        "prefix of {:?} up to {} is not zero -- {}",
        output,
        offset,
        message
    );

    assert_eq!(
        &output[offset..offset + target.len()],
        target,
        "output window from {} not equal to target -- {}",
        offset,
        message
    );

    assert!(
        output[offset + target.len()..].iter().all(iszero),
        "suffix of {:?} starting from {} is not zero -- {}",
        output,
        offset + target.len(),
        message
    );
}

struct FullStore(usize);
impl std::fmt::Display for FullStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "full SIMD store at byte offset {}", self.0)
    }
}

struct PredicatedStore {
    api: &'static str,
    keep_first: usize,
    sub: usize,
}
impl std::fmt::Display for PredicatedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`{}` with `keep_first` = {}, `sub` = {}",
            self.api, self.keep_first, self.sub
        )
    }
}

// Test the store operations in `SIMDVector`.
// Require that the input type is constructible from an unsigned, 64-bit integer to enable
// initialization and comparison of the loaded result.
pub(crate) fn test_store_simd<T, const N: usize, V>(arch: V::Arch)
where
    T: Default
        + std::marker::Copy
        + std::cmp::PartialEq
        + std::fmt::Debug
        + std::ops::AddAssign
        + bytemuck::Pod,
    usize: USizeConvertTo<T>,
    Const<N>: ArrayType<T, Type = [T; N]>,
    bitmask::BitMask<N, V::Arch>: SIMDMask<Arch = V::Arch>,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>>,
{
    let elsize = std::mem::size_of::<T>();

    // Test full-width stores at every byte offset to verify unaligned store correctness.
    // A `u8` buffer of `2 * N * elsize` bytes lets us place the write at every byte
    // position and confirm `store_simd` behaves like `ptr::write_unaligned`.
    let mut output = vec![0u8; elsize * 2 * N];

    let mut input = [T::default(); N];
    iota_slice(input.as_mut_slice());
    // Add 1 to each value so the first entry is non-zero.
    for i in input.iter_mut() {
        *i += 1.test_convert();
    }
    let v = V::from_array(arch, input);

    for i in 0..=N * elsize {
        output.fill(0);

        // SAFETY: `i + N * elsize <= 2 * N * elsize`, so the write is in bounds.
        unsafe { v.store_simd(output.as_mut_ptr().add(i).cast::<T>()) };

        check(&output, bytes(&input), i, &FullStore(i));
    }

    let mut output = vec![0u8; elsize * (N + 1)];
    let base = output.as_mut_ptr();

    // Test predicated stores at every byte offset.
    // The buffer is `(N + 1) * elsize` bytes so that writing beyond `kept` elements would
    // exceed the allocation, letting miri catch over-writes.
    for keep_first in 0..=N + 5 {
        let kept = keep_first.min(N);
        let expected = bytes(&input[..kept]);
        for sub in 0..(elsize + 1) {
            let offset = elsize * (N - kept + 1) - sub;

            // SAFETY: for all three stores below: `offset + kept * elsize <= (N + 1) * elsize`,
            // so `ptr` through `ptr + kept` elements is within the allocation.
            // Each API must not write beyond `keep_first` elements.
            let ptr = unsafe { base.add(offset).cast::<T>() };

            let label = |api| PredicatedStore {
                api,
                keep_first,
                sub,
            };

            output.fill(0);

            // SAFETY:
            // * `ptr` is valid for `kept` elements.
            // * Each API must not write beyond `keep_first` elements.
            unsafe { v.store_simd_first(ptr, keep_first) };
            check(&output, expected, offset, &label("store_simd_first"));

            output.fill(0);
            // SAFETY: Same as `store_simd_first`.
            unsafe { v.store_simd_masked_logical(ptr, V::Mask::keep_first(arch, keep_first)) };
            check(
                &output,
                expected,
                offset,
                &label("store_simd_masked_logical"),
            );

            output.fill(0);
            // SAFETY: Same as `store_simd_first`.
            unsafe {
                v.store_simd_masked(
                    ptr,
                    <Const<N> as BitMaskType<V::Arch>>::Type::keep_first(arch, keep_first),
                )
            };
            check(&output, expected, offset, &label("store_simd_masked"));
        }
    }
}
