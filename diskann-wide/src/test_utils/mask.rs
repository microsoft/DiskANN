/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{Architecture, BitMask, Const, SIMDMask, SupportedLaneCount};

// The maximum length supported for BitMasks.
const MAXLEN: usize = 64;

/// Utilities for checking compliance an implementation of the `SIMDMask` trait.
pub(crate) fn test_keep_first<T, const N: usize, A, F>(arch: A, mut f: F)
where
    A: Architecture,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A> + From<T>,
    T: SIMDMask<Arch = A, BitMask = BitMask<N, A>>,
    F: FnMut(T, BitMask<N, A>),
{
    assert_eq!(T::LANES, N);
    assert!(MAXLEN >= T::LANES);

    // The number of lanes we're going to check.
    let lanes = T::LANES;
    for first in 0..=(lanes + 5) {
        let full_mask = T::keep_first(arch, first);
        let bit_mask = BitMask::<N, A>::keep_first(arch, first);

        // Check manually through the interface.
        for i in 0..=MAXLEN {
            // Check values through the `get` API.
            // Options implement equal if the underlying type implements equal.
            assert_eq!(full_mask.get(i), bit_mask.get(i), "i = {}", i);
            assert_eq!(full_mask.get_unchecked(i), bit_mask.get_unchecked(i));
        }

        // Allow the caller to inspect the results.
        f(full_mask, bit_mask);

        // Make sure the conversion from `full_mask` to `bit_mask` succeeds.
        let from_full: BitMask<N, A> = full_mask.into();
        assert_eq!(from_full, bit_mask);
    }
}

fn check_with_fn<T, const N: usize, A, F>(mask: T, bitmask: BitMask<N, A>, mut f: F)
where
    A: Architecture,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    T: SIMDMask<Arch = A, BitMask = BitMask<N, A>>,
    F: FnMut(usize) -> bool,
{
    assert_eq!(T::LANES, N);
    assert!(MAXLEN >= T::LANES);
    let lanes = T::LANES;

    for i in 0..=MAXLEN {
        assert_eq!(mask.get(i), bitmask.get(i));
        assert_eq!(mask.get_unchecked(i), bitmask.get_unchecked(i));
        if i < lanes {
            assert_eq!(mask.get(i).unwrap(), f(i));
        }
    }
}

pub(crate) fn test_from_fn<T, const N: usize, A, F>(arch: A, mut checker: F)
where
    A: Architecture,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    T: SIMDMask<Arch = A, BitMask = BitMask<N, A>>,
    F: FnMut(T, BitMask<N, A>),
{
    assert_eq!(T::LANES, N);
    assert!(MAXLEN >= T::LANES);
    let lanes = T::LANES;

    // ALL FALSE
    {
        let dut = |_: usize| false;
        let m = T::from_fn(arch, dut);
        let bm = BitMask::<N, _>::from_fn(arch, dut);
        assert_eq!(m.count(), 0);
        check_with_fn(m, bm, dut);
        checker(m, bm);
    }

    // ALL TRUE
    {
        let dut = |_: usize| true;
        let m = T::from_fn(arch, dut);
        let bm = BitMask::<N, _>::from_fn(arch, dut);

        assert_eq!(m.count(), lanes);
        check_with_fn(m, bm, dut);
        checker(m, bm);
    }

    // Patterns.
    // The logic here is to generate patterns like this:
    //
    // * Iteration 0
    //   0 1 0 1 0 1 ...
    //   1 0 1 0 1 0 ...
    //
    // * Iteration 1
    //   0 0 1 1 0 0 1 1 ...
    //   1 1 0 0 1 1 0 0 ...
    //
    // * Iteration 2
    //   0 0 0 0 1 1 1 1 ...
    //   1 1 1 1 0 0 0 0 ...
    //
    // etc.
    let mut step = 2;
    while 2 * step <= lanes {
        let f = |i: usize| (i % step) < (step / 2);
        {
            let m = T::from_fn(arch, f);
            let bm = BitMask::<N, _>::from_fn(arch, f);
            assert_eq!(m.count(), lanes / 2);
            check_with_fn(m, bm, f);
            checker(m, bm);
        }

        // Invert the logic of the previous function.
        let f = |i: usize| !f(i);
        {
            let m = T::from_fn(arch, f);
            let bm = BitMask::<N, _>::from_fn(arch, f);
            assert_eq!(m.count(), lanes / 2);
            check_with_fn(m, bm, f);
            checker(m, bm);
        }

        step *= 2;
    }
}

pub(crate) fn test_reductions<T, const N: usize, A, F>(arch: A, mut checker: F)
where
    A: Architecture,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    T: SIMDMask<Arch = A, BitMask = BitMask<N, A>>,
    F: FnMut(T, BitMask<N, A>),
{
    assert_eq!(T::LANES, N);
    assert!(MAXLEN >= T::LANES);
    let lanes = T::LANES;

    // No lanes set.
    {
        let f = |_: usize| false;
        let m = T::from_fn(arch, f);
        assert!(!m.any());
        assert!(!m.all());
        assert!(m.none());
        assert_eq!(m.count(), 0);

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }

    // All lanes set.
    {
        let f = |_: usize| true;
        let m = T::from_fn(arch, f);
        assert!(m.any());
        assert!(m.all());
        assert!(!m.none());
        assert_eq!(m.count(), lanes);

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }

    // Only one lane set across all lanes.
    for i in 0..lanes {
        let f = |j: usize| i == j;

        let m = T::from_fn(arch, f);
        assert!(m.any());
        assert!(!m.all());
        assert!(!m.none());
        assert_eq!(m.count(), 1);

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }
}

pub(crate) fn test_first<T, const N: usize, A, F>(arch: A, mut checker: F)
where
    A: Architecture,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    T: SIMDMask<Arch = A, BitMask = BitMask<N, A>>,
    F: FnMut(T, BitMask<N, A>),
{
    assert_eq!(T::LANES, N);
    assert!(MAXLEN >= T::LANES);
    let lanes = T::LANES;

    // All lanes unset.
    {
        let f = |_: usize| false;
        let m = T::from_fn(arch, f);
        assert_eq!(m.first(), None);

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }

    // All lanes set.
    {
        let f = |_: usize| true;
        let m = T::from_fn(arch, f);
        assert_eq!(m.first(), Some(0));

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }

    // Odd lanes set.
    {
        let f = |i: usize| i % 2 == 1;
        let m = T::from_fn(arch, f);
        assert_eq!(m.first(), Some(1));

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }

    // Second half lanes set.
    {
        let f = |i: usize| i >= lanes / 2;
        let m = T::from_fn(arch, f);
        assert_eq!(m.first(), Some(lanes / 2));

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }

    // Exhaustive single lane set.
    for i in 0..lanes {
        let f = |l: usize| l == i;
        let m = T::from_fn(arch, f);
        assert_eq!(m.first(), Some(i));

        let bm = BitMask::<N, _>::from_fn(arch, f);
        checker(m, bm);
    }
}
