/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Split a type into or join from two halves.
///
/// For example, even dimensional fixed size arrays of length `N` will be split so the first
/// `N / 2` elements are in the low half, and the last `N / 2` elements are in the high half.
pub trait SplitJoin {
    /// The type of the halved element.
    type Halved;

    /// Split `self` into two equal halves.
    fn split(self) -> LoHi<Self::Halved>;

    /// Create `self` by joining the two halves.
    fn join(halves: LoHi<Self::Halved>) -> Self;
}

/// Representation of the low and high halves associated with an implementation of
/// [`SplitJoin`].
#[derive(Debug, Clone, Copy)]
pub struct LoHi<T> {
    /// The first half of a split entity.
    pub lo: T,
    /// The second half of a split entity.
    pub hi: T,
}

impl<T> LoHi<T> {
    /// Construct a new `LoHi` from the low and high parts.
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }

    /// Join the `lo` and `hi` portions.
    pub fn join<U>(self) -> U
    where
        U: SplitJoin<Halved = T>,
    {
        U::join(self)
    }

    /// Return a new [`LoHi`] with the function `f` applied to the pairwise members of
    /// of `self` and `x`.
    ///
    /// If it does not panic, `f` will be invoked exactly twice, first on `lo`, then on `hi`.
    pub fn map_with<U, F, R>(self, x: LoHi<U>, mut f: F) -> LoHi<R>
    where
        F: FnMut(T, U) -> R,
    {
        let lo = f(self.lo, x.lo);
        let hi = f(self.hi, x.hi);
        LoHi { lo, hi }
    }

    /// Return a new [`LoHi`] with the function `f` applied to each member.
    ///
    /// If it does not panic, `f` will be invoked exactly twice, first on `lo`, then on `hi`.
    pub fn map<F, R>(self, mut f: F) -> LoHi<R>
    where
        F: FnMut(T) -> R,
    {
        let lo = f(self.lo);
        let hi = f(self.hi);
        LoHi { lo, hi }
    }
}

macro_rules! array_splitjoin {
    ($N:literal) => {
        impl<T: Copy> SplitJoin for [T; $N] {
            type Halved = [T; { $N / 2 }];

            #[inline(always)]
            fn split(self) -> LoHi<Self::Halved> {
                const BASE: usize = { $N / 2 };
                LoHi {
                    lo: core::array::from_fn(|i| self[i]),
                    hi: core::array::from_fn(|i| self[BASE + i]),
                }
            }

            #[inline(always)]
            fn join(lohi: LoHi<Self::Halved>) -> Self {
                const BASE: usize = { $N / 2 };
                core::array::from_fn(|i| {
                    if i < BASE {
                        lohi.lo[i]
                    } else {
                        lohi.hi[i - BASE]
                    }
                })
            }
        }
    };
}

array_splitjoin!(2);
array_splitjoin!(4);
array_splitjoin!(8);
array_splitjoin!(16);
array_splitjoin!(32);
array_splitjoin!(64);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::fmt::Display;

    use rand::{
        SeedableRng,
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
    };

    use super::*;

    fn test_split<T>(full: &[T], lo: &[T], hi: &[T], context: &dyn Display)
    where
        T: PartialEq + std::fmt::Debug,
    {
        let full_len = full.len();
        assert_eq!(
            full_len % 2,
            0,
            "full length must be even, instead got {} -- {}",
            full_len,
            context
        );
        let half_len = full_len / 2;
        assert_eq!(
            half_len,
            lo.len(),
            "unexpected \"lo\" length -- {}",
            context
        );
        assert_eq!(
            half_len,
            hi.len(),
            "unexpected \"hi\" length -- {}",
            context
        );

        for i in 0..half_len {
            assert_eq!(
                full[i], lo[i],
                "low check failed at index {} -- {}",
                i, context
            );
        }

        for i in 0..half_len {
            assert_eq!(
                full[i + half_len],
                hi[i],
                "high check failed at index {} -- {}",
                i,
                context
            );
        }
    }

    struct Lazy<'a, T> {
        base: &'a [T],
        lo: &'a [T],
        hi: &'a [T],
    }

    impl<T> std::fmt::Display for Lazy<'_, T>
    where
        T: std::fmt::Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "base = {:?}, lo = {:?}, hi = {:?}",
                self.base, self.lo, self.hi
            )
        }
    }

    macro_rules! test_splitjoin {
        ($fn:ident, $len:literal, $trials:literal, $seed:literal) => {
            #[test]
            fn $fn() {
                const NUM_TRIALS: usize = $trials;
                let mut rng = StdRng::seed_from_u64($seed);
                for _ in 0..NUM_TRIALS {
                    let base: [i8; $len] =
                        core::array::from_fn(|_| StandardUniform {}.sample(&mut rng));

                    let LoHi { lo, hi } = base.split();

                    let context = Lazy {
                        base: &base,
                        lo: &lo,
                        hi: &hi,
                    };

                    test_split(&base, &lo, &hi, &context);

                    let rejoined = <[i8; $len]>::join(LoHi::new(lo, hi));
                    assert_eq!(base, rejoined);
                }
            }
        };
    }

    test_splitjoin!(test_splitjoin_2, 2, 100, 0x5943d0578df47cdd);
    test_splitjoin!(test_splitjoin_4, 4, 100, 0xc735a1c37c9a8c2c);
    test_splitjoin!(test_splitjoin_8, 8, 100, 0x4dcf648800b9f9b6);
    test_splitjoin!(test_splitjoin_16, 16, 50, 0xf7386a0621134477);
    test_splitjoin!(test_splitjoin_32, 32, 50, 0xb3b0ded762020295);
    test_splitjoin!(test_splitjoin_64, 64, 25, 0x0fc17da7d8a9e1d0);
}
