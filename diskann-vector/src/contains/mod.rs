/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

cfg_if::cfg_if! {
    if #[cfg(all(target_arch="x86_64", target_feature = "avx2"))] {
        mod x86_64;
        use x86_64::{contains_simd_u32, contains_simd_u64};
    } else {
        mod fallback;
        use fallback::{contains_simd_u32, contains_simd_u64};
    }
}

/// A SIMD-accelerated version of
/// [`std::slice::contains`](https://doc.rust-lang.org/std/primitive.slice.html#method.contains)
pub trait ContainsSimd: Sized {
    fn contains_simd(vector: &[Self], target: Self) -> bool;
}

impl ContainsSimd for u32 {
    fn contains_simd(vector: &[u32], target: u32) -> bool {
        contains_simd_u32(vector, target)
    }
}

impl ContainsSimd for u64 {
    fn contains_simd(vector: &[u64], target: u64) -> bool {
        contains_simd_u64(vector, target)
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform, Uniform},
        Rng, SeedableRng,
    };

    use super::*;

    /// Test `contains_simd` for dimension of slice from 0 to `max_dim`.
    ///
    /// This tests works by initializing a slice with the elements `[0, 1, ... dim - 1]`
    /// and then searching for each of `[0, 1, ... dim - 1]` (as well as a few higher
    /// values which are not expected to be in the slice.
    ///
    /// This ensures that we can match all possible locations in any length slice up to
    /// `max_dim`.
    fn test_contains<T>(max_dim: usize)
    where
        T: ContainsSimd + Copy + TryFrom<usize> + PartialEq,
        <T as TryFrom<usize>>::Error: std::fmt::Debug,
    {
        for dim in 0..max_dim {
            println!("working on dim {dim}");
            let v: Vec<T> = (0..dim).map(|i| i.try_into().unwrap()).collect();

            // All of these queries should return success.
            for query in 0..dim {
                assert!(
                    T::contains_simd(&v, query.try_into().unwrap()),
                    "expected query {} to be iota slice of dimension {}",
                    query,
                    dim
                );
            }

            // None of these should return success.
            for query in dim..dim + 10 {
                assert!(
                    !T::contains_simd(&v, query.try_into().unwrap()),
                    "expected query {} not to be iota slice of dimension {}",
                    query,
                    dim
                );
            }
        }
    }

    #[test]
    fn test_contains_u32() {
        test_contains::<u32>(128);
    }

    #[test]
    fn test_contains_u64() {
        test_contains::<u64>(128);
    }

    #[test]
    fn test_contains_simd_u32() {
        let vector = vec![5, 7, 6, 3, 2, 1, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        test_contains_simd::<u32>(vector, vec![9]);
    }

    #[test]
    fn test_contains_simd_u64() {
        let vector = vec![5, 7, 6, 3, 2, 1, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        test_contains_simd::<u64>(vector, vec![9]);
    }

    #[test]
    fn test_contains_simd_multiple_of_8_u32() {
        let vector = vec![5, 7, 6, 3, 2, 1, 4, 0];
        test_contains_simd::<u32>(vector, vec![9, 8]);
    }

    #[test]
    fn test_contains_simd_multiple_of_8_u64() {
        let vector = vec![5, 7, 6, 3, 2, 1, 4, 0];
        test_contains_simd::<u64>(vector, vec![9, 8]);
    }

    #[test]
    fn test_contains_simd_non_multiple_of_8_u32() {
        let vector = vec![5, 7, 6, 3, 2, 1, 4, 0, 11];
        test_contains_simd::<u32>(vector, vec![9, 8]);
    }

    #[test]
    fn test_contains_simd_non_multiple_of_8_u64() {
        let vector = vec![5, 7, 6, 3, 2, 1, 4, 0, 11];
        test_contains_simd::<u64>(vector, vec![9, 8]);
    }
    fn test_contains_simd<T>(vector: Vec<T>, not_present: Vec<T>)
    where
        T: ContainsSimd + Copy,
    {
        not_present.iter().for_each(|item| {
            assert!(!T::contains_simd(&vector, *item));
        });

        vector.iter().for_each(|item| {
            assert!(T::contains_simd(&vector, *item));
        });
    }

    const NUM_TRIALS: usize = 1000;

    // Fuzz testing.
    #[test]
    fn contains_works_when_item_is_present() {
        // The distribution used select the length of slice being tests.
        let dim_dist = Uniform::new(1, 1000).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..NUM_TRIALS {
            let v: Vec<_> = (0..dim_dist.sample(&mut rng))
                .map(|_| StandardUniform {}.sample(&mut rng))
                .collect();
            let index_of_item = rng.random_range(0..v.len());
            let item = v[index_of_item];

            assert!(u32::contains_simd(&v, item));
        }
    }

    #[test]
    fn contains_works_when_item_is_not_present() {
        // The distribution used select the length of slice being tests.
        let dim_dist = Uniform::new(1, 1000).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..NUM_TRIALS {
            let v: Vec<_> = (0..dim_dist.sample(&mut rng))
                .map(|_| StandardUniform {}.sample(&mut rng))
                .collect();

            let mut item = StandardUniform {}.sample(&mut rng);
            while v.contains(&item) {
                item = StandardUniform {}.sample(&mut rng);
            }

            assert!(!u32::contains_simd(&v, item));
        }
    }
}
