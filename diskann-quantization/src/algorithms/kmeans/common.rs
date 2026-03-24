/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{SIMDMulAdd, SIMDSumTree, SIMDVector};

/// Compute the squared L2 norm of the argument.
pub(crate) fn square_norm(x: &[f32]) -> f32 {
    let px: *const f32 = x.as_ptr();
    let len = x.len();

    diskann_wide::alias!(f32s = f32x8);

    let mut i = 0;
    let mut s = f32s::default(diskann_wide::ARCH);

    // The number of 32-bit blocks over the underlying slice.
    if i + 32 <= len {
        let mut s0 = f32s::default(diskann_wide::ARCH);
        let mut s1 = f32s::default(diskann_wide::ARCH);
        let mut s2 = f32s::default(diskann_wide::ARCH);
        let mut s3 = f32s::default(diskann_wide::ARCH);
        while i + 32 <= len {
            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i)) };
            s0 = vx.mul_add_simd(vx, s0);

            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i + 8)) };
            s1 = vx.mul_add_simd(vx, s1);

            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i + 16)) };
            s2 = vx.mul_add_simd(vx, s2);

            // SAFETY: The memory range `[i, i + 32)` is valid by the loop bounds.
            let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i + 24)) };
            s3 = vx.mul_add_simd(vx, s3);

            i += 32;
        }

        s = (s0 + s1) + (s2 + s3)
    }

    while i + 8 <= len {
        // SAFETY: The memory range `[i, i + 8)` is valid by the loop bounds.
        let vx = unsafe { f32s::load_simd(diskann_wide::ARCH, px.add(i)) };
        s = vx.mul_add_simd(vx, s);
        i += 8;
    }

    let remainder = len - i;
    if remainder != 0 {
        // SAFETY: The pointer add is valid because `i < len` (strict inequality), so the
        // base pointer belongs to the memory owned by `x`.
        //
        // Furthermore, the load is valid for the first `remainder` items.
        let vx = unsafe { f32s::load_simd_first(diskann_wide::ARCH, px.add(i), remainder) };
        s = vx.mul_add_simd(vx, s);
    }

    s.sum_tree()
}

#[cfg(test)]
mod tests {
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use super::*;

    /////////////////
    // Square Norm //
    /////////////////

    fn square_norm_reference(x: &[f32]) -> f32 {
        x.iter().map(|&i| i * i).sum()
    }

    fn test_square_norm_impl<R: Rng>(
        dim: usize,
        ntrials: usize,
        relative_error: f32,
        absolute_error: f32,
        rng: &mut R,
    ) {
        let distribution = Uniform::<f32>::new(-1.0, 1.0).unwrap();
        let mut x: Vec<f32> = vec![0.0; dim];
        for trial in 0..ntrials {
            x.iter_mut().for_each(|i| *i = distribution.sample(rng));
            let expected = square_norm_reference(&x);
            let got = square_norm(&x);

            let this_absolute_error = (expected - got).abs();
            let this_relative_error = this_absolute_error / expected.abs();

            let absolute_ok = this_absolute_error <= absolute_error;
            let relative_ok = this_relative_error <= relative_error;

            if !absolute_ok && !relative_ok {
                panic!(
                    "recieved abolute/relative errors of {}/{} when the bounds were {}/{}\n\
                     dim = {}, trial = {} of {}",
                    this_absolute_error,
                    this_relative_error,
                    absolute_error,
                    relative_error,
                    dim,
                    trial,
                    ntrials,
                )
            }
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const NTRIALS: usize = 1;
            const MAX_DIM: usize = 80;
        } else {
            const NTRIALS: usize = 100;
            const MAX_DIM: usize = 128;
        }
    }

    #[test]
    fn test_square_norm() {
        let mut rng = StdRng::seed_from_u64(0x71d00ad8c7105273);
        for dim in 0..MAX_DIM {
            let relative_error = 8.0e-7;
            let absolute_error = 1.0e-5;

            test_square_norm_impl(dim, NTRIALS, relative_error, absolute_error, &mut rng);
        }
    }
}
