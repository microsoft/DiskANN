/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::Architecture;
#[cfg(any(test, target_arch = "x86_64"))]
use diskann_wide::{SIMDMulAdd, SIMDVector};
use thiserror::Error;

/// Implicitly multiply the argument `x` by a Hadamard matrix and scale the results by `1 / x.len().sqrt()`.
///
/// This function does not allocate and operates in place.
///
/// # Error
///
/// Returns an error if `x.len()` is not a power of two.
///
/// # See Also
///
/// * <https://en.wikipedia.org/wiki/Hadamard_matrix>
pub fn hadamard_transform(x: &mut [f32]) -> Result<(), NotPowerOfTwo> {
    // Defer application of target features until after retargeting `V4` to `V3`.
    //
    // This is because we do not have a better implementation for `V4`.
    diskann_wide::arch::dispatch1_no_features(HadamardTransform, x)
}

#[derive(Debug, Error)]
#[error("Hadamard input vector must have a length that is a power of two")]
pub struct NotPowerOfTwo;

/// Implicitly multiply the argument `x` by a Hadamard matrix and scale the results by `1 / x.len().sqrt()`.
///
/// This function does not allocate and operates in place.
///
/// # Error
///
/// Returns an error if `x.len()` is not a power of two.
///
/// # See Also
///
/// * <https://en.wikipedia.org/wiki/Hadamard_matrix>
#[derive(Debug, Clone, Copy)]
pub struct HadamardTransform;

impl diskann_wide::arch::Target1<diskann_wide::arch::Scalar, Result<(), NotPowerOfTwo>, &mut [f32]>
    for HadamardTransform
{
    #[inline(never)]
    fn run(self, arch: diskann_wide::arch::Scalar, x: &mut [f32]) -> Result<(), NotPowerOfTwo> {
        (HadamardTransformOuter).run(arch, x)
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V3,
        Result<(), NotPowerOfTwo>,
        &mut [f32],
    > for HadamardTransform
{
    #[inline(never)]
    fn run(self, arch: diskann_wide::arch::x86_64::V3, x: &mut [f32]) -> Result<(), NotPowerOfTwo> {
        arch.run1(HadamardTransformOuter, x)
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V4,
        Result<(), NotPowerOfTwo>,
        &mut [f32],
    > for HadamardTransform
{
    #[inline(never)]
    fn run(self, arch: diskann_wide::arch::x86_64::V4, x: &mut [f32]) -> Result<(), NotPowerOfTwo> {
        arch.retarget().run1(HadamardTransformOuter, x)
    }
}

////////////////////
// Implementation //
////////////////////

#[derive(Debug, Clone, Copy)]
pub struct HadamardTransformOuter;

impl<A> diskann_wide::arch::Target1<A, Result<(), NotPowerOfTwo>, &mut [f32]>
    for HadamardTransformOuter
where
    A: diskann_wide::Architecture,
    HadamardTransformRecursive: for<'a> diskann_wide::arch::Target1<A, (), &'a mut [f32]>,
{
    #[inline(always)]
    fn run(self, arch: A, x: &mut [f32]) -> Result<(), NotPowerOfTwo> {
        let len = x.len();

        if !len.is_power_of_two() {
            return Err(NotPowerOfTwo);
        }

        // Nothing to do for length-1 transforms.
        if len == 1 {
            return Ok(());
        }

        // Perform the implicit matrix multiplication.
        arch.run1(HadamardTransformRecursive, x);

        // Scale the result.
        let m = 1.0 / (x.len() as f32).sqrt();
        x.iter_mut().for_each(|i| *i *= m);

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct HadamardTransformRecursive;

impl diskann_wide::arch::Target1<diskann_wide::arch::Scalar, (), &mut [f32]>
    for HadamardTransformRecursive
{
    /// A recursive helper for the divide-and-conquer step of Hadamard matrix multplication.
    ///
    /// # Preconditions
    ///
    /// This function is private with the following pre-conditions:
    ///
    /// * `x.len()` must be a power of 2.
    /// * `x.len()` must be at least 2.
    #[inline]
    fn run(self, arch: diskann_wide::arch::Scalar, x: &mut [f32]) {
        let len = x.len();
        debug_assert!(len.is_power_of_two());
        debug_assert!(len >= 2);

        if len == 2 {
            let l = x[0];
            let r = x[1];
            x[0] = l + r;
            x[1] = l - r;
        } else {
            // Recursive case - divide and conquer.
            let (left, right) = x.split_at_mut(len / 2);

            arch.run1(self, left);
            arch.run1(self, right);

            std::iter::zip(left.iter_mut(), right.iter_mut()).for_each(|(l, r)| {
                let a = *l + *r;
                let b = *l - *r;
                *l = a;
                *r = b;
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target1<diskann_wide::arch::x86_64::V3, (), &mut [f32]>
    for HadamardTransformRecursive
{
    /// A recursive helper for the divide-and-conquer step of Hadamard matrix multplication.
    ///
    /// # Preconditions
    ///
    /// This function is private with the following pre-conditions:
    ///
    /// * `x.len()` must be a power of 2.
    /// * `x.len()` must be at least 2.
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V3, x: &mut [f32]) {
        let len = x.len();
        debug_assert!(len.is_power_of_two());
        debug_assert!(len >= 2);

        if let Ok(array) = <&mut [f32] as TryInto<&mut [f32; 64]>>::try_into(x) {
            // We have a faster implementation for working with 64-elements at a time. Invoke
            // that if possible.
            //
            // Lint: This conversion into an array will never fail because we've checked that the
            // length is indeed 64.
            micro_kernel_64(arch, array);
        } else if len == 2 {
            // This is only reachable if the original argument to `hadamard_transform` was
            // shorter than 64.
            let l = x[0];
            let r = x[1];
            x[0] = l + r;
            x[1] = l - r;
        } else {
            // Recursive case - divide and conquer.
            let (left, right) = x.split_at_mut(len / 2);

            arch.run1(self, left);
            arch.run1(self, right);

            std::iter::zip(left.iter_mut(), right.iter_mut()).for_each(|(l, r)| {
                let a = *l + *r;
                let b = *l - *r;
                *l = a;
                *r = b;
            });
        }
    }
}

/// The 8x8 Hadamard matrix.
#[cfg(any(test, target_arch = "x86_64"))]
const HADAMARD_8: [[f32; 8]; 8] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
];

/// This micro-kernel computes a full 64-element Hadamard transform by first computing
/// eight 8-element transforms via a matrix multiplication kernel and then using the
/// recursive formulation to compute the full 64-element transform.
#[cfg(any(test, target_arch = "x86_64"))]
#[inline(always)]
fn micro_kernel_64<A>(arch: A, x: &mut [f32; 64])
where
    A: Architecture,
{
    // Output registers
    let mut d0 = A::f32x8::splat(arch, 0.0);
    let mut d1 = A::f32x8::splat(arch, 0.0);
    let mut d2 = A::f32x8::splat(arch, 0.0);
    let mut d3 = A::f32x8::splat(arch, 0.0);
    let mut d4 = A::f32x8::splat(arch, 0.0);
    let mut d5 = A::f32x8::splat(arch, 0.0);
    let mut d6 = A::f32x8::splat(arch, 0.0);
    let mut d7 = A::f32x8::splat(arch, 0.0);

    let p: *const f32 = HADAMARD_8.as_ptr().cast();
    let src: *const f32 = x.as_ptr();
    let mut process_patch = |offset: usize| {
        // SAFETY: The unsafe actions in the enclosing block all consist of performing
        // arithmetic on pointers and dereferencing said pointers.
        //
        // The pointers accessed are the pointer for the 8x8 Hadamard matrix (with 64 valid
        // entries), and the input array (also with valid entries).
        //
        // The argument `offset` takes the values 0, 2, 4, 6.
        //
        // All the pointer arithmetic is performed so that accesses with `offset` as one
        // of these four values is in-bounds.
        unsafe {
            let c0 = A::f32x8::load_simd(arch, p.add(8 * offset));
            let c1 = A::f32x8::load_simd(arch, p.add(8 * (offset + 1)));

            let r0 = A::f32x8::splat(arch, src.add(offset).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8).read());
            d0 = r0.mul_add_simd(c0, d0);
            d1 = r1.mul_add_simd(c0, d1);

            let r0 = A::f32x8::splat(arch, src.add(offset + 1).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 9).read());
            d0 = r0.mul_add_simd(c1, d0);
            d1 = r1.mul_add_simd(c1, d1);

            let r0 = A::f32x8::splat(arch, src.add(offset + 8 * 2).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8 * 3).read());
            d2 = r0.mul_add_simd(c0, d2);
            d3 = r1.mul_add_simd(c0, d3);

            let r0 = A::f32x8::splat(arch, src.add(offset + 8 * 2 + 1).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8 * 3 + 1).read());
            d2 = r0.mul_add_simd(c1, d2);
            d3 = r1.mul_add_simd(c1, d3);

            let r0 = A::f32x8::splat(arch, src.add(offset + 8 * 4).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8 * 5).read());
            d4 = r0.mul_add_simd(c0, d4);
            d5 = r1.mul_add_simd(c0, d5);

            let r0 = A::f32x8::splat(arch, src.add(offset + 8 * 4 + 1).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8 * 5 + 1).read());
            d4 = r0.mul_add_simd(c1, d4);
            d5 = r1.mul_add_simd(c1, d5);

            let r0 = A::f32x8::splat(arch, src.add(offset + 8 * 6).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8 * 7).read());
            d6 = r0.mul_add_simd(c0, d6);
            d7 = r1.mul_add_simd(c0, d7);

            let r0 = A::f32x8::splat(arch, src.add(offset + 8 * 6 + 1).read());
            let r1 = A::f32x8::splat(arch, src.add(offset + 8 * 7 + 1).read());
            d6 = r0.mul_add_simd(c1, d6);
            d7 = r1.mul_add_simd(c1, d7);
        }
    };

    // Do the 8x8 matrix multiplication to compute the eight individual 8-element transforms
    // and store the results into `d0-7`.
    for o in 0..4 {
        process_patch(2 * o);
    }

    // Now that we have the individual 8-dimensional transformations, we can begin swizzling
    // them together to construct the full 64-element transform.
    //
    // This computes four 16-element transforms.
    let e0 = d0 + d1;
    let e1 = d0 - d1;

    let e2 = d2 + d3;
    let e3 = d2 - d3;

    let e4 = d4 + d5;
    let e5 = d4 - d5;

    let e6 = d6 + d7;
    let e7 = d6 - d7;

    // Compute two 32-element transforms.
    let f0 = e0 + e2;
    let f1 = e1 + e3;

    let f2 = e0 - e2;
    let f3 = e1 - e3;

    let f4 = e4 + e6;
    let f5 = e5 + e7;

    let f6 = e4 - e6;
    let f7 = e5 - e7;

    // Compute the full 64-element transform and write-back the results.
    let dst: *mut f32 = x.as_mut_ptr();

    // SAFETY: The pointer `dst` is valid for writing up to 64-elements, which is what
    // we do here.
    unsafe {
        (f0 + f4).store_simd(dst);
        (f1 + f5).store_simd(dst.add(8));
        (f2 + f6).store_simd(dst.add(16));
        (f3 + f7).store_simd(dst.add(24));
        (f0 - f4).store_simd(dst.add(32));
        (f1 - f5).store_simd(dst.add(40));
        (f2 - f6).store_simd(dst.add(48));
        (f3 - f7).store_simd(dst.add(56));
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;
    use diskann_utils::views::{self, Matrix, MatrixView};

    /// Retrieve the 8x8 hadamard matrix as a `Matrix`.
    fn get_hadamard_8() -> Matrix<f32> {
        let v: Box<[f32]> = HADAMARD_8.iter().flatten().copied().collect();
        Matrix::try_from(v, 8, 8).unwrap()
    }

    fn hadamard_by_sylvester(dim: usize) -> Matrix<f32> {
        assert_ne!(dim, 0);
        // Base case.
        if dim == 1 {
            Matrix::new(1.0, dim, dim)
        } else {
            let half = dim / 2;
            let sub = hadamard_by_sylvester(half);
            let mut m = Matrix::<f32>::new(0.0, dim, dim);

            for c in 0..m.ncols() {
                for r in 0..m.nrows() {
                    let mut v = sub[(r % half, c % half)];
                    if c >= half && r >= half {
                        v = -v;
                    }
                    m[(c, r)] = v;
                }
            }
            m
        }
    }

    // Ensure that our 8x8 constant Hadamard matrix stays consistent.
    #[test]
    fn test_hadamard_8() {
        let h8 = get_hadamard_8();
        let reference = hadamard_by_sylvester(8);
        assert_eq!(h8.as_slice(), reference.as_slice());
    }

    // A naive reference implementation.
    fn matmul(a: MatrixView<f32>, b: MatrixView<f32>) -> Matrix<f32> {
        assert_eq!(a.ncols(), b.nrows());
        let mut c = Matrix::new(0.0, a.nrows(), b.ncols());

        for i in 0..c.nrows() {
            for j in 0..c.ncols() {
                let mut v = 0.0;
                for k in 0..a.ncols() {
                    v = a[(i, k)].mul_add(b[(k, j)], v);
                }
                c[(i, j)] = v;
            }
        }
        c
    }

    #[test]
    fn test_micro_kernel_64() {
        let mut src = {
            let mut rng = StdRng::seed_from_u64(0xde1936d651285fc8);
            let init = views::Init(|| StandardUniform {}.sample(&mut rng));
            Matrix::new(init, 64, 1)
        };

        let h = hadamard_by_sylvester(64);
        let reference = matmul(h.as_view(), src.as_view());

        micro_kernel_64(diskann_wide::ARCH, src.as_mut_slice().try_into().unwrap());

        assert_eq!(reference.nrows(), src.nrows());
        assert_eq!(reference.ncols(), 1);
        assert_eq!(src.ncols(), 1);

        for j in 0..src.nrows() {
            let src = src[(j, 0)];
            let reference = reference[(j, 0)];

            let relative_error = (src - reference).abs() / src.abs().max(reference.abs());
            assert!(
                relative_error < 5e-6,
                "Got a relative error of {} for row {} - reference = {}, got = {}",
                relative_error,
                j,
                reference,
                src
            );
        }
    }

    // End-to-end tests.
    fn test_hadamard_transform(dim: usize, seed: u64) {
        let src = {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = views::Init(|| StandardUniform {}.sample(&mut rng));
            Matrix::new(init, dim, 1)
        };

        let h = hadamard_by_sylvester(dim);

        let mut reference = matmul(h.as_view(), src.as_view());
        reference
            .as_mut_slice()
            .iter_mut()
            .for_each(|i| *i /= (dim as f32).sqrt());

        // Queue up a list of implementations.
        type Implementation = Box<dyn Fn(&mut [f32])>;

        #[cfg_attr(not(target_arch = "x86_64"), expect(unused_mut))]
        let mut impls: Vec<(Implementation, &'static str)> = vec![
            (
                Box::new(|x| hadamard_transform(x).unwrap()),
                "public entry point",
            ),
            (
                Box::new(|x| {
                    diskann_wide::arch::Scalar::new()
                        .run1(HadamardTransform, x)
                        .unwrap()
                }),
                "scalar recursive implementation",
            ),
        ];

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            impls.push((
                Box::new(move |x| arch.run1(HadamardTransform, x).unwrap()),
                "x86-64-v3",
            ));
        }

        for (f, kernel) in impls.into_iter() {
            let mut src_clone = src.clone();
            f(src_clone.as_mut_slice());

            assert_eq!(reference.nrows(), src_clone.nrows());
            assert_eq!(reference.ncols(), 1);
            assert_eq!(src_clone.ncols(), 1);

            for j in 0..src_clone.nrows() {
                let src_clone = src_clone[(j, 0)];
                let reference = reference[(j, 0)];

                let relative_error =
                    (src_clone - reference).abs() / src_clone.abs().max(reference.abs());
                assert!(
                    relative_error < 5e-5,
                    "Got a relative error of {} for row {} - reference = {}, got = {} -- dim = {}: kernel = {}",
                    relative_error,
                    j,
                    reference,
                    src_clone,
                    dim,
                    kernel,
                );
            }
        }
    }

    #[test]
    fn test_hadamard_transform_1() {
        test_hadamard_transform(1, 0xcdb7283f806f237d);
    }

    #[test]
    fn test_hadamard_transform_2() {
        test_hadamard_transform(2, 0x1e8bba190423842c);
    }

    #[test]
    fn test_hadamard_transform_4() {
        test_hadamard_transform(4, 0x6cdcb7e1fe0fa296);
    }

    #[test]
    fn test_hadamard_transform_8() {
        test_hadamard_transform(8, 0xd120b32a83158c80);
    }

    #[test]
    fn test_hadamard_transform_16() {
        test_hadamard_transform(16, 0x56ef310cc7e42faa);
    }

    #[test]
    fn test_hadamard_transform_32() {
        test_hadamard_transform(32, 0xf2a1395699390b95);
    }

    #[test]
    fn test_hadamard_transform_64() {
        test_hadamard_transform(64, 0x31e6a1bfe4958c8a);
    }

    #[test]
    fn test_hadamard_transform_128() {
        test_hadamard_transform(128, 0xe13a35f4b9392747);
    }

    #[test]
    fn test_hadamard_transform_256() {
        test_hadamard_transform(256, 0xf71bb8e26e79681c);
    }

    // Test the error cases.
    #[test]
    fn test_error() {
        // Supplying an empty-slice is an error.
        assert!(matches!(hadamard_transform(&mut []), Err(NotPowerOfTwo)));

        for dim in [3, 31, 33, 40, 63, 65, 100, 127, 129] {
            let mut v = vec![0.0f32; dim];
            assert!(matches!(hadamard_transform(&mut v), Err(NotPowerOfTwo)));
        }
    }
}
