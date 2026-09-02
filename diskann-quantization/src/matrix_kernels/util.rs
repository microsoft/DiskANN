/*
 * Copyright (c) Microsoft Corporationk:wa
 * .
 * Licensed under the MIT license.
 */

use diskann_wide::arch::Scalar;
use half::f16;

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

/////////////
// Convert //
/////////////

pub(super) trait Convert<To, From> {
    fn convert(self, to: &mut [To], from: &[From]);
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Converter<A>(A);

impl<A> Converter<A> {
    pub(super) const fn new(arch: A) -> Self {
        Self(arch)
    }
}

impl<A> Convert<f32, f16> for Converter<A>
where
    A: diskann_wide::Architecture,
    diskann_vector::conversion::SliceCast<f32, f16>:
        for<'a, 'b> diskann_wide::arch::Target2<A, (), &'a mut [f32], &'a [f16]>,
{
    fn convert(self, to: &mut [f32], from: &[f16]) {
        self.0.run2(
            diskann_vector::conversion::SliceCast::<f32, f16>::new(),
            to,
            from,
        )
    }
}

//////////
// Load //
//////////

pub(super) trait LoadStore<T, const N: usize>
where
    T: Copy,
{
    fn load(self, src: &[T]) -> [T; N];
    fn store(self, v: [T; N], dst: &mut [T]);
}

impl<T, const N: usize> LoadStore<T, N> for Scalar
where
    T: Copy + Default,
{
    fn load(self, src: &[T]) -> [T; N] {
        core::array::from_fn(|i| src.get(i).copied().unwrap_or(T::default()))
    }

    fn store(self, v: [T; N], dst: &mut [T]) {
        dst.copy_from_slice(&v[..dst.len()])
    }
}

macro_rules! impl_loadstore {
    ($T:ty, $N:literal, $wide:ident, $arch:ty) => {
        impl LoadStore<$T, $N> for $arch {
            #[inline(always)]
            fn load(self, src: &[$T]) -> [$T; $N] {
                use diskann_wide::SIMDVector;
                diskann_wide::alias!(wide = <$arch>::$wide);

                // SAFETY: Loading from `src` up to `src.len()` is valid.
                unsafe { wide::load_simd_first(self, src.as_ptr(), src.len()) }.to_array()
            }

            #[inline(always)]
            fn store(self, v: [$T; $N], dst: &mut [$T]) {
                use diskann_wide::SIMDVector;
                diskann_wide::alias!(wide = <$arch>::$wide);

                let w = wide::from_array(self, v);

                // SAFETY: Storing to `dst` up to `dst.len()` is valid.
                unsafe { wide::store_simd_first(w, dst.as_mut_ptr(), dst.len()) }
            }
        }
    };
}

impl_loadstore!(f32, 8, f32x8, V3);
impl_loadstore!(f32, 16, f32x16, V3);

impl_loadstore!(f32, 8, f32x8, V4);
impl_loadstore!(f32, 16, f32x16, V4);

impl LoadStore<f32, 32> for V4 {
    #[inline(always)]
    fn load(self, src: &[f32]) -> [f32; 32] {
        use diskann_wide::{LoHi, SIMDVector};
        diskann_wide::alias!(wide = <V4>::f32x16);

        // SAFETY: Loading the first `src.len().min(16)` elements from `src` is valid.
        let lo = unsafe { wide::load_simd_first(self, src.as_ptr(), src.len()) }.to_array();

        // SAFETY: This only reads `src.len() - 16` values if `src.len()` exceeds 16.
        let hi = unsafe {
            wide::load_simd_first(self, src.as_ptr().offset(16), src.len().saturating_sub(16))
        }
        .to_array();

        LoHi::new(lo, hi).join()
    }

    #[inline(always)]
    fn store(self, v: [f32; 32], dst: &mut [f32]) {
        use diskann_wide::{LoHi, SIMDVector, SplitJoin};
        diskann_wide::alias!(wide = <V4>::f32x16);

        let LoHi { lo, hi } = v.split();

        // SAFETY: Storing the first `dst.len().min(16)` elements to `dst` is valid.
        unsafe { wide::from_array(self, lo).store_simd_first(dst.as_mut_ptr(), dst.len()) };

        // SAFETY: This only writes if `dst.len() - 16` values if `dst.len()` exceeds 16.
        unsafe {
            wide::from_array(self, hi)
                .store_simd_first(dst.as_mut_ptr().offset(16), dst.len().saturating_sub(16))
        };
    }
}

//////////
// Fold //
//////////

#[derive(Debug, Clone, Copy)]
pub(super) struct Folder;

impl Folder {
    #[inline]
    pub(super) fn fold<const N: usize, T, F>(x: [T; N], f: F) -> T
    where
        Self: Fold<N>,
        F: Fn(T, T) -> T,
    {
        (Self).__fold(x, f)
    }
}

pub(super) trait Fold<const N: usize> {
    fn __fold<T, F>(self, x: [T; N], f: F) -> T
    where
        F: Fn(T, T) -> T;
}

impl Fold<1> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 1], _f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0] = x;
        a0
    }
}

impl Fold<2> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 2], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1] = x;
        f(a0, a1)
    }
}

impl Fold<3> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 3], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2] = x;
        f(f(a0, a1), a2)
    }
}

impl Fold<4> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 4], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3] = x;
        f(f(a0, a1), f(a2, a3))
    }
}

impl Fold<5> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 5], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3, a4] = x;
        self.__fold([f(a0, a1), f(a2, a3), a4], f)
    }
}

impl Fold<6> for Folder {
    #[inline]
    fn __fold<T, F>(self, x: [T; 6], f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        let [a0, a1, a2, a3, a4, a5] = x;
        self.__fold([f(a0, a1), f(a2, a3), f(a4, a5)], f)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_fold() {
        fn max(x: usize, y: usize) -> usize {
            x.max(y)
        }

        // One
        assert_eq!(Folder::fold([1], max), 1);
        assert_eq!(Folder::fold([2], max), 2);
        assert_eq!(Folder::fold([3], max), 3);

        // Two
        assert_eq!(Folder::fold([0, 10], max), 10);
        assert_eq!(Folder::fold([10, 0], max), 10);

        // Three
        assert_eq!(Folder::fold([0, 0, 10], max), 10);
        assert_eq!(Folder::fold([0, 10, 0], max), 10);
        assert_eq!(Folder::fold([10, 0, 0], max), 10);

        // Four
        assert_eq!(Folder::fold([0, 0, 0, 10], max), 10);
        assert_eq!(Folder::fold([0, 0, 10, 0], max), 10);
        assert_eq!(Folder::fold([0, 10, 0, 0], max), 10);
        assert_eq!(Folder::fold([10, 0, 0, 0], max), 10);
    }
}
