/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::arch::Scalar;
use half::f16;

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
    /// Load up to the first `src.len()` and return the results in an array.
    ///
    /// The remaining items items should be left in a default state.
    fn load(self, src: &[T]) -> [T; N];

    /// Store the first up-to `dst.len()` items in `v` into `dst`.
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

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::arch::x86_64::{V3, V4};

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
                wide::load_simd_first(
                    self,
                    src.as_ptr().wrapping_offset(16),
                    src.len().saturating_sub(16),
                )
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

            if let Some(rest) = dst.len().checked_sub(16) {
                // SAFETY: This only writes if `dst.len() - 16` values if `dst.len()` exceeds 16.
                unsafe {
                    wide::from_array(self, hi).store_simd_first(dst.as_mut_ptr().add(16), rest)
                };
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;

    use diskann_wide::arch::aarch64::Neon;

    impl_loadstore!(f32, 4, f32x4, Neon);
    impl_loadstore!(f32, 8, f32x8, Neon);
    impl_loadstore!(f32, 16, f32x16, Neon);
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

    #[cfg(target_arch = "x86_64")]
    use diskann_wide::arch::x86_64::{V3, V4};

    #[cfg(target_arch = "aarch64")]
    use diskann_wide::arch::aarch64::Neon;

    trait FromUsize {
        fn from_usize(v: usize) -> Self;
    }

    impl FromUsize for f32 {
        fn from_usize(v: usize) -> Self {
            v as f32
        }
    }

    fn double<T>(x: usize) -> T
    where
        T: FromUsize,
    {
        T::from_usize(2 * x)
    }

    fn test_load_store_inner<A, T, const N: usize>(arch: A)
    where
        A: LoadStore<T, N> + Copy,
        T: FromUsize + PartialEq + Copy + Default + std::fmt::Debug,
    {
        // For these loops - the source and destination slices are intentionally allocated
        // on each iteration to enable Miri to detect invalid, out-of-bounds accesses.
        for i in 0..2 * N {
            let src: Vec<T> = (0..i).map(double).collect();
            let mut dst: Vec<T> = vec![T::default(); N.min(i)];

            let expected: [T; N] =
                core::array::from_fn(|j| if j < i { double(j) } else { T::default() });

            // Load
            let v = arch.load(&src);
            assert_eq!(v, expected, "i = {i}");

            arch.store(v, &mut dst);
            assert_eq!(dst, &v[..N.min(i)], "i = {i}");
        }
    }

    macro_rules! test_load_store {
        ($f:ident, $arch:expr, $($T:ty => { $($N:literal),+ $(,)? }),+ $(,)?) => {
            #[test]
            fn $f() {
                if let Some(arch) = $arch {
                    $(
                        $(
                            test_load_store_inner::<_, $T, $N>(arch);
                        )+
                    )+
                }
            }
        }
    }

    test_load_store!(
        test_load_store_scalar,
        Some(Scalar),
        f32 => { 4, 8, 16 },
    );

    #[cfg(target_arch = "x86_64")]
    test_load_store!(
        test_load_store_v3,
        V3::new_checked(),
        f32 => { 8, 16 },
    );

    #[cfg(target_arch = "x86_64")]
    test_load_store!(
        test_load_store_v4,
        V4::new_checked_miri(),
        f32 => { 8, 16, 32 },
    );

    #[cfg(target_arch = "aarch64")]
    test_load_store!(
        test_load_store_neon,
        Neon::new_checked(),
        f32 => { 4, 8, 16 },
    );

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

        // Five
        assert_eq!(Folder::fold([0, 0, 0, 0, 10], max), 10);
        assert_eq!(Folder::fold([0, 0, 0, 10, 0], max), 10);
        assert_eq!(Folder::fold([0, 0, 10, 0, 0], max), 10);
        assert_eq!(Folder::fold([0, 10, 0, 0, 0], max), 10);
        assert_eq!(Folder::fold([10, 0, 0, 0, 0], max), 10);

        // Six
        assert_eq!(Folder::fold([0, 0, 0, 0, 0, 10], max), 10);
        assert_eq!(Folder::fold([0, 0, 0, 0, 10, 0], max), 10);
        assert_eq!(Folder::fold([0, 0, 0, 10, 0, 0], max), 10);
        assert_eq!(Folder::fold([0, 0, 10, 0, 0, 0], max), 10);
        assert_eq!(Folder::fold([0, 10, 0, 0, 0, 0], max), 10);
        assert_eq!(Folder::fold([10, 0, 0, 0, 0, 0], max), 10);
    }
}
