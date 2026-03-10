/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::traits::{SIMDMask, SIMDVector};

// Helper macros to more quickly define intrinsics.
// The pattern used for all the x86 definitions is pretty uniform, so wrap it into
// a macro.
macro_rules! x86_define_register {
    ($type:ident, $impl:ty, $mask:ty, $scalar:ty, $lanes:literal, $arch:ty) => {
        #[derive(Debug, Clone, Copy)]
        #[allow(non_camel_case_types)]
        #[repr(transparent)]
        pub struct $type(pub $impl);

        impl $type {
            /// Convert `self` to its corresponding [`crate::Emulated`] type.
            #[inline(always)]
            pub fn emulated(self) -> $crate::Emulated<$scalar, $lanes> {
                $crate::Emulated::from_array($crate::arch::Scalar, self.to_array())
            }
        }

        impl $crate::AsSIMD<$type> for $crate::Emulated<$scalar, $lanes> {
            #[inline(always)]
            fn as_simd(self, arch: $arch) -> $type {
                $type::from_array(arch, self.to_array())
            }
        }

        impl SIMDVector for $type {
            type Arch = $arch;
            type Scalar = $scalar;
            type Underlying = $impl;

            type Mask = $mask;
            type ConstLanes = Const<$lanes>;
            const LANES: usize = $lanes;
            const EMULATED: bool = false;

            #[inline(always)]
            fn arch(self) -> $arch {
                // SAFETY: The existence of `self` provides a witness that it is safe to
                // instantiate its architecture.
                unsafe { <$arch>::new() }
            }

            #[inline(always)]
            fn default(arch: $arch) -> Self {
                <Self as X86Default>::x86_default(arch)
            }

            #[inline(always)]
            fn to_underlying(self) -> Self::Underlying {
                self.0
            }

            #[inline(always)]
            fn from_underlying(_: $arch, repr: Self::Underlying) -> Self {
                Self(repr)
            }

            #[inline(always)]
            fn to_array(self) -> [$scalar; $lanes] {
                // SAFETY: Provided the scalar type is an integer or floating point,
                // then all bit patterns are valid between source and destination types.
                // (provided an x86 intrinsic is one of the transmuted types).
                //
                // The source argument is taken by value (no reference conversion) and
                // as long as `T` is `[repr(C)]`, then `[T; N]` will be `[repr(C)]`.
                //
                // The intrinsic types are `[repr(simd)]` which amounts to `[repr(C)]` and
                // change.
                unsafe { std::mem::transmute::<Self, [$scalar; $lanes]>(self) }
            }

            #[inline(always)]
            fn from_array(_: $arch, x: [$scalar; $lanes]) -> Self {
                // SAFETY: Provided the scalar type is an integer or floating point,
                // then all bit patterns are valid between source and destination types.
                // (provided an x86 intrinsic is one of the transmuted types).
                //
                // The source argument is taken by value (no reference conversion) and
                // as long as `T` is `[repr(C)]`, then `[T; N]` will be `[repr(C)]`.
                //
                // The intrinsic types are `[repr(simd)]` which amounts to `[repr(C)]` and
                // change.
                unsafe { std::mem::transmute::<[$scalar; $lanes], Self>(x) }
            }

            #[inline(always)]
            fn splat(arch: $arch, value: Self::Scalar) -> Self {
                <Self as X86Splat>::x86_splat(arch, value)
            }

            #[inline(always)]
            unsafe fn load_simd(arch: $arch, ptr: *const $scalar) -> Self {
                // SAFETY: This has the same safety constraints as the caller.
                unsafe { <Self as X86LoadStore>::load_simd(arch, ptr) }
            }

            #[inline(always)]
            unsafe fn load_simd_masked_logical(
                arch: $arch,
                ptr: *const $scalar,
                mask: $mask,
            ) -> Self {
                // SAFETY: This has the same safety constraints as the caller.
                unsafe { <Self as X86LoadStore>::load_simd_masked_logical(arch, ptr, mask) }
            }

            #[inline(always)]
            unsafe fn load_simd_first(arch: $arch, ptr: *const $scalar, first: usize) -> Self {
                // SAFETY: This has the same safety constraints as the caller.
                unsafe { <Self as X86LoadStore>::load_simd_first(arch, ptr, first) }
            }

            #[inline(always)]
            unsafe fn store_simd(self, ptr: *mut $scalar) {
                // SAFETY: This has the same safety constraints as the caller.
                unsafe { <Self as X86LoadStore>::store_simd(self, ptr) }
            }

            #[inline(always)]
            unsafe fn store_simd_masked_logical(self, ptr: *mut $scalar, mask: $mask) {
                // SAFETY: This has the same safety constraints as the caller.
                unsafe { <Self as X86LoadStore>::store_simd_masked_logical(self, ptr, mask) }
            }

            #[inline(always)]
            unsafe fn store_simd_first(self, ptr: *mut $scalar, first: usize) {
                // SAFETY: This has the same safety constraints as the caller.
                unsafe { <Self as X86LoadStore>::store_simd_first(self, ptr, first) }
            }
        }
    };
}

// Externalize splat implementations to enable fine-grained overloading.
pub(super) trait X86Splat: SIMDVector {
    fn x86_splat(arch: <Self as SIMDVector>::Arch, value: <Self as SIMDVector>::Scalar) -> Self;
}

pub(super) trait X86Default: SIMDVector {
    fn x86_default(arch: <Self as SIMDVector>::Arch) -> Self;
}

pub(super) trait X86LoadStore: SIMDVector {
    unsafe fn load_simd(
        arch: <Self as SIMDVector>::Arch,
        ptr: *const <Self as SIMDVector>::Scalar,
    ) -> Self;
    unsafe fn load_simd_masked_logical(
        arch: <Self as SIMDVector>::Arch,
        ptr: *const <Self as SIMDVector>::Scalar,
        mask: Self::Mask,
    ) -> Self;
    unsafe fn load_simd_first(
        arch: <Self as SIMDVector>::Arch,
        ptr: *const <Self as SIMDVector>::Scalar,
        first: usize,
    ) -> Self {
        // SAFETY: The implementation of `X86LoadStore` is trusted.
        unsafe {
            <Self as X86LoadStore>::load_simd_masked_logical(
                arch,
                ptr,
                Self::Mask::keep_first(arch, first),
            )
        }
    }

    unsafe fn store_simd(self, ptr: *mut <Self as SIMDVector>::Scalar);
    unsafe fn store_simd_masked_logical(
        self,
        ptr: *mut <Self as SIMDVector>::Scalar,
        mask: Self::Mask,
    );
    unsafe fn store_simd_first(self, ptr: *mut <Self as SIMDVector>::Scalar, first: usize) {
        // SAFETY: The implementation of `X86LoadStore` is trusted.
        unsafe {
            <Self as X86LoadStore>::store_simd_masked_logical(
                self,
                ptr,
                Self::Mask::keep_first(self.arch(), first),
            )
        }
    }
}

macro_rules! x86_retarget {
    ($T:path => $U:path) => {
        impl $T {
            #[inline(always)]
            pub fn retarget(self) -> $U {
                <$U>::from_underlying(self.arch().into(), self.to_underlying())
            }

            pub fn from(self, other: $U) -> Self {
                Self::from_underlying(self.arch(), other.to_underlying())
            }
        }
    };
}

/// Utility macro for defining `X86Splat`.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call.
/// That is - any intrinsics invoked must be compatible with `$type`'s associated architecture.
macro_rules! x86_define_splat {
    ($type:ty, $intrinsic:expr, $requires:literal) => {
        impl X86Splat for $type {
            #[inline(always)]
            fn x86_splat(
                _: <Self as SIMDVector>::Arch,
                value: <Self as SIMDVector>::Scalar,
            ) -> Self {
                // SAFETY: The presence of `Arch` proves that this function is safe to call.
                Self(unsafe { $intrinsic(value) })
            }
        }
    };
    // This variant of the macro performs a bitcast to the value that needs to be
    // broadcasted in order to get the types correct for the x86 intrinsic.
    ($type:ty as $cast:ty, $intrinsic:expr, $requires:literal) => {
        impl X86Splat for $type {
            #[inline(always)]
            fn x86_splat(
                _: <Self as SIMDVector>::Arch,
                value: <Self as SIMDVector>::Scalar,
            ) -> Self {
                // SAFETY: The presence of `Arch` proves that this function is safe to call.
                Self(unsafe { $intrinsic(value as $cast) })
            }
        }
    };
}

/// Utility macro for defining `X86Default`.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call.
/// That is - any intrinsics invoked must be compatible with `$type`'s associated architecture.
macro_rules! x86_define_default {
    ($type:ty, $intrinsic:expr, $requires:literal) => {
        impl X86Default for $type {
            #[inline(always)]
            fn x86_default(_: <Self as SIMDVector>::Arch) -> Self {
                // SAFETY: The invoker of this macro must pass the `target_feature`
                // requirement of the intrinsic.
                //
                // That way, if the intrinsic is not available, we get a compile-time error.
                Self(unsafe { $intrinsic() })
            }
        }
    };
}

/// SAFETY: It is the invoker's responsibility to ensure that the provided intrinsics are
/// safe to call.
///
/// That is - any intrinsics invoked must be compatible with `$type`'s associated architecture.
macro_rules! x86_splitjoin {
    (__m512i, $type:path, $half:path) => {
        impl $crate::SplitJoin for $type {
            type Halved = $half;

            #[inline(always)]
            fn split(self) -> crate::LoHi<Self::Halved> {
                // SAFETY: This must only be instantiated for architecture supporting AVX512DQ.
                unsafe {
                    crate::LoHi::new(
                        Self::Halved::from_underlying(
                            self.arch(),
                            _mm512_extracti32x8_epi32(self.0, 0),
                        ),
                        Self::Halved::from_underlying(
                            self.arch(),
                            _mm512_extracti32x8_epi32(self.0, 1),
                        ),
                    )
                }
            }

            #[inline(always)]
            fn join(lohi: crate::LoHi<Self::Halved>) -> Self {
                // SAFETY: Required by instantiator.
                let v = Self::default(lohi.lo.arch()).to_underlying();

                // SAFETY: `_mm512_inserti32x8` requires `AVX512DQ`.
                let v = unsafe {
                    _mm512_inserti32x8(_mm512_inserti32x8(v, lohi.lo.0, 0), lohi.hi.0, 1)
                };
                Self(v)
            }
        }
    };
    ($type:path, $half:path, $split:path, $join:path, $requires:literal) => {
        impl $crate::SplitJoin for $type {
            type Halved = $half;

            #[inline(always)]
            fn split(self) -> $crate::LoHi<$half> {
                // SAFETY: Required by instantiator.
                unsafe { $crate::LoHi::new($half($split(self.0, 0)), $half($split(self.0, 1))) }
            }

            #[inline(always)]
            fn join(lohi: $crate::LoHi<$half>) -> Self {
                // SAFETY: Required by instantiator.
                Self(unsafe { $join(lohi.hi.0, lohi.lo.0) })
            }
        }
    };
}

/// Implement [`ZipUnzip`] for a 256-bit vector type (`$type`) whose halved
/// type (`$half`) is a 128-bit vector, using within-lane shuffles.
///
/// The `zip` / `unzip` methods (producing `LoHi`) use 128-bit `pshufb` +
/// `punpckl/hqdq` (5 µops).  The `zip_flat` / `unzip_flat` methods use 256-bit
/// `vpshufb` + `vpermd` (2 µops) by rearranging within each lane and then
/// fixing the cross-lane dword ordering.
///
/// # Parameters
///
/// * `$unpacklo`, `$unpackhi` — 128-bit unpack intrinsics for the `zip` path
/// * `$deinterleave_mask` — 128-bit `pshufb` mask that groups even-indexed
///   elements in the low qword and odd-indexed in the high qword
/// * `$interleave_mask` — 128-bit `pshufb` mask that does the inverse:
///   given `[evens..., odds...]`, produces `[e0, o0, e1, o1, ...]`
///
/// # Safety
///
/// The caller must ensure that the supplied intrinsics are valid for the
/// architecture token stored in `$type`.
macro_rules! x86_zipunzip {
    (
        $type:path, $half:path,
        $unpacklo:ident, $unpackhi:ident,
        $deinterleave_mask:expr,
        $interleave_mask:expr
    ) => {
        impl $crate::ZipUnzip for $type {
            #[inline(always)]
            fn zip(halves: $crate::LoHi<<Self as $crate::SplitJoin>::Halved>) -> Self {
                use $crate::SplitJoin;
                // SAFETY: Caller asserts that these intrinsics are within the
                // capabilities of the architecture stored in the type.
                unsafe {
                    let lo_raw = halves.lo.to_underlying();
                    let hi_raw = halves.hi.to_underlying();
                    let lo = $unpacklo(lo_raw, hi_raw);
                    let hi = $unpackhi(lo_raw, hi_raw);
                    <$type>::join($crate::LoHi::new(
                        <$half>::from_underlying(halves.lo.arch(), lo),
                        <$half>::from_underlying(halves.lo.arch(), hi),
                    ))
                }
            }

            #[inline(always)]
            fn unzip(self) -> $crate::LoHi<<Self as $crate::SplitJoin>::Halved> {
                use $crate::SplitJoin;
                let halves = self.split();
                // SAFETY: Caller asserts that these intrinsics are within the
                // capabilities of the architecture stored in the type.
                unsafe {
                    let mask = $deinterleave_mask;
                    let lo = _mm_shuffle_epi8(halves.lo.to_underlying(), mask);
                    let hi = _mm_shuffle_epi8(halves.hi.to_underlying(), mask);
                    let evens = _mm_unpacklo_epi64(lo, hi);
                    let odds = _mm_unpackhi_epi64(lo, hi);
                    $crate::LoHi::new(
                        <$half>::from_underlying(halves.lo.arch(), evens),
                        <$half>::from_underlying(halves.lo.arch(), odds),
                    )
                }
            }

            #[inline(always)]
            fn zip_flat(self) -> Self {
                // vpermd to swap middle dword-pairs, then vpshufb to interleave
                // within each lane.
                // SAFETY: Caller asserts AVX2 is available.
                unsafe {
                    let dword_fixup = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
                    let fixed = _mm256_permutevar8x32_epi32(self.0, dword_fixup);
                    let mask128 = $interleave_mask;
                    let mask256 = _mm256_set_m128i(mask128, mask128);
                    Self(_mm256_shuffle_epi8(fixed, mask256))
                }
            }

            #[inline(always)]
            fn unzip_flat(self) -> Self {
                // vpshufb to deinterleave within each lane, then vpermd to fix
                // cross-lane dword ordering.
                // SAFETY: Caller asserts AVX2 is available.
                unsafe {
                    let mask128 = $deinterleave_mask;
                    let mask256 = _mm256_set_m128i(mask128, mask128);
                    let shuffled = _mm256_shuffle_epi8(self.0, mask256);
                    let dword_fixup = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
                    Self(_mm256_permutevar8x32_epi32(shuffled, dword_fixup))
                }
            }
        }
    };
}

/// Implement [`ZipUnzip`] for a 256-bit vector type (`$type`) with 32-bit
/// elements, using a single cross-lane `vpermd` for the flat variants.
///
/// Both `zip_flat` and `unzip_flat` compile to one `_mm256_permutevar8x32_epi32`.
/// The `zip` and `unzip` methods delegate through `join`/`split` respectively.
///
/// # Safety
///
/// The caller must ensure AVX2 (or better) is available.
macro_rules! x86_zipunzip_perm32 {
    ($type:path) => {
        impl $crate::ZipUnzip for $type {
            #[inline(always)]
            fn zip(halves: $crate::LoHi<<Self as $crate::SplitJoin>::Halved>) -> Self {
                <Self as $crate::SplitJoin>::join(halves).zip_flat()
            }

            #[inline(always)]
            fn unzip(self) -> $crate::LoHi<<Self as $crate::SplitJoin>::Halved> {
                <Self as $crate::SplitJoin>::split(self.unzip_flat())
            }

            #[inline(always)]
            fn zip_flat(self) -> Self {
                // Interleave: [a0,a1,a2,a3,b0,b1,b2,b3] → [a0,b0,a1,b1,a2,b2,a3,b3]
                // SAFETY: Caller asserts AVX2 is available.
                unsafe {
                    let idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                    Self(_mm256_permutevar8x32_epi32(self.0, idx))
                }
            }

            #[inline(always)]
            fn unzip_flat(self) -> Self {
                // Deinterleave: [a0,b0,a1,b1,a2,b2,a3,b3] → [a0,a1,a2,a3,b0,b1,b2,b3]
                // SAFETY: Caller asserts AVX2 is available.
                unsafe {
                    let idx = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
                    Self(_mm256_permutevar8x32_epi32(self.0, idx))
                }
            }
        }
    };
}

/// Implement [`ZipUnzip`] for a 256-bit vector type using a single cross-lane
/// permute instruction.
///
/// The caller provides the permute intrinsic (`$perm`) and index constructor
/// (`$setr`). The macro builds interleave and deinterleave index vectors and
/// generates `zip_flat`/`unzip_flat` overrides; `zip`/`unzip` delegate through
/// `join`/`split`.
///
/// # Parameters
///
/// * `$type`  — the 256-bit vector type (e.g. `u8x32`)
/// * `$perm`  — the permute intrinsic, e.g. `_mm256_permutexvar_epi8`
/// * `$zip_idx` — expression producing `__m256i` with the zip index vector
/// * `$unzip_idx` — expression producing `__m256i` with the unzip index vector
///
/// # Safety
///
/// The caller must ensure the permute intrinsic is within the capabilities of
/// the architecture token stored in `$type`.
///
/// **Note**: Currently unused — V4 does not yet enable VBMI/BW. The macro is
/// retained for the upcoming Ice Lake+ target-feature PR.
#[allow(unused_macros)]
macro_rules! x86_zipunzip_crosslane {
    ($type:path, $perm:ident, $zip_idx:expr, $unzip_idx:expr) => {
        impl $crate::ZipUnzip for $type {
            #[inline(always)]
            fn zip(halves: $crate::LoHi<<Self as $crate::SplitJoin>::Halved>) -> Self {
                <Self as $crate::SplitJoin>::join(halves).zip_flat()
            }

            #[inline(always)]
            fn unzip(self) -> $crate::LoHi<<Self as $crate::SplitJoin>::Halved> {
                <Self as $crate::SplitJoin>::split(self.unzip_flat())
            }

            #[inline(always)]
            fn zip_flat(self) -> Self {
                // SAFETY: Caller asserts the permute intrinsic is available.
                unsafe {
                    let idx = $zip_idx;
                    Self($perm(idx, self.0))
                }
            }

            #[inline(always)]
            fn unzip_flat(self) -> Self {
                // SAFETY: Caller asserts the permute intrinsic is available.
                unsafe {
                    let idx = $unzip_idx;
                    Self($perm(idx, self.0))
                }
            }
        }
    };
}

macro_rules! x86_avx512_int_comparisons {
    ($type:ty, $intrinsic:ident, $requires:literal) => {
        impl $crate::SIMDPartialEq for $type {
            #[inline(always)]
            fn eq_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Caller asserts that this intrinsic is safe to call for the
                // architecture stored in `$type`.
                Self::Mask::from_underlying(self.arch(), unsafe {
                    $intrinsic::<_MM_CMPINT_EQ>(self.0, other.0)
                })
            }

            #[inline(always)]
            fn ne_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Caller asserts that this intrinsic is safe to call for the
                // architecture stored in `$type`.
                Self::Mask::from_underlying(self.arch(), unsafe {
                    $intrinsic::<_MM_CMPINT_NE>(self.0, other.0)
                })
            }
        }

        impl $crate::SIMDPartialOrd for $type {
            #[inline(always)]
            fn lt_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Caller asserts that this intrinsic is safe to call for the
                // architecture stored in `$type`.
                Self::Mask::from_underlying(self.arch(), unsafe {
                    $intrinsic::<_MM_CMPINT_LT>(self.0, other.0)
                })
            }

            #[inline(always)]
            fn le_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Caller asserts that this intrinsic is safe to call for the
                // architecture stored in `$type`.
                Self::Mask::from_underlying(self.arch(), unsafe {
                    $intrinsic::<_MM_CMPINT_LE>(self.0, other.0)
                })
            }

            #[inline(always)]
            fn gt_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Caller asserts that this intrinsic is safe to call for the
                // architecture stored in `$type`.
                Self::Mask::from_underlying(self.arch(), unsafe {
                    $intrinsic::<_MM_CMPINT_NLE>(self.0, other.0)
                })
            }

            #[inline(always)]
            fn ge_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Caller asserts that this intrinsic is safe to call for the
                // architecture stored in `$type`.
                Self::Mask::from_underlying(self.arch(), unsafe {
                    $intrinsic::<_MM_CMPINT_NLT>(self.0, other.0)
                })
            }
        }
    };
}

macro_rules! x86_avx512_load_store {
    ($T:ty,
     $load:ident,
     $mask_load:ident,
     $store:ident,
     $mask_store:ident,
     $cast:ty,
     $requires:literal
    ) => {
        impl $crate::arch::x86_64::macros::X86LoadStore for $T {
            #[inline(always)]
            unsafe fn load_simd(
                arch: <Self as $crate::SIMDVector>::Arch,
                ptr: *const <Self as $crate::SIMDVector>::Scalar,
            ) -> Self {
                // SAFETY: Instantiator asserts that `$load` is withihn the capabilities
                // of the associated `Arch`.
                Self::from_underlying(arch, unsafe { $load(ptr.cast::<$cast>()) })
            }

            #[inline(always)]
            unsafe fn load_simd_masked_logical(
                arch: <Self as $crate::SIMDVector>::Arch,
                ptr: *const <Self as $crate::SIMDVector>::Scalar,
                mask: <Self as $crate::SIMDVector>::Mask,
            ) -> Self {
                // SAFETY: Instantiator asserts that `$mask_load` is withihn the capabilities
                // of the associated `Arch`.
                Self::from_underlying(arch, unsafe { $mask_load(mask.0, ptr.cast::<$cast>()) })
            }

            #[inline(always)]
            unsafe fn store_simd(self, ptr: *mut <Self as $crate::SIMDVector>::Scalar) {
                // SAFETY: Instantiator asserts that `$store` is withihn the capabilities
                // of the associated `Arch`.
                unsafe { $store(ptr.cast::<$cast>(), self.0) }
            }

            #[inline(always)]
            unsafe fn store_simd_masked_logical(
                self,
                ptr: *mut <Self as $crate::SIMDVector>::Scalar,
                mask: <Self as $crate::SIMDVector>::Mask,
            ) {
                // SAFETY: Instantiator asserts that `$mask_store` is withihn the capabilities
                // of the associated `Arch`.
                unsafe { $mask_store(ptr.cast::<$cast>(), mask.0, self.0) }
            }
        }
    };
}

pub(crate) use x86_avx512_int_comparisons;
pub(crate) use x86_avx512_load_store;
pub(crate) use x86_define_default;
pub(crate) use x86_define_register;
pub(crate) use x86_define_splat;
pub(crate) use x86_retarget;
pub(crate) use x86_splitjoin;
pub(crate) use x86_zipunzip;
#[allow(unused_imports)]
pub(crate) use x86_zipunzip_crosslane;
pub(crate) use x86_zipunzip_perm32;
