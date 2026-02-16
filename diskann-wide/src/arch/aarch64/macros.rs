/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::SIMDVector;

macro_rules! aarch64_define_register {
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
                <Self as AArchSplat>::aarch_default(arch)
            }

            fn to_underlying(self) -> Self::Underlying {
                self.0
            }

            fn from_underlying(_: $arch, repr: Self::Underlying) -> Self {
                Self(repr)
            }

            fn to_array(self) -> [$scalar; $lanes] {
                // SAFETY: Provided the scalar type is an integer or floating point,
                // then all bit pattens are valid between source and destination types.
                // (provided an x86 intrinsic is one of the transmuted types).
                //
                // The source argument is taken by value (no reference conversion) and
                // as long as `T` is `[repr(C)]`, then `[T; N]` will be `[repr(C)]`.
                //
                // The intrinsic types are `[repr(simd)]` which amounts to `[repr(C)]` and
                // change.
                unsafe { std::mem::transmute::<Self, [$scalar; $lanes]>(self) }
            }

            fn from_array(_: $arch, x: [$scalar; $lanes]) -> Self {
                // SAFETY: Provided the scalar type is an integer or floating point,
                // then all bit pattens are valid between source and destination types.
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
                <Self as AArchSplat>::aarch_splat(arch, value)
            }

            #[inline(always)]
            unsafe fn load_simd(arch: $arch, ptr: *const $scalar) -> Self {
                // SAFETY: Inherited from caller.
                unsafe { <Self as AArchLoadStore>::load_simd(arch, ptr) }
            }

            #[inline(always)]
            unsafe fn load_simd_masked_logical(
                arch: $arch,
                ptr: *const $scalar,
                mask: $mask,
            ) -> Self {
                // SAFETY: Inherited from caller.
                unsafe { <Self as AArchLoadStore>::load_simd_masked_logical(arch, ptr, mask) }
            }

            #[inline(always)]
            unsafe fn load_simd_first(arch: $arch, ptr: *const $scalar, first: usize) -> Self {
                // SAFETY: Inherited from caller.
                unsafe { <Self as AArchLoadStore>::load_simd_first(arch, ptr, first) }
            }

            #[inline(always)]
            unsafe fn store_simd(self, ptr: *mut $scalar) {
                // SAFETY: Inherited from caller.
                unsafe { <Self as AArchLoadStore>::store_simd(self, ptr) }
            }

            #[inline(always)]
            unsafe fn store_simd_masked_logical(self, ptr: *mut $scalar, mask: $mask) {
                // SAFETY: Inherited from caller.
                unsafe { <Self as AArchLoadStore>::store_simd_masked_logical(self, ptr, mask) }
            }

            #[inline(always)]
            unsafe fn store_simd_first(self, ptr: *mut $scalar, first: usize) {
                // SAFETY: Inherited from caller.
                unsafe { <Self as AArchLoadStore>::store_simd_first(self, ptr, first) }
            }
        }
    };
}

pub(super) trait AArchSplat: SIMDVector {
    fn aarch_splat(arch: <Self as SIMDVector>::Arch, value: <Self as SIMDVector>::Scalar) -> Self;

    fn aarch_default(arch: <Self as SIMDVector>::Arch) -> Self;
}

pub(super) trait AArchLoadStore: SIMDVector {
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
    ) -> Self;

    unsafe fn store_simd(self, ptr: *mut <Self as SIMDVector>::Scalar);
    unsafe fn store_simd_masked_logical(
        self,
        ptr: *mut <Self as SIMDVector>::Scalar,
        mask: Self::Mask,
    );
    unsafe fn store_simd_first(self, ptr: *mut <Self as SIMDVector>::Scalar, first: usize);
}

/// Utility macro for defining `AArchSplat`.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call.
macro_rules! aarch64_define_splat {
    ($type:ty, $intrinsic:expr) => {
        impl AArchSplat for $type {
            #[inline(always)]
            fn aarch_splat(
                _arch: <Self as SIMDVector>::Arch,
                value: <Self as SIMDVector>::Scalar,
            ) -> Self {
                // SAFETY: Instantiator asserts that `$intrinsic` is allowed by `Arch`.
                Self(unsafe { $intrinsic(value) })
            }

            #[inline(always)]
            fn aarch_default(arch: <Self as SIMDVector>::Arch) -> Self {
                Self::aarch_splat(arch, <Self as SIMDVector>::Scalar::default())
            }
        }
    };
}

macro_rules! aarch64_define_loadstore {
    ($type:ty, $load:expr, $store:expr, $lanes:literal) => {
        impl AArchLoadStore for $type {
            #[inline(always)]
            unsafe fn load_simd(
                _arch: <Self as SIMDVector>::Arch,
                ptr: *const <Self as SIMDVector>::Scalar,
            ) -> Self {
                // SAFETY: Instantiator asserts that `$load` is allowed by `Arch`.
                Self(unsafe { $load(ptr) })
            }

            #[inline(always)]
            unsafe fn load_simd_masked_logical(
                arch: <Self as SIMDVector>::Arch,
                ptr: *const <Self as SIMDVector>::Scalar,
                mask: Self::Mask,
            ) -> Self {
                // SAFETY: Inherited from caller.
                let e = unsafe {
                    Emulated::<_, $lanes>::load_simd_masked_logical(
                        $crate::arch::Scalar,
                        ptr,
                        mask.bitmask().as_scalar(),
                    )
                };

                Self::from_array(arch, e.to_array())
            }

            #[inline(always)]
            unsafe fn load_simd_first(
                arch: <Self as SIMDVector>::Arch,
                ptr: *const <Self as SIMDVector>::Scalar,
                first: usize,
            ) -> Self {
                // SAFETY: Inherited from caller.
                let e = unsafe {
                    Emulated::<_, $lanes>::load_simd_first($crate::arch::Scalar, ptr, first)
                };

                Self::from_array(arch, e.to_array())
            }

            #[inline(always)]
            unsafe fn store_simd(self, ptr: *mut <Self as SIMDVector>::Scalar) {
                // SAFETY: Instantiator asserts that `$store` is allowed by `Arch`.
                unsafe { $store(ptr, self.0) }
            }

            unsafe fn store_simd_masked_logical(
                self,
                ptr: *mut <Self as SIMDVector>::Scalar,
                mask: Self::Mask,
            ) {
                let e = Emulated::<_, $lanes>::from_array($crate::arch::Scalar, self.to_array());

                // SAFETY: Inherited from caller.
                unsafe { e.store_simd_masked_logical(ptr, mask.bitmask().as_scalar()) }
            }

            #[inline(always)]
            unsafe fn store_simd_first(self, ptr: *mut <Self as SIMDVector>::Scalar, first: usize) {
                let e = Emulated::<_, $lanes>::from_array($crate::arch::Scalar, self.to_array());

                // SAFETY: Inherited from caller.
                unsafe { e.store_simd_first(ptr, first) }
            }
        }
    };
}

macro_rules! aarch64_define_cmp {
    ($type:ty, $eq:ident, ($not:expr), $lt:ident, $le:ident, $gt:ident, $ge:ident) => {
        impl SIMDPartialEq for $type {
            #[inline(always)]
            fn eq_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self::Mask::from_underlying(self.arch(), unsafe { $eq(self.0, other.0) })
            }

            #[inline(always)]
            fn ne_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self::Mask::from_underlying(self.arch(), unsafe { $not($eq(self.0, other.0)) })
            }
        }

        impl SIMDPartialOrd for $type {
            #[inline(always)]
            fn lt_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self::Mask::from_underlying(self.arch(), unsafe { $lt(self.0, other.0) })
            }

            #[inline(always)]
            fn le_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self::Mask::from_underlying(self.arch(), unsafe { $le(self.0, other.0) })
            }

            #[inline(always)]
            fn gt_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self::Mask::from_underlying(self.arch(), unsafe { $gt(self.0, other.0) })
            }

            #[inline(always)]
            fn ge_simd(self, other: Self) -> Self::Mask {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self::Mask::from_underlying(self.arch(), unsafe { $ge(self.0, other.0) })
            }
        }
    };
}

/// Utility macro for defining simple operations that lower to a single intrinsic.
///
/// SAFETY: It is the invoker's responsibility to ensure that the intrinsic is safe to call
macro_rules! aarch64_define_fma {
    ($type:ty, integer) => {
        impl SIMDMulAdd for $type {
            #[inline(always)]
            fn mul_add_simd(self, rhs: Self, accumulator: Self) -> $type {
                self * rhs + accumulator
            }
        }
    };
    // This variant maps the implementation to an intrinsic.
    ($type:ty, $intrinsic:expr) => {
        impl SIMDMulAdd for $type {
            #[inline(always)]
            fn mul_add_simd(self, rhs: Self, accumulator: Self) -> $type {
                // SAFETY: The invoker of this macro must pass the `target_feature`
                // requirement of the intrinsic.
                //
                // That way, if the intrinsic is not available, we get a compile-time error.
                Self(unsafe { $intrinsic(accumulator.0, self.0, rhs.0) })
            }
        }
    };
}

/// # Notes on vector shifts.
///
/// Neon only has the `vector`x`vector` left shift function. However, it takes signed
/// arguments for the shift amount. Right shifts are achieved by using negative left-shifts.
///
/// To maintain consistency in `Wide`, we only allow positive left shifts and positive right
/// shifts.
///
/// * Left shifts: We need to clamp the shift amount between 0 and the maximum shift
///   (inclusive). This is done by first reinrepreting the shift vector as unsigned
///   (`cvtpre`), taking the unsigned `min` with the maximal shift, and then reinterpret
///   to signed (`cvtpost`).
///
///   If the shift vector is already unsigned, then `cvtpre` can be the identity.
///
/// * Right shifts: Right shifts follow the same logic as left shifts, just with a final
///   negative before invoking the left-shift intrinsic.
///
/// # Shifts by a Scalar
///
/// LLVM is not smart enough to constant propagate properly though a `splat` followed by
/// a vector shift if we do the range-limitation after the `splat`. So, the scalar shift
/// operations perform the "cast-to-positive + min + cast-to-signed" in the scalar space
/// before splatting. LLVM optimizes this correctly.
macro_rules! aarch64_define_bitops {
    ($type:ty,
     $not:ident,
     $and:ident,
     $or:ident,
     $xor:ident,
     ($shlv:ident, $mask:literal, $neg:ident, $min:ident, $cvtpost:path, $cvtpre:path),
     ($unsigned:ty, $signed:ty, $broadcast_signed:ident),
    ) => {
        impl std::ops::Not for $type {
            type Output = Self;
            #[inline(always)]
            fn not(self) -> Self {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self(unsafe { $not(self.0) })
            }
        }

        impl std::ops::BitAnd for $type {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self(unsafe { $and(self.0, rhs.0) })
            }
        }

        impl std::ops::BitOr for $type {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self(unsafe { $or(self.0, rhs.0) })
            }
        }

        impl std::ops::BitXor for $type {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                //
                // It is the caller's responsibility to instantiate the macro with an
                // intrinsics also gated by "neon".
                Self(unsafe { $xor(self.0, rhs.0) })
            }
        }

        ///////////////////
        // vector shifts //
        ///////////////////

        impl std::ops::Shr for $type {
            type Output = Self;
            #[inline(always)]
            fn shr(self, rhs: Self) -> Self {
                use $crate::AsSIMD;
                if cfg!(miri) {
                    self.emulated().shr(rhs.emulated()).as_simd(self.arch())
                } else {
                    // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                    //
                    // It is the caller's responsibility to instantiate the macro with an
                    // intrinsics also gated by "neon".
                    Self(unsafe {
                        $shlv(
                            self.0,
                            $neg($cvtpost($min(
                                $cvtpre(rhs.0),
                                $cvtpre(<$type as SIMDVector>::splat(self.arch(), $mask).0),
                            ))),
                        )
                    })
                }
            }
        }

        impl std::ops::Shl for $type {
            type Output = Self;
            #[inline(always)]
            fn shl(self, rhs: Self) -> Self {
                use $crate::AsSIMD;
                if cfg!(miri) {
                    self.emulated().shl(rhs.emulated()).as_simd(self.arch())
                } else {
                    // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                    //
                    // It is the caller's responsibility to instantiate the macro with an
                    // intrinsics also gated by "neon".
                    Self(unsafe {
                        $shlv(
                            self.0,
                            $cvtpost($min(
                                $cvtpre(rhs.0),
                                $cvtpre(<$type as SIMDVector>::splat(self.arch(), $mask).0),
                            )),
                        )
                    })
                }
            }
        }

        ///////////////////
        // scalar shifts //
        ///////////////////

        impl std::ops::Shr<<$type as SIMDVector>::Scalar> for $type {
            type Output = Self;
            #[inline(always)]
            fn shr(self, rhs: <$type as SIMDVector>::Scalar) -> Self {
                use $crate::AsSIMD;
                if cfg!(miri) {
                    self.emulated().shr(rhs).as_simd(self.arch())
                } else {
                    // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                    //
                    // It is the caller's responsibility to instantiate the macro with an
                    // intrinsics also gated by "neon".
                    Self(unsafe {
                        $shlv(
                            self.0,
                            $broadcast_signed(-((rhs as $unsigned).min($mask) as $signed)),
                        )
                    })
                }
            }
        }

        impl std::ops::Shl<<$type as SIMDVector>::Scalar> for $type {
            type Output = Self;
            #[inline(always)]
            fn shl(self, rhs: <$type as SIMDVector>::Scalar) -> Self {
                use $crate::AsSIMD;
                if cfg!(miri) {
                    self.emulated().shl(rhs).as_simd(self.arch())
                } else {
                    // SAFETY: Inclusion of this macro is gated by the "neon" feature.
                    //
                    // It is the caller's responsibility to instantiate the macro with an
                    // intrinsics also gated by "neon".
                    Self(unsafe {
                        $shlv(
                            self.0,
                            $broadcast_signed((rhs as $unsigned).min($mask) as $signed),
                        )
                    })
                }
            }
        }
    };
}

/// SAFETY: It is the invoker's responsibility to ensure that the provided intrinsics are
/// safe to call. T hat is - any intrinsics invoked must be compatible with `$type`'s
/// associated architecture.
macro_rules! aarch64_splitjoin {
    ($type:path, $half:path, $getlo:ident, $gethi:ident, $join:ident) => {
        impl $crate::SplitJoin for $type {
            type Halved = $half;

            #[inline(always)]
            fn split(self) -> $crate::LoHi<Self::Halved> {
                // SAFETY: This should only be instantiated for types where the associated
                // architecture provides a license to use it.
                unsafe {
                    $crate::LoHi::new(
                        Self::Halved::from_underlying(self.arch(), $getlo(self.to_underlying())),
                        Self::Halved::from_underlying(self.arch(), $gethi(self.to_underlying())),
                    )
                }
            }

            #[inline(always)]
            fn join(lohi: $crate::LoHi<Self::Halved>) -> Self {
                // SAFETY: This should only be instantiated for types where the associated
                // architecture provides a license to use it.
                unsafe {
                    Self::from_underlying(
                        lohi.lo.arch(),
                        $join(lohi.lo.to_underlying(), lohi.hi.to_underlying()),
                    )
                }
            }
        }
    };
}

pub(crate) use aarch64_define_bitops;
pub(crate) use aarch64_define_cmp;
pub(crate) use aarch64_define_fma;
pub(crate) use aarch64_define_loadstore;
pub(crate) use aarch64_define_register;
pub(crate) use aarch64_define_splat;
pub(crate) use aarch64_splitjoin;
