/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Wide - Cross Architecture SIMD
//!
//! This crate attempts to provide (mostly) Miri-compatible, cross-platform SIMD with support
//! for light-weight architecture dispatching.
//!
//! ## Traits
//!
//! SIMD vectors are weird types as they behave both like scalars and containers. Primary
//! traits exposed by `wide` are:
//!
//! * [`SIMDVector`]: General trait for working with a SIMD vector, including creation and
//!   data access.
//!
//! * [`SIMDMask`]: Basically a SIMD boolean. Comparisons between `SIMDVectors` are done
//!   lanewise, with the mask containing the results for each lane. Each [`SIMDVector`] has
//!   an associated mask.
//!
//! * [`Architecture`]: SIMD instructions are architecture specific. Some server CPUs like
//!   new(ish) x86 models support AVX512, while most consumer CPUs do not yet support that
//!   instruction set extension.
//!
//!   To allow compilation of single binaries that support multiple architectures, `wide` has
//!   taken the position that the [`Architecture`] is largely explicit when it comes to SIMD
//!   types.
//!
//!   Generic, cross-architecture algorithms are still supported by using an [`Architecture`]s
//!   associated SIMD types.
//!
//! A host of secondary SIMD related traits are also exported, all prefixed with `SIMD`.
//! Refer to the documentation on each trait for more information.
//!
//! ## Structs
//!
//! Types implementing [`SIMDMask`] can take a variety of architecture specific shapes.
//! To that end, each architecture-specific [`SIMDMask`] is associated with a [`BitMask`],
//! where bit `i` is set to 1 if the corresponding lane in the full mask representation
//! evaluates to a logic `true`, and `0` otherwise.
//!
//! Masks can be converted to and from their corresponding [`BitMask`] as needed.
//!
//! ## Safety
//!
//! One source of unsafety in SIMD is the accidental use of an intrinsic that is not supported
//! by the current runtime CPU. This is made safe in `wide` by using the following strategy:
//!
//! * Each [`SIMDVector`] and [`SIMDMask`] type is uniquely associated with an [`Architecture`].
//!
//! * Construction of a new [`SIMDVector`] or [`SIMDMask`] requires either an instance of its
//!   associated architecture, or a [`SIMDVector`]/[`SIMDMask`] of the same [`Architecture`].
//!
//! * [`Architecture`] instances can only be obtained:
//!
//!   - From an instance of a [`SIMDVector`]/[`SIMDMask`] associated with that [`Architecture`].
//!   - From one of the safe constructors like [`arch::dispatch`] or `new_checked` which
//!     perform runtime checks necessary to ensure the compatibility.
//!   - Through an `unsafe` constructor, on which case all bets are off.
//!
//! So an [`Architecture`] is needed to bootstrap the use of SIMD, but from then on, the
//! existence of SIMD types for a given [`Architecture`] serve as proof-of-safety.
//!
//! ## Special Architectures
//!
//! Some [`Architecture`]s are special and always available to use safely:
//!
//! * [`arch::Scalar`]: An architecture that uses emulation via loops to implement
//!   SIMD-like operations. This architecture is safe because no special hardware intrinsics
//!   are invoked.
//!
//! * [`arch::Current`]: The [`Architecture`] that is the closest fit to the current
//!   compilation target. This is not always [`arch::Scalar`]. For example, if compiling
//!   for `x86-64-v3`, then the [`arch::Current`] will be [`arch::x86_64::V3`]. This is
//!   safe because it only uses intrinsics that are already available for the compiler to use.
//!
//!   The current architecture can be obtained using with [`arch::current()`] or the
//!   constant [`crate::ARCH`].
//!
//! # Dev Docs
//!
//! ## Adding a new `TxN` vector type.
//!
//! 1. Implement the type for the backends in `arch` (you can usually follow and slightly
//!    modify the existing examples).
//!
//! 2. Implement for `Emulated` for the implementations that require macro instantiation.
//!
//! 3. Add the type to the [`Architecture`] trait.
//!
//! At each step, be sure to include tests, which should be fairly straight forward.
//!
//! ## Adding a New Implementation to an Existing Trait
//!
//! Basically do steps 2-4 of the above list.
//!
//! ## Adding a New Trait
//!
//! 1. If needed, provide a reference implementation in the `reference` module.
//!
//! 2. If it's a relatively simple op, adding a new macro in `test_utils/ops.rs` that
//!    invokes the reference implementation may be all that's needed.
//!
//!    More complicated operations may require their own test harness (see
//!    `test_tuils/dot_product.rs`).
//!
//!    Tests should go through the utilities in `test_utils::driver` to ensure adequate
//!    coverage and low compile time.
//!
//! 3. Implement the trait for the needed types, implementing for [`Emulated`],
//!    architecture-specific types, [`Architecture`].
//!
//! # Testing and Architectural Levels
//!
//! By default, `wide` will only run tests supported by the current runtime hardware. This
//! allows the tests to pass on a wide variety of machines during development.
//!
//! However, this can mean that tests targeting architecture not supported by the runtime
//! hardware will silently succeed.
//!
//! To ensure all tests either run, or generate an error if the runtime hardware does not
//! support a test, set the environment variable
//! ```text
//! WIDE_TEST_MIN_ARCH="all"
//! ```
//! Various back-end specific values are supported. Note that this variable sets the
//! minimum level of tests that are **required** to run. Tests for higher architecture
//! levels will still be run if supported by the runtime hardware.
//!
//! ## x86_64
//!
//! * `x86-64-v4`: Target Wide's [`arch::x86_64::V4`] architecture.
//! * `x86-64-v3`: Target Wide's [`arch::x86_64::V3`] architecture.
//! * `scalar`: Target the scalar architecture.

mod constant;
pub use constant::{Const, Constant, SupportedLaneCount};

pub(crate) mod reference;
pub use reference::{cast_f16_to_f32, cast_f32_to_f16};

mod traits;
pub use traits::{
    AsSIMD, SIMDAbs, SIMDCast, SIMDDotProduct, SIMDFloat, SIMDMask, SIMDMinMax, SIMDMulAdd,
    SIMDPartialEq, SIMDPartialOrd, SIMDReinterpret, SIMDSelect, SIMDSigned, SIMDSumTree,
    SIMDUnsigned, SIMDVector,
};

mod splitjoin;
pub use splitjoin::{LoHi, SplitJoin};

mod bitmask;
pub use bitmask::{BitMask, FromInt};

#[cfg(target_arch = "x86_64")]
pub(crate) mod doubled;

mod emulated;
pub use emulated::Emulated;

pub mod lifetime;

/////////////////////////////
// Architecture Resolution //
/////////////////////////////

pub mod arch;
pub use arch::Architecture;

/// The current architecture that is the closest fit for the current compilation target.
///
/// The type [`Wide`] is always configured to use this as its associated architecture type.
pub const ARCH: arch::Current = arch::current();

///////////////////////
// Alias Definitions //
///////////////////////

/// Convenience aliases for aliasing SIMD types.
///
/// There are currently four supported flavors (the examples below use `f32x4` as an example
/// identifier:
///
/// 1. `diskann_wide::alias!(f32x4) => type f32x4 = <diskann_wide::arch::Current as diskann_wide::Architecture>::f32x4`:
///    Type alias directly to the compile-time architecture's type.
///
/// 2. `diskann_wide::alias!(f32s = f32x4) => type f32s = <diskann_wide::arch::Current as
///    diskann_wide::Architecture>::f32x4`: Type alias a SIMD type with a custom name.
///
/// 3. `diskann_wide::alias!(f32s = <A>::f32x4) => type f32s = <A as diskann_wide::Architecture>::f32x4`:
///    Type alias a SIMD type from a specific architecture.
///
/// 4. `diskann_wide::alias!(f32s<A> = f32x4) => type f32s<A> = <A as diskann_wide::Architecture>::f32x4`:
///    Type alias a SIMD type in a generic context. This can be useful to work around errors
///    like
///    ```text
///    use of generic parameter from outer item
///    ```
#[macro_export]
macro_rules! alias {
    ($var:ident) => {
        $crate::alias!($var = $var);
    };
    ($var:ident = $type:ident) => {
        $crate::alias!($var = <diskann_wide::arch::Current>::$type);
    };
    ($var:ident = <$arch:ty>::$type:ident) => {
        #[allow(non_camel_case_types)]
        type $var = <$arch as $crate::Architecture>::$type;
    };
    ($var:ident<$arch:ident> = $type:ident) => {
        #[allow(non_camel_case_types)]
        type $var<$arch> = <$arch as $crate::Architecture>::$type;
    };
}

//////////////
// Internal //
//////////////

#[cfg(all(test, target_arch = "x86_64"))]
const TEST_MIN_ARCH: &str = "WIDE_TEST_MIN_ARCH";

#[cfg(all(test, target_arch = "x86_64"))]
fn get_test_arch() -> Option<String> {
    match std::env::var(TEST_MIN_ARCH) {
        Ok(v) => Some(v),
        Err(e) => match e {
            std::env::VarError::NotPresent => None,
            std::env::VarError::NotUnicode(s) => panic!("could not parse test arch: {s:?}"),
        },
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) mod helpers;

#[cfg(test)]
pub(crate) mod test_utils;

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn generic_architecture<A>(arch: A) -> f32
    where
        A: Architecture,
    {
        alias!(f32s<A> = f32x4);
        f32s::<A>::from_array(arch, [1.0, 2.0, 3.0, 4.0]).sum_tree()
    }

    #[test]
    fn test_generic() {
        assert_eq!(generic_architecture(arch::Scalar), 10.0);
    }
}
