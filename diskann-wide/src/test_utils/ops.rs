/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

// Common test traits.
use super::common::{self, ScalarTraits};
#[cfg(target_arch = "x86_64")]
use crate::SplitJoin;
use crate::{
    BitMask, Const, SIMDMask, SIMDMinMax, SIMDPartialEq, SIMDPartialOrd, SIMDSumTree, SIMDVector,
    SupportedLaneCount, arch,
    reference::{ReferenceScalarOps, ReferenceShifts, TreeReduce},
};

#[cfg(target_arch = "x86_64")]
fn identity<T>(x: T) -> T {
    x
}

pub(crate) fn test_splat<T, const N: usize, V>(arch: V::Arch)
where
    T: ScalarTraits,
    Const<N>: SupportedLaneCount,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>>,
{
    for i in T::splat_test_values() {
        assert!(
            V::splat(arch, i)
                .to_array()
                .into_iter()
                .all(|v| v.exact_eq(i))
        );
    }
}

/// Test that calling `op` on each input yields the exact same value as that in the
/// corresponding position in `got`.
#[inline(never)]
pub(crate) fn test_unary_op<T, U>(input: &[T], got: &[U], op: &dyn Fn(T) -> U, context: &str)
where
    T: Copy + Debug,
    U: Copy + Debug + ScalarTraits,
{
    let len = input.len();
    assert_eq!(len, got.len());
    for i in 0..len {
        let expected = op(input[i]);
        assert!(
            expected.exact_eq(got[i]),
            "failed for input {:?}. Got {:?}, expected {:?} at index {}. context: {}",
            input[i],
            got[i],
            expected,
            i,
            context
        )
    }
}

/// To support the fuzzy matching for NaNs and +/-0.0 in the min-max implementations, we need
/// to abstract the checking logic in `test_binary_op`.
pub(crate) trait CheckBinary<T, U> {
    fn check(&self, left: T, right: T, got: U) -> Result<(), String>;
}

/// Supplying a closure directly implies "exact_eq" semantics.
impl<T, U, F> CheckBinary<T, U> for F
where
    T: Copy + Debug,
    U: Copy + Debug + ScalarTraits,
    F: Fn(T, T) -> U,
{
    fn check(&self, left: T, right: T, got: U) -> Result<(), String> {
        let expected = (self)(left, right);
        if got.exact_eq(expected) {
            Ok(())
        } else {
            Err(format!("{:?}", expected))
        }
    }
}

fn check_minmax_non_standard_f32(
    left: f32,
    right: f32,
    got: f32,
    standard: f32,
) -> Result<(), String> {
    if got.exact_eq(standard) {
        return Ok(());
    }

    // If the result is 0, then we will accept either -0.0 or +0.0
    if standard == 0.0 && (got.exact_eq(0.0) || got.exact_eq(-0.0)) {
        return Ok(());
    }

    // We will accept either the IEEE value (which will return the non-NAN argument) or NaN.
    if (left.is_nan() || right.is_nan()) && got.is_nan() {
        return Ok(());
    }

    // Result was not a success. Return the expected value.
    let nan = "NaN_f32";
    let expected = if standard.is_nan() {
        // The standard implementation returns NaN only if both arguments are NaN.
        nan.to_string()
    } else if left.is_nan() {
        // Either the RHS or NaN is accepted.
        format!("{:?}/{}", right, nan)
    } else if right.is_nan() {
        // Either the LHS or NaN is accepted.
        format!("{:?}/{}", left, nan)
    } else {
        format!("{:?}", standard)
    };

    Err(expected)
}

fn check_minmax_standard_f32(got: f32, standard: f32) -> Result<(), String> {
    if got.exact_eq(standard) {
        return Ok(());
    }

    // If the result is 0, then we will accept either -0.0 or +0.0
    if standard == 0.0 && (got.exact_eq(0.0) || got.exact_eq(-0.0)) {
        return Ok(());
    }

    let expected = if standard == 0.0 {
        "0.0/-0.0".to_string()
    } else {
        standard.to_string()
    };
    Err(expected)
}

/// A checker for non-IEEE compliant max implementations.
///
/// The allowed non-compliance is strictly with the handling of NaNs.
///
/// IEEE specifies that `max` between a NaN and non-NaN value `x` should return `x`. However,
/// it is common for Intel hardware to return `NaN` instead for such operations.
///
/// The [`FastMax`] checker will accept either the IEEE result or `NaN` in such situations.
///
/// If the expected result is +/-0.0, than a zero of either sign will be accepted.
///
/// If neither argument is `NaN`, then the results must match the IEEE result.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FastMax;

impl CheckBinary<f32, f32> for FastMax {
    fn check(&self, left: f32, right: f32, got: f32) -> Result<(), String> {
        // `f32::max` is documented in the Rust standard to be IEEE compliant.
        check_minmax_non_standard_f32(left, right, got, left.max(right))
    }
}

/// A checker for Rust standard compliant max implementations.
///
/// If one argument is NaN, than the other is returned. A NaN is only returned if both
/// arguments are NaN.
///
/// If the expected result is +/-0.0, than a zero of either sign will be accepted.
///
/// If neither argument is `NaN`, then the results must match the IEEE result.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StandardMax;

impl CheckBinary<f32, f32> for StandardMax {
    fn check(&self, left: f32, right: f32, got: f32) -> Result<(), String> {
        check_minmax_standard_f32(got, left.max(right))
    }
}

/// A checker for non-IEEE compliant min implementations.
///
/// The allowed non-compliance is strictly with the handling of NaNs.
///
/// IEEE specifies that `min` between a NaN and non-NaN value `x` should return `x`. However,
/// it is common for Intel hardware to return `NaN` instead for such operations.
///
/// The [`FastMax`] checker will accept either the IEEE result or `NaN` in such situations.
///
/// If the expected result is +/-0.0, than a zero of either sign will be accepted.
///
/// If neither argument is `NaN`, then the results must match the IEEE result.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FastMin;

impl CheckBinary<f32, f32> for FastMin {
    fn check(&self, left: f32, right: f32, got: f32) -> Result<(), String> {
        // `f32::min` is documented in the Rust standard to be IEEE compliant.
        check_minmax_non_standard_f32(left, right, got, left.min(right))
    }
}

/// A checker for Rust standard compliant min implementations.
///
/// If one argument is NaN, than the other is returned. A NaN is only returned if both
/// arguments are NaN.
///
/// If the expected result is +/-0.0, than a zero of either sign will be accepted.
///
/// If neither argument is `NaN`, then the results must match the IEEE result.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StandardMin;

impl CheckBinary<f32, f32> for StandardMin {
    fn check(&self, left: f32, right: f32, got: f32) -> Result<(), String> {
        check_minmax_standard_f32(got, left.min(right))
    }
}

macro_rules! exact_minmax {
    ($T:ty) => {
        impl CheckBinary<$T, $T> for StandardMax {
            fn check(&self, left: $T, right: $T, got: $T) -> Result<(), String> {
                let expected = left.expected_max_(right);
                if expected.exact_eq(got) {
                    Ok(())
                } else {
                    Err(expected.to_string())
                }
            }
        }

        impl CheckBinary<$T, $T> for FastMax {
            fn check(&self, left: $T, right: $T, got: $T) -> Result<(), String> {
                StandardMax.check(left, right, got)
            }
        }

        impl CheckBinary<$T, $T> for StandardMin {
            fn check(&self, left: $T, right: $T, got: $T) -> Result<(), String> {
                let expected = left.expected_min_(right);
                if expected.exact_eq(got) {
                    Ok(())
                } else {
                    Err(expected.to_string())
                }
            }
        }

        impl CheckBinary<$T, $T> for FastMin {
            fn check(&self, left: $T, right: $T, got: $T) -> Result<(), String> {
                StandardMin.check(left, right, got)
            }
        }

    };
    ($($T:ty),+ $(,)?) => {
        $(exact_minmax!($T);)+
    }
}

exact_minmax!(u8, u16, u32, u64, i8, i16, i32, i64);

// Call `op` on each pair of elements in `left` and `right` and check that the value is
// exactly equal to the corresponding entry in `got`.
#[inline(never)]
pub(crate) fn test_binary_op<T, U>(
    left: &[T],
    right: &[T],
    got: &[U],
    op: &dyn CheckBinary<T, U>,
    context: &str,
) where
    T: Copy + Debug,
    U: ScalarTraits,
{
    let len = left.len();
    assert_eq!(len, right.len());
    assert_eq!(len, got.len());
    for i in 0..len {
        if let Err(expected) = op.check(left[i], right[i], got[i]) {
            panic!(
                "failed for op({:?}, {:?}). Got {:?} but expected {} at index {}. context: {}",
                left[i], right[i], got[i], expected, i, context
            )
        }
    }
}

#[inline(never)]
pub(crate) fn test_trinary_op<T, U>(
    a: &[T],
    b: &[T],
    c: &[U],
    got: &[T],
    op: &dyn Fn(T, T, U) -> T,
    context: &str,
) where
    T: Copy + Debug + ScalarTraits,
    U: Copy + Debug + ScalarTraits,
{
    let len = a.len();
    assert_eq!(len, b.len());
    assert_eq!(len, c.len());
    assert_eq!(len, got.len());
    for i in 0..len {
        let expected = op(a[i], b[i], c[i]);
        assert!(
            expected.exact_eq(got[i]),
            "failed for op({:?}, {:?}, {:?}). Got {:?} but expected {:?} at index{}. Context: {}",
            a[i],
            b[i],
            c[i],
            got[i],
            expected,
            i,
            context,
        );
    }
}

/// This invocation defines a test with a unique name for the given wide type.
///
/// Centralizing this logic in one place lets us have uniform tests for all different wide
/// types.
///
/// !!! NOTE
///
///    These macros are very finicky.
///
///    If you change the body of the macro, make sure to test that errors still get thrown
///    if the implementation is incorrect.
///
///    As an example, you can change the `+` in `test_add` to `-` and run the test suite.
///
///    You should see failures across the board.
macro_rules! test_add {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<add_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;
                type T = $wide $(< $($ps),+>)?;
                type Scalar = <T as SIMDVector>::Scalar;
                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar], b: &[Scalar] | {
                        let got = (
                            <T>::from_array(arch, a.try_into().unwrap()) +
                            <T>::from_array(arch, b.try_into().unwrap())
                        ).to_array();
                        test_utils::test_binary_op(
                            &a,
                            &b,
                            &got,
                            &|l: Scalar, r: Scalar| { l.expected_add_(r) },
                            "binary addition",
                        )
                    };
                    let n: usize = <T as SIMDVector>::LANES;
                    crate::test_utils::driver::drive_binary(&f, (n, n), $seed);
                }
            }
        }
    }
}

macro_rules! test_sub {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<sub_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;
                type T = $wide $(< $($ps),+>)?;
                type Scalar = <T as SIMDVector>::Scalar;

                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar], b: &[Scalar]| {
                        let got = (
                            <T>::from_array(arch, a.try_into().unwrap()) -
                            <T>::from_array(arch, b.try_into().unwrap())
                        ).to_array();
                        test_utils::test_binary_op(
                            &a,
                            &b,
                            &got,
                            &|l: Scalar, r: Scalar| { l.expected_sub_(r) },
                            "binary subtraction",
                        )
                    };

                    let n: usize = T::LANES;
                    $crate::test_utils::driver::drive_binary(&f, (n, n), $seed);
                }
            }
        }
    };
}

macro_rules! test_mul {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<mul_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;
                type T = $wide $(< $($ps),+>)?;
                type Scalar = <T as SIMDVector>::Scalar;

                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar], b: &[Scalar]| {
                        let got = (
                            <T>::from_array(arch, a.try_into().unwrap()) *
                            <T>::from_array(arch, b.try_into().unwrap())
                        ).to_array();
                        test_utils::test_binary_op(
                            &a,
                            &b,
                            &got,
                            &|l: Scalar, r: Scalar| { l.expected_mul_(r) },
                            "binary multiplication",
                        )
                    };

                    let n = T::LANES;
                    $crate::test_utils::driver::drive_binary(&f, (n, n), $seed);
                }
            }
        }
    };
}

macro_rules! test_fma {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<fma_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::{SIMDVector, SIMDMulAdd};
                type T = $wide $(< $($ps),+>)?;
                type Scalar = <T as SIMDVector>::Scalar;

                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar], b: &[Scalar], c: &[Scalar]| {
                        let got = <T>::from_array(
                            arch,
                            a.try_into().unwrap()
                        ).mul_add_simd(
                            <T>::from_array(arch, b.try_into().unwrap()),
                            <T>::from_array(arch, c.try_into().unwrap()),
                        ).to_array();

                        test_utils::test_trinary_op(
                            &a,
                            &b,
                            &c,
                            &got,
                            &|x, y, z| { x.expected_fma_(y, z) },
                            "fuseed multiply-add",
                        )
                    };

                    let n: usize = T::LANES;
                    $crate::test_utils::driver::drive_ternary(&f, (n, n, n), $seed);
                }
            }
        }
    };
}

macro_rules! test_lossless_convert {
    (
        $from:ident $(< $($fs:tt),+ >)? => $to:ident $(< $($ts:tt),+ >)?,
        $seed:literal,
        $arch:expr
    ) => {
        paste::paste! {
            #[test]
            fn [<convert_ $from:lower $(_$($fs )x+)? _to_ $to:lower $(_$($ts )x+)?>]() {
                use $crate::SIMDVector;
                type From = $from $(< $($fs),+>)?;
                type To = $to $(< $($ts),+>)?;

                if let Some(arch) = $arch {
                    let f = move |input: &[<From as SIMDVector>::Scalar]| {
                        let got: To = <From>::from_array(arch, input.try_into().unwrap()).into();

                        test_utils::test_unary_op(
                            &input,
                            &(got.to_array()),
                            &|x| { x.into() },
                            "conversion",
                        )
                    };

                    let n = To::LANES;
                    crate::test_utils::driver::drive_unary(&f, n, $seed);
                }
            }
        }
    }
}

macro_rules! test_cast {
    (
        $from:ident $(< $($fs:tt),+ >)? => $to:ident $(< $($ts:tt),+ >)?,
        $seed:literal,
        $arch:expr
    ) => {
        paste::paste! {
            #[test]
            fn [<cast_ $from:lower $(_$($fs )x+)? _to_ $to:lower $(_$($ts )x+)?>]() {
                use $crate::{SIMDVector, reference::ReferenceCast};

                type From = $from $(< $($fs),+>)?;
                type To = $to $(< $($ts),+>)?;

                if let Some(arch) = $arch {
                    let f = move |input: &[<From as SIMDVector>::Scalar]| {
                        let got: To = <From>::from_array(
                            arch,
                            input.try_into().unwrap()
                        ).cast::<<To as SIMDVector>::Scalar>();

                        test_utils::test_unary_op(
                            &input,
                            &(got.to_array()),
                            &|x| { x.reference_cast() },
                            "cast",
                        )
                    };

                    let n = To::LANES;
                    crate::test_utils::driver::drive_unary(&f, n, $seed);
                }
            }
        }
    }
}

macro_rules! test_abs {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<abs_ $wide:lower $(_$($ps )x+)?>]() {
                use crate::{SIMDVector, SIMDAbs, reference::ReferenceAbs};

                type T = $wide $(< $($ps),+>)?;

                if let Some(arch) = $arch {
                    let f = move |input: &[<T as SIMDVector>::Scalar]| {
                        let got = <T>::from_array(
                            arch,
                            input.try_into().unwrap()
                        ).abs_simd().to_array();

                        $crate::test_utils::test_unary_op(
                            &input,
                            &got,
                            &|x| { x.expected_abs_() },
                            "absolute value",
                        )
                    };
                    let n: usize = T::LANES;
                    $crate::test_utils::driver::drive_unary(&f, n, $seed);
                }
            }
        }
    };
}

macro_rules! test_select {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<select_ $wide:lower $(_$($ps )x+)?>]() {
                use crate::{SIMDVector, SIMDMask, SIMDSelect};

                type T = $wide $(< $($ps),+>)?;
                type Mask = <T as SIMDVector>::Mask;

                if let Some(arch) = $arch {
                    let f = move |
                        x: &[<T as SIMDVector>::Scalar],
                        y: &[<T as SIMDVector>::Scalar],
                        mask: &[bool],
                    | {
                        let simd_mask = Mask::from_fn(arch, |i| mask[i]);
                        let simd_x = <T>::from_array(arch, x.try_into().unwrap());
                        let simd_y = <T>::from_array(arch, y.try_into().unwrap());

                        let got = simd_mask.select(simd_x, simd_y);
                        $crate::test_utils::test_trinary_op(
                            &x,
                            &y,
                            &mask,
                            &got.to_array(),
                            &|x, y, b| { if b {x} else {y} },
                            "select",
                        )
                    };
                    let n: usize = T::LANES;
                    $crate::test_utils::driver::drive_ternary(&f, (n, n, n), $seed);
                }
            }
        }
    };
}

////////////
// MinMax //
////////////

pub(crate) fn test_minmax_impl<V, const N: usize, A>(arch: A, a: &[V::Scalar], b: &[V::Scalar])
where
    A: arch::Sealed,
    Const<N>: SupportedLaneCount,
    V: SIMDVector<Arch = A, ConstLanes = Const<N>> + SIMDMinMax,
    V::Scalar: ScalarTraits + ReferenceScalarOps,
    StandardMax: CheckBinary<V::Scalar, V::Scalar>,
    StandardMin: CheckBinary<V::Scalar, V::Scalar>,
    FastMax: CheckBinary<V::Scalar, V::Scalar>,
    FastMin: CheckBinary<V::Scalar, V::Scalar>,
{
    let a: &[V::Scalar; N] = a.try_into().unwrap();
    let b: &[V::Scalar; N] = b.try_into().unwrap();

    let wa = V::from_array(arch, *a);
    let wb = V::from_array(arch, *b);

    // Fast Max
    let got = wa.max_simd(wb).to_array();
    test_binary_op(a, b, &got, &FastMax, "fast_max");

    // Fast Min
    let got = wa.min_simd(wb).to_array();
    test_binary_op(a, b, &got, &FastMin, "fast_min");

    // IEEE Max
    let got = wa.max_simd_standard(wb).to_array();
    test_binary_op(a, b, &got, &StandardMax, "standard_max");

    // IEEE Min
    let got = wa.min_simd_standard(wb).to_array();
    test_binary_op(a, b, &got, &StandardMin, "standard_min");
}

macro_rules! test_minmax {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<minmax_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;

                type T = $wide $(< $($ps),+>)?;
                type Scalar = <T as $crate::SIMDVector>::Scalar;

                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar], b: &[Scalar]| {
                        $crate::test_utils::ops::test_minmax_impl::<
                            T,
                            { T::LANES },
                            _,
                        >(
                            arch,
                            a,
                            b,
                        );
                    };
                    let n: usize = T::LANES;
                    $crate::test_utils::driver::drive_binary(&f, (n, n), $seed);
                }
            }
        }
    };
}

//////////////////////////////
// PartialOrd and PartialEq //
//////////////////////////////

pub(crate) fn test_cmp_impl<V, T, const N: usize, A>(arch: A, a: &[T], b: &[T])
where
    A: arch::Sealed,
    T: ScalarTraits + std::fmt::Display + PartialOrd + PartialEq,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    V: SIMDVector<Arch = A, Scalar = T, ConstLanes = Const<N>> + SIMDPartialOrd + SIMDPartialEq,
    V::Mask: Into<BitMask<N, A>>,
{
    let a: &[T; N] = a.try_into().unwrap();
    let b: &[T; N] = b.try_into().unwrap();

    let wa = V::from_array(arch, *a);
    let wb = V::from_array(arch, *b);

    // equal
    let got = common::promote_to_array(wa.eq_simd(wb).into());
    test_binary_op(a, b, &got, &|l, r| l == r, "eq_simd");

    // not-equal
    let got = common::promote_to_array(wa.ne_simd(wb).into());
    test_binary_op(a, b, &got, &|l, r| l != r, "ne_simd");

    // less-than
    let got = common::promote_to_array(wa.lt_simd(wb).into());
    test_binary_op(a, b, &got, &|l, r| l < r, "lt_simd");

    // less-than-or-equal
    let got = common::promote_to_array(wa.le_simd(wb).into());
    test_binary_op(a, b, &got, &|l, r| l <= r, "le_simd");

    // greater-than
    let got = common::promote_to_array(wa.gt_simd(wb).into());
    test_binary_op(a, b, &got, &|l, r| l > r, "gt_simd");

    // greater-than-or-equal
    let got = common::promote_to_array(wa.ge_simd(wb).into());
    test_binary_op(a, b, &got, &|l, r| l >= r, "ge_simd");
}

macro_rules! test_cmp {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<partial_ord_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;

                type T = $wide $(< $($ps),+>)?;

                if let Some(arch) = $arch {
                    let f = move |
                        a: &[<T as SIMDVector>::Scalar],
                        b: &[<T as SIMDVector>::Scalar]
                    | {
                        test_utils::ops::test_cmp_impl::<
                            T,
                            <T as SIMDVector>::Scalar,
                            { <T as SIMDVector>::LANES },
                            _
                        >(arch, a, b)
                    };

                    let n = <T>::LANES;
                    $crate::test_utils::driver::drive_binary(
                        &f,
                        (n, n),
                        $seed,
                    );
                }
            }
        }
    }
}

////////////
// BitOps //
////////////

pub(crate) trait BitOps:
    Sized
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Not<Output = Self>
    + std::ops::Shr<Output = Self>
    + std::ops::Shl<Output = Self>
{
}

impl<T> BitOps for T where
    T: Sized
        + std::ops::BitAnd<Output = Self>
        + std::ops::BitOr<Output = Self>
        + std::ops::BitXor<Output = Self>
        + std::ops::Not<Output = Self>
        + std::ops::Shr<Output = Self>
        + std::ops::Shl<Output = Self>
{
}

pub(crate) fn test_bitops_impl<V, T, const N: usize, A>(arch: A, a: &[T], b: &[T])
where
    A: arch::Sealed,
    T: BitOps + Debug + Copy + Eq + ReferenceShifts + ScalarTraits,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    V: SIMDVector<Arch = A, Scalar = T, ConstLanes = Const<N>> + BitOps,
    V::Mask: Into<BitMask<N, A>>,
{
    let a: &[T; N] = a.try_into().unwrap();
    let b: &[T; N] = b.try_into().unwrap();

    let wa = V::from_array(arch, *a);
    let wb = V::from_array(arch, *b);

    // bit-and
    let got = (wa & wb).to_array();
    test_binary_op(a, b, &got, &|l, r| l & r, "bitand");

    // bit-or
    let got = (wa | wb).to_array();
    test_binary_op(a, b, &got, &|l, r| l | r, "bitor");

    // bit-xor
    let got = (wa ^ wb).to_array();
    test_binary_op(a, b, &got, &|l, r| l ^ r, "bitxor");

    // not
    let got = (!wa).to_array();
    test_unary_op(a, &got, &|l| !l, "not");

    // shr
    let got = (wa >> wb).to_array();
    test_binary_op(a, b, &got, &|l: T, r: T| l.expected_shr_(r), "shr");

    // shl
    let got = (wa << wb).to_array();
    test_binary_op(a, b, &got, &|l: T, r: T| l.expected_shl_(r), "shl");
}

pub(crate) fn test_scalar_shift_impl<V, T, const N: usize, A>(arch: A, a: &[T], b: &[T])
where
    A: arch::Sealed,
    T: BitOps + Debug + Copy + Eq + ReferenceShifts + ScalarTraits,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    V: SIMDVector<Arch = A, Scalar = T, ConstLanes = Const<N>>
        + std::ops::Shl<T, Output = V>
        + std::ops::Shr<T, Output = V>,
    V::Mask: Into<BitMask<N, A>>,
{
    let a: &[T; N] = a.try_into().unwrap();
    assert_eq!(b.len(), 1);
    let b = b[0];

    let wa = V::from_array(arch, *a);
    let vb: [T; N] = [b; N];

    // shr
    let got = (wa >> b).to_array();
    test_binary_op(a, &vb, &got, &|l: T, r: T| l.expected_shr_(r), "shr");

    // shl
    let got = (wa << b).to_array();
    test_binary_op(a, &vb, &got, &|l: T, r: T| l.expected_shl_(r), "shl");
}

macro_rules! test_bitops {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<test_bitops_ $wide:lower $(_$($ps )x+)?>]() {
                use crate::SIMDVector;
                type T = $wide $(< $($ps),+>)?;
                if let Some(arch) = $arch {
                    let f = move |
                        a: &[<T as SIMDVector>::Scalar],
                        b: &[<T as SIMDVector>::Scalar]
                    | {
                        test_utils::ops::test_bitops_impl::<
                            T,
                            <T as SIMDVector>::Scalar,
                            { <T>::LANES },
                            _
                        >(arch, a, b)
                    };
                    let n = <T>::LANES;
                    $crate::test_utils::driver::drive_binary(
                        &f,
                        (n, n),
                        $seed,
                    );
                }
            }

            #[test]
            fn [<test_shifts_ $wide:lower $(_$($ps )x+)?>]() {
                use crate::SIMDVector;
                type T = $wide $(< $($ps),+>)?;
                if let Some(arch) = $arch {
                    let f = move |
                        a: &[<T as SIMDVector>::Scalar],
                        b: &[<T as SIMDVector>::Scalar]
                    | {
                        test_utils::ops::test_scalar_shift_impl::<
                            T,
                            <T as SIMDVector>::Scalar,
                            { <T>::LANES },
                            _
                        >(arch, a, b)
                    };
                    let n = <T>::LANES;
                    $crate::test_utils::driver::drive_binary(
                        &f,
                        (n, 1),
                        $seed,
                    );
                }
            }
        }
    }
}

pub fn test_sumtree_impl<V, T, const N: usize>(arch: V::Arch, a: &[T])
where
    T: Copy + Debug + ScalarTraits + ReferenceScalarOps,
    [T; N]: TreeReduce<Scalar = T>,
    Const<N>: SupportedLaneCount,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>> + SIMDSumTree,
{
    let a: &[T; N] = a.try_into().unwrap();
    let got = V::from_array(arch, *a).sum_tree();
    let expected = a.tree_reduce(|i, j| i.expected_add_(j));
    assert!(
        got.exact_eq(expected),
        "failed for {:?}, got {:?}, expected {:?}",
        a,
        got,
        expected
    );
}

macro_rules! test_sumtree {
    ($wide:ident $(< $($ps:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<sumtree_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;

                type T = $wide $(< $($ps),+>)?;
                type Scalar = <T as SIMDVector>::Scalar;

                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar]| {
                        $crate::test_utils::ops::test_sumtree_impl::<
                            T,
                            Scalar,
                            { <T>::LANES },
                        >(arch, a)
                    };

                    let n = <T>::LANES;
                    $crate::test_utils::driver::drive_unary(&f, n, $seed)
                }
            }
        }
    }
}

///////////////
// SplitJoin //
///////////////
#[cfg(target_arch = "x86_64")]
pub fn test_splitjoin_impl<V, H, T, const N: usize, const N2: usize>(arch: V::Arch, a: &[T])
where
    T: Copy + Debug + ScalarTraits,
    [T; N]: SplitJoin<Halved = [T; N2]>,
    Const<N>: SupportedLaneCount,
    Const<N2>: SupportedLaneCount,
    V: SIMDVector<Scalar = T, ConstLanes = Const<N>> + SplitJoin<Halved = H>,
    H: SIMDVector<Scalar = T, ConstLanes = Const<N2>>,
{
    use crate::{LoHi, SplitJoin};

    assert!(2 * N2 == N, "split/join should logically halve dimensions");
    let a: &[T; N] = a.try_into().unwrap();

    let LoHi {
        lo: lo_expected,
        hi: hi_expected,
    } = a.split();
    let LoHi { lo, hi } = V::from_array(arch, *a).split();

    test_unary_op(&lo.to_array(), &lo_expected, &identity, "split low");
    test_unary_op(&hi.to_array(), &hi_expected, &identity, "split high");

    let joined: V = LoHi::new(lo, hi).join();
    test_unary_op(&joined.to_array(), a, &identity, "join");
}

#[cfg(target_arch = "x86_64")]
macro_rules! test_splitjoin {
    ($wide:ident $(< $($ps:tt),+ >)? => $half:ident $(< $($hs:tt),+ >)?, $seed:literal, $arch:expr) => {
        paste::paste! {
            #[test]
            fn [<splitjoin_ $wide:lower $(_$($ps )x+)?>]() {
                use $crate::SIMDVector;
                type Wide = $wide $(< $($ps),+>)?;
                type Half = $half $(< $($hs),+>)?;

                type Scalar = <Wide as SIMDVector>::Scalar;

                if let Some(arch) = $arch {
                    let f = move |a: &[Scalar]| {
                        $crate::test_utils::ops::test_splitjoin_impl::<
                            Wide,
                            Half,
                            Scalar,
                            { <Wide>::LANES },
                            { <Half>::LANES },
                        >(arch, a)
                    };

                    let n = <Wide>::LANES;
                    $crate::test_utils::driver::drive_unary(&f, n, $seed)
                }
            }
        }
    }
}

///////////////////
// Macro Exports //
///////////////////

pub(crate) use test_abs;
pub(crate) use test_add;
pub(crate) use test_bitops;
pub(crate) use test_cast;
pub(crate) use test_cmp;
pub(crate) use test_fma;
pub(crate) use test_lossless_convert;
pub(crate) use test_minmax;
pub(crate) use test_mul;
pub(crate) use test_select;
#[cfg(target_arch = "x86_64")]
pub(crate) use test_splitjoin;
pub(crate) use test_sub;
pub(crate) use test_sumtree;

///////////
// Tests //
///////////

#[test]
fn test_fast_max_f32() {
    let zero = 0.0f32;
    let neg_zero = -0.0f32;
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    assert!(FastMax.check(1.0, 2.0, 2.0).is_ok());
    assert!(FastMax.check(1.0, 2.0, 1.0).is_err());

    assert!(FastMax.check(1.0, inf, inf).is_ok());
    assert!(FastMax.check(1.0, inf, f32::MAX).is_err());

    assert!(FastMax.check(inf, 1.0, inf).is_ok());
    assert!(FastMax.check(inf, 1.0, f32::MAX).is_err());

    assert!(FastMax.check(inf, neg_inf, inf).is_ok());
    assert!(FastMax.check(inf, neg_inf, neg_inf).is_err());

    // Signed zeros.
    assert!(FastMax.check(zero, zero, zero).is_ok());
    assert!(FastMax.check(zero, zero, neg_zero).is_ok());
    assert!(FastMax.check(zero, neg_zero, zero).is_ok());
    assert!(FastMax.check(zero, neg_zero, neg_zero).is_ok());

    assert!(FastMax.check(neg_zero, zero, zero).is_ok());
    assert!(FastMax.check(neg_zero, zero, neg_zero).is_ok());
    assert!(FastMax.check(neg_zero, neg_zero, zero).is_ok());
    assert!(FastMax.check(neg_zero, neg_zero, neg_zero).is_ok());

    assert!(FastMax.check(-1.0f32, neg_zero, neg_zero).is_ok());
    assert!(FastMax.check(-1.0f32, neg_zero, zero).is_ok());
    assert!(FastMax.check(-1.0f32, neg_zero, 1.0).is_err());

    assert!(FastMax.check(neg_zero, -1.0f32, neg_zero).is_ok());
    assert!(FastMax.check(neg_zero, -1.0f32, zero).is_ok());
    assert!(FastMax.check(neg_zero, -1.0f32, 1.0).is_err());

    // NaN handling.
    assert!(FastMax.check(zero, nan, zero).is_ok());
    assert!(FastMax.check(zero, nan, neg_zero).is_ok());
    assert!(FastMax.check(zero, nan, nan).is_ok());
    assert!(FastMax.check(zero, nan, 1.0).is_err());

    assert!(FastMax.check(neg_zero, nan, zero).is_ok());
    assert!(FastMax.check(neg_zero, nan, neg_zero).is_ok());
    assert!(FastMax.check(neg_zero, nan, nan).is_ok());
    assert!(FastMax.check(neg_zero, nan, 1.0).is_err());

    assert!(FastMax.check(nan, zero, zero).is_ok());
    assert!(FastMax.check(nan, zero, neg_zero).is_ok());
    assert!(FastMax.check(nan, zero, nan).is_ok());
    assert!(FastMax.check(nan, zero, 1.0).is_err());

    assert!(FastMax.check(nan, neg_zero, zero).is_ok());
    assert!(FastMax.check(nan, neg_zero, neg_zero).is_ok());
    assert!(FastMax.check(nan, neg_zero, nan).is_ok());
    assert!(FastMax.check(nan, neg_zero, 1.0).is_err());
}

#[test]
fn test_standard_max_f32() {
    let zero = 0.0f32;
    let neg_zero = -0.0f32;
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    assert!(StandardMax.check(1.0, 2.0, 2.0).is_ok());
    assert!(StandardMax.check(1.0, 2.0, 1.0).is_err());

    assert!(StandardMax.check(1.0, inf, inf).is_ok());
    assert!(StandardMax.check(1.0, inf, f32::MAX).is_err());

    assert!(StandardMax.check(inf, 1.0, inf).is_ok());
    assert!(StandardMax.check(inf, 1.0, f32::MAX).is_err());

    assert!(StandardMax.check(inf, neg_inf, inf).is_ok());
    assert!(StandardMax.check(inf, neg_inf, neg_inf).is_err());

    // Signed zeros.
    assert!(StandardMax.check(zero, zero, zero).is_ok());
    assert!(StandardMax.check(zero, zero, neg_zero).is_ok());
    assert!(StandardMax.check(zero, neg_zero, zero).is_ok());
    assert!(StandardMax.check(zero, neg_zero, neg_zero).is_ok());

    assert!(StandardMax.check(neg_zero, zero, zero).is_ok());
    assert!(StandardMax.check(neg_zero, zero, neg_zero).is_ok());
    assert!(StandardMax.check(neg_zero, neg_zero, zero).is_ok());
    assert!(StandardMax.check(neg_zero, neg_zero, neg_zero).is_ok());

    assert!(StandardMax.check(-1.0f32, neg_zero, neg_zero).is_ok());
    assert!(StandardMax.check(-1.0f32, neg_zero, zero).is_ok());
    assert!(StandardMax.check(-1.0f32, neg_zero, 1.0).is_err());

    assert!(StandardMax.check(neg_zero, -1.0f32, neg_zero).is_ok());
    assert!(StandardMax.check(neg_zero, -1.0f32, zero).is_ok());
    assert!(StandardMax.check(neg_zero, -1.0f32, 1.0).is_err());

    // NaN handling.
    assert!(StandardMax.check(zero, nan, zero).is_ok());
    assert!(StandardMax.check(zero, nan, neg_zero).is_ok());
    assert!(StandardMax.check(zero, nan, nan).is_err());
    assert!(StandardMax.check(zero, nan, 1.0).is_err());

    assert!(StandardMax.check(neg_zero, nan, zero).is_ok());
    assert!(StandardMax.check(neg_zero, nan, neg_zero).is_ok());
    assert!(StandardMax.check(neg_zero, nan, nan).is_err());
    assert!(StandardMax.check(neg_zero, nan, 1.0).is_err());

    assert!(StandardMax.check(nan, zero, zero).is_ok());
    assert!(StandardMax.check(nan, zero, neg_zero).is_ok());
    assert!(StandardMax.check(nan, zero, nan).is_err());
    assert!(StandardMax.check(nan, zero, 1.0).is_err());

    assert!(StandardMax.check(nan, neg_zero, zero).is_ok());
    assert!(StandardMax.check(nan, neg_zero, neg_zero).is_ok());
    assert!(StandardMax.check(nan, neg_zero, nan).is_err());
    assert!(StandardMax.check(nan, neg_zero, 1.0).is_err());
}

#[test]
fn test_fast_min_f32() {
    let zero = 0.0f32;
    let neg_zero = -0.0f32;
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    assert!(FastMin.check(1.0, 2.0, 1.0).is_ok());
    assert!(FastMin.check(1.0, 2.0, 2.0).is_err());

    assert!(FastMin.check(1.0, neg_inf, neg_inf).is_ok());
    assert!(FastMin.check(1.0, neg_inf, f32::MIN).is_err());

    assert!(FastMin.check(neg_inf, 1.0, neg_inf).is_ok());
    assert!(FastMin.check(neg_inf, 1.0, f32::MIN).is_err());

    assert!(FastMin.check(inf, neg_inf, neg_inf).is_ok());
    assert!(FastMin.check(inf, neg_inf, inf).is_err());

    // Signed zeros.
    assert!(FastMin.check(zero, zero, zero).is_ok());
    assert!(FastMin.check(zero, zero, neg_zero).is_ok());
    assert!(FastMin.check(zero, neg_zero, zero).is_ok());
    assert!(FastMin.check(zero, neg_zero, neg_zero).is_ok());

    assert!(FastMin.check(neg_zero, zero, zero).is_ok());
    assert!(FastMin.check(neg_zero, zero, neg_zero).is_ok());
    assert!(FastMin.check(neg_zero, neg_zero, zero).is_ok());
    assert!(FastMin.check(neg_zero, neg_zero, neg_zero).is_ok());

    assert!(FastMin.check(1.0f32, neg_zero, neg_zero).is_ok());
    assert!(FastMin.check(1.0f32, neg_zero, zero).is_ok());
    assert!(FastMin.check(1.0f32, neg_zero, 1.0).is_err());

    assert!(FastMin.check(neg_zero, 1.0f32, neg_zero).is_ok());
    assert!(FastMin.check(neg_zero, 1.0f32, zero).is_ok());
    assert!(FastMin.check(neg_zero, 1.0f32, 1.0).is_err());

    // NaN handling.
    assert!(FastMin.check(zero, nan, zero).is_ok());
    assert!(FastMin.check(zero, nan, neg_zero).is_ok());
    assert!(FastMin.check(zero, nan, nan).is_ok());
    assert!(FastMin.check(zero, nan, 1.0).is_err());

    assert!(FastMin.check(neg_zero, nan, zero).is_ok());
    assert!(FastMin.check(neg_zero, nan, neg_zero).is_ok());
    assert!(FastMin.check(neg_zero, nan, nan).is_ok());
    assert!(FastMin.check(neg_zero, nan, 1.0).is_err());

    assert!(FastMin.check(nan, zero, zero).is_ok());
    assert!(FastMin.check(nan, zero, neg_zero).is_ok());
    assert!(FastMin.check(nan, zero, nan).is_ok());
    assert!(FastMin.check(nan, zero, 1.0).is_err());

    assert!(FastMin.check(nan, neg_zero, zero).is_ok());
    assert!(FastMin.check(nan, neg_zero, neg_zero).is_ok());
    assert!(FastMin.check(nan, neg_zero, nan).is_ok());
    assert!(FastMin.check(nan, neg_zero, 1.0).is_err());
}

#[test]
fn test_standard_min_f32() {
    let zero = 0.0f32;
    let neg_zero = -0.0f32;
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let neg_inf = f32::NEG_INFINITY;

    assert!(StandardMin.check(1.0, 2.0, 1.0).is_ok());
    assert!(StandardMin.check(1.0, 2.0, 2.0).is_err());

    assert!(StandardMin.check(1.0, neg_inf, neg_inf).is_ok());
    assert!(StandardMin.check(1.0, neg_inf, f32::MIN).is_err());

    assert!(StandardMin.check(neg_inf, 1.0, neg_inf).is_ok());
    assert!(StandardMin.check(neg_inf, 1.0, f32::MIN).is_err());

    assert!(StandardMin.check(inf, neg_inf, neg_inf).is_ok());
    assert!(StandardMin.check(inf, neg_inf, inf).is_err());

    // Signed zeros.
    assert!(StandardMin.check(zero, zero, zero).is_ok());
    assert!(StandardMin.check(zero, zero, neg_zero).is_ok());
    assert!(StandardMin.check(zero, neg_zero, zero).is_ok());
    assert!(StandardMin.check(zero, neg_zero, neg_zero).is_ok());

    assert!(StandardMin.check(neg_zero, zero, zero).is_ok());
    assert!(StandardMin.check(neg_zero, zero, neg_zero).is_ok());
    assert!(StandardMin.check(neg_zero, neg_zero, zero).is_ok());
    assert!(StandardMin.check(neg_zero, neg_zero, neg_zero).is_ok());

    assert!(StandardMin.check(1.0f32, neg_zero, neg_zero).is_ok());
    assert!(StandardMin.check(1.0f32, neg_zero, zero).is_ok());
    assert!(StandardMin.check(1.0f32, neg_zero, 1.0).is_err());

    assert!(StandardMin.check(neg_zero, 1.0f32, neg_zero).is_ok());
    assert!(StandardMin.check(neg_zero, 1.0f32, zero).is_ok());
    assert!(StandardMin.check(neg_zero, 1.0f32, 1.0).is_err());

    // NaN handling.
    assert!(StandardMin.check(zero, nan, zero).is_ok());
    assert!(StandardMin.check(zero, nan, neg_zero).is_ok());
    assert!(StandardMin.check(zero, nan, nan).is_err());
    assert!(StandardMin.check(zero, nan, 1.0).is_err());

    assert!(StandardMin.check(neg_zero, nan, zero).is_ok());
    assert!(StandardMin.check(neg_zero, nan, neg_zero).is_ok());
    assert!(StandardMin.check(neg_zero, nan, nan).is_err());
    assert!(StandardMin.check(neg_zero, nan, 1.0).is_err());

    assert!(StandardMin.check(nan, zero, zero).is_ok());
    assert!(StandardMin.check(nan, zero, neg_zero).is_ok());
    assert!(StandardMin.check(nan, zero, nan).is_err());
    assert!(StandardMin.check(nan, zero, 1.0).is_err());

    assert!(StandardMin.check(nan, neg_zero, zero).is_ok());
    assert!(StandardMin.check(nan, neg_zero, neg_zero).is_ok());
    assert!(StandardMin.check(nan, neg_zero, nan).is_err());
    assert!(StandardMin.check(nan, neg_zero, 1.0).is_err());
}
