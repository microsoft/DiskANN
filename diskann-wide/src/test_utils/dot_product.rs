/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{Const, SIMDDotProduct, SIMDVector, SupportedLaneCount, reference::ReferenceScalarOps};

/////////////////////////
// Expected Reductions //
/////////////////////////

// A dot product from `From` to `To`, collapsing every `Every` into a single element.
pub(crate) struct DotSchema;

pub(crate) trait ExpectedDot<To, Left, Right, const EVERY: usize> {
    // The expected operation to be applied to groups of `EVERY` elements at a time and
    // accumulated with the scalar accumulator.
    fn expected_dot_impl(accumulator: To, left: &[Left; EVERY], right: &[Right; EVERY]) -> To;

    //////////////////////
    // Supplied Methods //
    //////////////////////

    fn expected_dot(accumulator: To, left: &[Left], right: &[Right]) -> To {
        Self::expected_dot_impl(
            accumulator,
            left.try_into().unwrap(),
            right.try_into().unwrap(),
        )
    }
}

// Elements are promoted to `i32`, pairwise multiplied, summed, then accumulated.
impl ExpectedDot<i32, i16, i16, 2> for DotSchema {
    fn expected_dot_impl(accumulator: i32, left: &[i16; 2], right: &[i16; 2]) -> i32 {
        let l0: i32 = left[0].into();
        let l1: i32 = left[1].into();

        let r0: i32 = right[0].into();
        let r1: i32 = right[1].into();

        accumulator.expected_add_((l0.expected_mul_(r0)).expected_add_(l1.expected_mul_(r1)))
    }
}

// Elements are promoted to `i32` - pairwise multiplied, summed, then accumulated.
impl ExpectedDot<i32, i8, u8, 4> for DotSchema {
    fn expected_dot_impl(accumulator: i32, left: &[i8; 4], right: &[u8; 4]) -> i32 {
        let l0: i32 = left[0].into();
        let l1: i32 = left[1].into();
        let l2: i32 = left[2].into();
        let l3: i32 = left[3].into();

        let r0: i32 = right[0].into();
        let r1: i32 = right[1].into();
        let r2: i32 = right[2].into();
        let r3: i32 = right[3].into();

        accumulator.expected_add_(
            l0.expected_mul_(r0)
                .expected_add_(l1.expected_mul_(r1))
                .expected_add_(l2.expected_mul_(r2))
                .expected_add_(l3.expected_mul_(r3)),
        )
    }
}
impl ExpectedDot<i32, u8, i8, 4> for DotSchema {
    fn expected_dot_impl(accumulator: i32, left: &[u8; 4], right: &[i8; 4]) -> i32 {
        Self::expected_dot_impl(accumulator, right, left)
    }
}

////////////////
// Test Macro //
////////////////

pub(crate) fn test_dot_product_impl<
    To,
    Left,
    Right,
    const FROM_N: usize,
    const TO_N: usize,
    const EVERY: usize,
>(
    arch: To::Arch,
    accumulator: &[To::Scalar],
    left: &[Left::Scalar],
    right: &[Right::Scalar],
) where
    Const<FROM_N>: SupportedLaneCount,
    Const<TO_N>: SupportedLaneCount,
    To: SIMDVector<ConstLanes = Const<TO_N>> + SIMDDotProduct<Left, Right>,
    Left: SIMDVector<Arch = To::Arch, ConstLanes = Const<FROM_N>>,
    Right: SIMDVector<Arch = To::Arch, ConstLanes = Const<FROM_N>>,
    DotSchema: ExpectedDot<To::Scalar, Left::Scalar, Right::Scalar, EVERY>,
    To::Scalar: PartialEq + Copy,
    Left::Scalar: Copy,
    Right::Scalar: Copy,
{
    assert_eq!(FROM_N / TO_N, EVERY);
    assert_eq!(FROM_N % TO_N, 0);

    let accumulator =
        <&[To::Scalar] as TryInto<[To::Scalar; TO_N]>>::try_into(accumulator).unwrap();
    let left = <&[Left::Scalar] as TryInto<[Left::Scalar; FROM_N]>>::try_into(left).unwrap();
    let right = <&[Right::Scalar] as TryInto<[Right::Scalar; FROM_N]>>::try_into(right).unwrap();

    let wa = To::from_array(arch, accumulator);
    let wl = Left::from_array(arch, left);
    let wr = Right::from_array(arch, right);

    let result = wa.dot_simd(wl, wr).to_array();

    // Check each result.
    let iter = std::iter::zip(left.chunks(EVERY), right.chunks(EVERY)).enumerate();
    for (i, (l, r)) in iter {
        let expected = DotSchema::expected_dot(accumulator[i], l, r);
        assert_eq!(expected, result[i])
    }
}

macro_rules! test_dot_product {
    (
        ($left:ident $(< $($ls:tt),+ >)?, $right:ident $(< $($rs:tt),+ >)?) => $to:ident $(< $($ts:tt),+ >)?,
        $seed:literal,
        $arch:expr
    ) => {
        paste::paste! {
            #[test]
            fn [<
                dot_product_
                $left:lower $(_$($ls )x+)?
                _and_
                $right:lower $(_$($rs )x+)?
                _to_
                $to:lower $(_$($ts )x+)?
            >]() {
                use $crate::SIMDVector;

                type To = $to $(< $($ts),+>)?;
                type Left = $left $(< $($ls),+>)?;
                type Right = $right $(< $($rs),+>)?;

                type ScalarTo = <To as SIMDVector>::Scalar;
                type ScalarLeft = <Left as SIMDVector>::Scalar;
                type ScalarRight = <Right as SIMDVector>::Scalar;

                const { assert!(Left::LANES == Right::LANES, "lanes must be equal") };

                if let Some(arch) = $arch {
                    let f = move |a: &[ScalarTo], b: &[ScalarLeft], c: &[ScalarRight]| {
                        $crate::test_utils::dot_product::test_dot_product_impl::<
                            To,
                            Left,
                            Right,
                            { <Left>::LANES },
                            { <To>::LANES },
                            { <Left>::LANES / <To>::LANES }
                        >(arch, a, b, c)
                    };

                    let nto = <To>::LANES;
                    let nfrom = <Left>::LANES;
                    $crate::test_utils::driver::drive_ternary(
                        &f,
                        (nto, nfrom, nfrom),
                        $seed,
                    );
                }
            }
        }
    }
}

// Export the test macro.
pub(crate) use test_dot_product;

#[cfg(test)]
mod tests {
    use super::*;

    // Ensure that the expected operation indeed promotes to `i32` for intermediate results.
    #[test]
    fn test_promotion_i16_i32_2() {
        let acc: i32 = i16::MAX.into();
        let left: [i16; 2] = [i16::MAX, i16::MAX];
        let right: [i16; 2] = [i16::MAX, i16::MAX];

        assert_eq!(
            <DotSchema as ExpectedDot<i32, i16, i16, 2>>::expected_dot(acc, &left, &right),
            2147385345
        );

        let acc: i32 = 0;
        let left: [i16; 2] = [i16::MIN, i16::MIN];
        let right: [i16; 2] = [i16::MIN, i16::MIN + 1]; // Subtract 1 so the result fits inside an
        // `i32`

        assert_eq!(
            <DotSchema as ExpectedDot<i32, i16, i16, 2>>::expected_dot(acc, &left, &right),
            2147450880
        );
    }

    #[test]
    fn test_u8_i8_to_i32() {
        let a: &[[u8; 4]] = &[
            [u8::MIN, u8::MIN, u8::MIN, u8::MIN],
            [u8::MIN, u8::MIN, u8::MIN, u8::MAX],
            [u8::MIN, u8::MIN, u8::MAX, u8::MIN],
            [u8::MIN, u8::MIN, u8::MAX, u8::MAX],
            [u8::MIN, u8::MAX, u8::MIN, u8::MIN],
            [u8::MIN, u8::MAX, u8::MIN, u8::MAX],
            [u8::MIN, u8::MAX, u8::MAX, u8::MIN],
            [u8::MIN, u8::MAX, u8::MAX, u8::MAX],
            [u8::MAX, u8::MIN, u8::MIN, u8::MIN],
            [u8::MAX, u8::MIN, u8::MIN, u8::MAX],
            [u8::MAX, u8::MIN, u8::MAX, u8::MIN],
            [u8::MAX, u8::MIN, u8::MAX, u8::MAX],
            [u8::MAX, u8::MAX, u8::MIN, u8::MIN],
            [u8::MAX, u8::MAX, u8::MIN, u8::MAX],
            [u8::MAX, u8::MAX, u8::MAX, u8::MIN],
            [u8::MAX, u8::MAX, u8::MAX, u8::MAX],
        ];

        let b: &[[i8; 4]] = &[
            [i8::MIN, i8::MIN, i8::MIN, i8::MIN],
            [i8::MIN, i8::MIN, i8::MIN, i8::MAX],
            [i8::MIN, i8::MIN, i8::MAX, i8::MIN],
            [i8::MIN, i8::MIN, i8::MAX, i8::MAX],
            [i8::MIN, i8::MAX, i8::MIN, i8::MIN],
            [i8::MIN, i8::MAX, i8::MIN, i8::MAX],
            [i8::MIN, i8::MAX, i8::MAX, i8::MIN],
            [i8::MIN, i8::MAX, i8::MAX, i8::MAX],
            [i8::MAX, i8::MIN, i8::MIN, i8::MIN],
            [i8::MAX, i8::MIN, i8::MIN, i8::MAX],
            [i8::MAX, i8::MIN, i8::MAX, i8::MIN],
            [i8::MAX, i8::MIN, i8::MAX, i8::MAX],
            [i8::MAX, i8::MAX, i8::MIN, i8::MIN],
            [i8::MAX, i8::MAX, i8::MIN, i8::MAX],
            [i8::MAX, i8::MAX, i8::MAX, i8::MIN],
            [i8::MAX, i8::MAX, i8::MAX, i8::MAX],
        ];

        let bases = [0, 1, -1, i16::MAX as i32, i16::MIN as i32];

        for left in a {
            for right in b {
                let dot: i32 = (*left)
                    .into_iter()
                    .zip((*right).into_iter())
                    .map(|(l, r)| (l as i32) * (r as i32))
                    .sum();
                for b in bases {
                    let expected = dot + b;
                    assert_eq!(
                        expected,
                        DotSchema::expected_dot(b, left, right),
                        "failed for: base = {}, left = {:?}, right = {:?}",
                        b,
                        left,
                        right,
                    );

                    assert_eq!(
                        expected,
                        DotSchema::expected_dot(b, right, left),
                        "failed for: base = {}, left = {:?}, right = {:?}",
                        b,
                        right,
                        left,
                    );
                }
            }
        }
    }
}
