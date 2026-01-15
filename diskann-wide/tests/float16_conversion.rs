/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! This test ensures that conversions between `f32` and `f16` are consistent.
//!
//! # `f16` to `f32`
//!
//! The conversion from `f16` to `f32` is expected to be lossless and is fairly
//! straight-forward to test.
//!
//! # `f32` to `f16`
//!
//! Wide uses a "round to nearest" protocol, meaning that `f32` values get rounded to the
//! nearest representable `f16` value. For values that lie exactly inbetween two values,
//! we round to the `f16` value with an "even" mantissa, meaning the least-significant bit
//! in the mantissa is zero.
//!
//! This seems like an odd distinction to make, but it's how the hardware intrinsics behave
//! and the mixture of up-or-down rounding can avoid introducing systematic bias into the
//! conversion.
//!
//! # Input File

use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use diskann_wide::{Const, SIMDVector, SupportedLaneCount};
use half::f16;

// File paths for float16 data.
fn test_file_path(name: &str) -> String {
    format!("{}/test_data/{}", env!("CARGO_MANIFEST_DIR"), name)
}

const FLOAT16_INPUT_FILE: &str = "float16_conversion.txt";

/// Parse the float16 to float32 reference file.
///
/// The file consists of 65536 lines, each with layout
/// ```text
/// 0x0000, <number>
/// ```
/// where the first hexadecimal number is a bit pattern for a float16 value and
/// `<number>` is its corresponding `f32` representation. There are some special values
/// for `<number>` though:
///
/// * "neg_infinity": For negative infinity
/// * "infinity": For infinity
/// * "nan": For NaN
///
/// If it is not one of these sentinel values, then it should be parseable as a `f32`.
///
/// Note that the values are sorted from lowest to highest.
fn parse_float16_reference_file() -> Vec<(f16, f32)> {
    let file = File::open(test_file_path(FLOAT16_INPUT_FILE)).unwrap();
    let reader = BufReader::new(file);

    let mut output = Vec::new();
    for line in reader.lines() {
        // Hope that `Lines` doesn't yield an error.
        let line = line.unwrap();

        // Split on the comma that should separate lines.
        let mut split = line.split(", ");
        let float16 = match split.next() {
            Some(field) => field,
            None => panic!("could not parse {}", line),
        };

        let float32 = match split.next() {
            Some(field) => field,
            None => panic!("could not parse {}", line),
        };

        // This is mainly an integrity check on the input file.
        assert!(split.next().is_none());

        // Try to parse the float16 as a `u16`. If successful, turn it into a `float16`.
        let float16: f16 = match u16::from_str_radix(float16.trim_start_matches("0x"), 16) {
            Ok(number) => f16::from_bits(number),
            Err(err) => panic!("could not parse {} as a u16, error: {}", float16, err),
        };

        let float32: f32 = match float32 {
            "neg_infinity" => f32::NEG_INFINITY,
            "infinity" => f32::INFINITY,
            "nan" => f32::NAN,
            other => match other.parse::<f32>() {
                Ok(number) => number,
                Err(err) => panic!("could not parse {} as a f32, error: {}", float32, err),
            },
        };

        output.push((float16, float32))
    }

    assert_eq!(
        output.len(),
        65536,
        "conversion file should have exactly 65536 entries"
    );
    output
}

////////////////
// f16 to f32 //
////////////////

fn test_f16_to_f32_exhaustive<T, U, const N: usize>(
    cases: &[(f16, f32)],
    convert: &dyn Fn(U) -> T,
    opname: &str,
) where
    Const<N>: SupportedLaneCount,
    T: SIMDVector<Arch = diskann_wide::arch::Current, Scalar = f32, ConstLanes = Const<N>>,
    U: SIMDVector<Arch = diskann_wide::arch::Current, Scalar = f16, ConstLanes = Const<N>>,
{
    for case in cases.iter() {
        let from = U::splat(diskann_wide::ARCH, case.0);
        let converted: T = convert(from);

        let check = |c: f32| {
            if case.1.is_nan() {
                assert!(c.is_nan(), "failed for case: {:?}. Op = {}", case, opname);
            } else {
                assert_eq!(c, case.1, "failed for case: {:?}. Op = {}", case, opname);
            }
        };

        for c in converted.to_array() {
            check(c);
        }

        // Also check scalar conversion.
        let converted = diskann_wide::cast_f16_to_f32(case.0);
        check(converted);
    }
}

diskann_wide::alias!(f32x8);
diskann_wide::alias!(f16x8);

#[test]
fn test_f16_to_f32() {
    let cases = parse_float16_reference_file();

    // Test via `into`.
    test_f16_to_f32_exhaustive::<f32x8, f16x8, 8>(&cases, &|x| x.into(), "into");

    // Test via `cast`.
    test_f16_to_f32_exhaustive::<f32x8, f16x8, 8>(&cases, &|x| x.cast(), "cast");
}

////////////////
// f32 to f16 //
////////////////

// The testing methodology here is a little more nuanced.
//
// We want to test the round-to-nearest behavior as well as the handling of NaNs.
fn test_f32_to_f16_exhaustive<T, U, const N: usize>(
    cases: &[(f16, f32)],
    convert: &dyn Fn(U) -> T,
    opname: &str,
) where
    Const<N>: SupportedLaneCount,
    T: SIMDVector<Arch = diskann_wide::arch::Current, Scalar = f16, ConstLanes = Const<N>>,
    U: SIMDVector<Arch = diskann_wide::arch::Current, Scalar = f32, ConstLanes = Const<N>>,
{
    let test_conversion = |from: f32, expected: f16| {
        // The conversion from `f16` to `f32` is lossless.
        let expected: f32 = diskann_wide::cast_f16_to_f32(expected);

        // First, go through bulk conversion.
        let wide = U::splat(diskann_wide::ARCH, from);
        let converted = convert(wide).to_array();

        let check = |c: f16| {
            let c: f32 = c.into();
            if expected.is_nan() {
                assert!(
                    c.is_nan(),
                    "failed for case: {:?}. Op = {}",
                    (from, expected),
                    opname
                );
            } else {
                assert_eq!(
                    c,
                    expected,
                    "failed for case: {:?}. Op = {}",
                    (from, expected),
                    opname
                );
            }
        };

        for c in converted {
            check(c);
        }

        // Fallback Conversion.
        let converted = diskann_wide::cast_f32_to_f16(from);
        check(converted);
    };

    // Check the special corner cases.
    test_conversion(f32::INFINITY, f16::INFINITY);
    test_conversion(f32::MAX, f16::INFINITY);
    test_conversion(f32::NEG_INFINITY, f16::NEG_INFINITY);
    test_conversion(f32::MIN, f16::NEG_INFINITY);
    test_conversion(f32::NAN, f16::NAN);

    // Now, go through our test cases.
    //
    // Try the exact conversion and the interpolate between two values to ensure we
    // correctly round to nearest.
    //
    // NOTE: This relies on reference file being ordered.
    let mut cases_handled = 0;
    for i in 0..cases.len() - 1 {
        let base = cases[i];
        let next = cases[i + 1];

        // First, check that the exact conversion works correctly.
        test_conversion(base.1, base.0);
        cases_handled += 1;

        // If the next value is a NaN, we're done.
        // All the NaNs are at the end of the reference file.
        if next.1.is_nan() {
            break;
        }

        // Assert that we have a strict order on the cases being handled.
        assert_eq!(base.1.total_cmp(&next.1), std::cmp::Ordering::Less);

        // The interpolation logic below breaks for infinite arguments.
        if base.1.is_infinite() || next.1.is_infinite() {
            continue;
        }

        // Now we know that both `base` and `next` are valid.
        //
        // We interpolate between the two float32 values to ensure that we round to
        // nearest.
        let step = (next.1 - base.1) / 4.0;

        // Closer to `base`.
        test_conversion(base.1 + step, base.0);
        // Closer to `next`.
        test_conversion(base.1 + 3.0 * step, next.0);

        // If we are halfway in between two numbers, then "round-to-nearest" uses a
        //
        // definition the behaves as-if rounding to the 16-bit floating point value with
        // a zero as its least significant bit.
        let bits = base.0.to_bits();
        if bits.is_multiple_of(2) {
            test_conversion(base.1 + 2.0 * step, base.0);
        } else {
            test_conversion(base.1 + 2.0 * step, next.0);
        }
    }
    assert_eq!(
        cases_handled, 63490,
        "expected to handle exactly 63490 non-NAN values for f16"
    );
}

#[test]
fn test_f32_to_f16() {
    let cases = parse_float16_reference_file();
    test_f32_to_f16_exhaustive::<f16x8, f32x8, 8>(&cases, &|x| x.cast(), "cast");
}
