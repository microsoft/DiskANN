/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Move a const-generic into the type domain to work around issues with the use and
/// compile-time computation involving const-generic parameters.
///
/// See the notes for the various helper traits in `traits.rs`.
pub struct Const<const N: usize> {}

/// A trait to model compile-time constants.
pub trait Constant {
    type Type;
    fn value() -> Self::Type;
}

impl<const N: usize> Constant for Const<N> {
    type Type = usize;
    fn value() -> Self::Type {
        N
    }
}

/// Mapping from a number of lanes to the underlying type for a bitmask.
///
/// In practice, the BitMask types are all unsigned integers.
/// However, there is not a great way to express that directly in Rust.
///
/// Instead, we have a subset of requirements:
/// * `std::default::Default`: For integers, defaults to 0.
/// * `std::marker::Copy`: Needed so bitmasks can be copy as well.
/// * `std::fmt::Debug`: Allow debug implementations.
/// * `std::cmp::Eq`: Allow efficient equality.
pub trait SupportedLaneCount {
    type BitMaskType: std::default::Default + std::marker::Copy + std::fmt::Debug + std::cmp::Eq;
}

impl SupportedLaneCount for Const<1> {
    type BitMaskType = u8;
}

impl SupportedLaneCount for Const<2> {
    type BitMaskType = u8;
}

impl SupportedLaneCount for Const<4> {
    type BitMaskType = u8;
}

impl SupportedLaneCount for Const<8> {
    type BitMaskType = u8;
}

impl SupportedLaneCount for Const<16> {
    type BitMaskType = u16;
}

impl SupportedLaneCount for Const<32> {
    type BitMaskType = u32;
}

impl SupportedLaneCount for Const<64> {
    type BitMaskType = u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const() {
        let x: usize = Const::<1>::value();
        assert_eq!(x, 1);

        let x: usize = Const::<2>::value();
        assert_eq!(x, 2);

        let x: usize = Const::<4>::value();
        assert_eq!(x, 4);

        let x: usize = Const::<8>::value();
        assert_eq!(x, 8);

        let x: usize = Const::<16>::value();
        assert_eq!(x, 16);

        let x: usize = Const::<32>::value();
        assert_eq!(x, 32);

        let x: usize = Const::<64>::value();
        assert_eq!(x, 64);
    }
}
