/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{
    arch,
    constant::{Const, SupportedLaneCount},
    splitjoin::{LoHi, SplitJoin},
};

/// A lane-wise mask represented as a bit-mask.
///
/// The representation for this type is the smallest unsigned integer capable of holding
/// `N` bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitMask<const N: usize, A: arch::Sealed = arch::Current>(
    pub <Const<N> as SupportedLaneCount>::BitMaskType,
    A,
)
where
    Const<N>: SupportedLaneCount;

impl<const N: usize, A> BitMask<N, A>
where
    Const<N>: SupportedLaneCount,
    A: arch::Sealed,
{
    pub fn as_scalar(self) -> BitMask<N, arch::emulated::Scalar> {
        BitMask::<N, arch::emulated::Scalar>(self.0, arch::emulated::Scalar)
    }

    pub fn as_current(self) -> BitMask<N, arch::Current> {
        BitMask::<N, arch::Current>(self.0, arch::current())
    }

    pub fn as_arch<B>(self, arch: B) -> BitMask<N, B>
    where
        B: arch::Sealed,
    {
        BitMask(self.0, arch)
    }

    pub(crate) fn get_arch(self) -> A {
        self.1
    }
}

/// Perform a potentially lossy conversion from a raw integer.
///
/// The associated constant `NARROWING` can be queried to check if the conversion is allowed
/// to narrow from the provided integer.
///
/// Narrowing conversions will only retain the lower bits.
pub trait FromInt<I, A: arch::Sealed> {
    /// Will the conversion only sample from the lower-order bits of the provided integer.
    const NARROWING: bool;
    /// Turn an integer into an instance of `Self`.
    fn from_int(arch: A, value: I) -> Self;
}

impl<A: arch::Sealed> FromInt<u8, A> for BitMask<1, A> {
    const NARROWING: bool = true;
    fn from_int(arch: A, value: u8) -> Self {
        Self(value & 0x1, arch)
    }
}

impl<A: arch::Sealed> FromInt<u8, A> for BitMask<2, A> {
    const NARROWING: bool = true;
    fn from_int(arch: A, value: u8) -> Self {
        Self(value & 0x3, arch)
    }
}

impl<A: arch::Sealed> FromInt<u8, A> for BitMask<4, A> {
    const NARROWING: bool = true;
    fn from_int(arch: A, value: u8) -> Self {
        Self(value & 0xF, arch)
    }
}

impl<A: arch::Sealed> FromInt<u8, A> for BitMask<8, A> {
    const NARROWING: bool = false;
    fn from_int(arch: A, value: u8) -> Self {
        Self(value, arch)
    }
}

impl<A: arch::Sealed> FromInt<u16, A> for BitMask<16, A> {
    const NARROWING: bool = false;
    fn from_int(arch: A, value: u16) -> Self {
        Self(value, arch)
    }
}

impl<A: arch::Sealed> FromInt<u32, A> for BitMask<32, A> {
    const NARROWING: bool = false;
    fn from_int(arch: A, value: u32) -> Self {
        Self(value, arch)
    }
}

impl<A: arch::Sealed> FromInt<u64, A> for BitMask<64, A> {
    const NARROWING: bool = false;
    fn from_int(arch: A, value: u64) -> Self {
        Self(value, arch)
    }
}

macro_rules! splitjoin {
    ($from:literal, $to:literal, $mask:literal, $full:ty, $half:ty) => {
        impl<A: arch::Sealed> SplitJoin for BitMask<$from, A> {
            type Halved = BitMask<$to, A>;
            fn split(self) -> LoHi<Self::Halved> {
                let arch = self.1;
                LoHi {
                    lo: Self::Halved::from_int(arch, (self.0 & $mask) as $half),
                    hi: Self::Halved::from_int(arch, ((self.0 >> $to) & $mask) as $half),
                }
            }

            fn join(lohi: LoHi<Self::Halved>) -> Self {
                let arch = lohi.lo.1;
                let lo: $full = lohi.lo.0.into();
                let hi: $full = lohi.hi.0.into();
                Self(hi << $to | lo, arch)
            }
        }
    };
}

splitjoin!(2, 1, 0x1, u8, u8);
splitjoin!(4, 2, 0x3, u8, u8);
splitjoin!(8, 4, 0xf, u8, u8);
splitjoin!(16, 8, 0xff, u16, u8);
splitjoin!(32, 16, 0xffff, u32, u16);
splitjoin!(64, 32, 0xffff_ffff, u64, u32);
