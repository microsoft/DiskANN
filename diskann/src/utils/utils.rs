/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    fmt::{Debug, Display},
    num::NonZeroUsize,
};

use num_traits::FromPrimitive;

use crate::{ANNError, ANNResult};

/// A constant representing the number one as a `NonZeroUsize`.
pub const ONE: NonZeroUsize = NonZeroUsize::new(1).unwrap();

/// Read exact number of POD type elements from a reader into a vector.
pub fn read_exact_into<T, R>(reader: &mut R, count: usize) -> std::io::Result<Vec<T>>
where
    T: bytemuck::Pod + Default,
    R: std::io::Read,
{
    let mut buf = vec![T::default(); count];
    // Casting Pod type to bytes always succeeds (u8 has alignment of 1)
    let byte_slice = bytemuck::must_cast_slice_mut(&mut buf);
    reader.read_exact(byte_slice)?;

    Ok(buf)
}

pub fn vecid_from_u32<T>(val: u32) -> ANNResult<T>
where
    T: FromPrimitive,
{
    let res = T::from_u32(val).ok_or_else(|| {
        ANNError::log_index_error(format_args!(
            "Failed to convert from u32 to VectorIdType for vector {}",
            val
        ))
    })?;

    Ok(res)
}

pub fn vecid_from_usize<T>(val: usize) -> ANNResult<T>
where
    T: FromPrimitive,
{
    let res = T::from_usize(val).ok_or_else(|| {
        ANNError::log_index_error(format_args!(
            "Failed to convert from usize to VectorIdType for vector {}",
            val
        ))
    })?;

    Ok(res)
}

/// In Rust, `From` conversion for `usize` from various primitive types is not implemented
/// because on some platforms, `usize` can be 64-bits.
///
/// In general, we expect `DiskANN` to only be run on 64-bit systems, where can can assume
/// the size of `usize` is 64-bits.
///
/// This trait allows us to fearlessly implement conversions to `usize` without resorting
/// to undefined conversions like `as_`.
/// ```rust
/// use diskann::utils::IntoUsize;
/// // Before, our usage pattern was like this.
/// let x: u64 = u64::MAX;
/// let x_usize: usize = x as usize;
///
/// // This same pattern was vulnerable to type changes.
/// let x: i64 = i64::MIN;
/// let x_usize: usize = x as usize; // !!! WHAT SHOULD THIS MEAN !!!
///
/// // Now, we can do this fearlessly.
/// let x: u64 = u64::MAX;
/// let x_usize: usize = x.into_usize();
///
/// ```
/// The following is now a compiler error.
/// ```compile_fail
/// use diskann::utils::IntoUsize;
/// let x: i64 = i64::MIN;
/// let x_usize: usize = x.into_usize();
/// ```
///
/// This trait is `unsafe` because defining it on a 32-bit architecture for 64-bit sources
/// will result in lossy conversions.
///
/// # Safety
///
/// While "safe" from a memory perspective, this can very much have unintended side-effects.
///
/// In order to safely implement this trait, the implementer must assert *at compile time*
/// that the conversion will be lossless.
pub unsafe trait IntoUsize {
    fn into_usize(self) -> usize;
}

macro_rules! impl_to_usize {
    ($type:ty) => {
        /// SAFETY: We have checked that the target pointer width is 64-bits, meaning we are on
        /// a 64-bit system so the conversion is lossless.
        #[cfg(target_pointer_width = "64")]
        unsafe impl IntoUsize for $type {
            fn into_usize(self) -> usize {
                // This breaks at compile time if somehow the `cfg` guard above fails.
                #[allow(unused)]
                const STATIC_ASSERT: () = {
                    if usize::BITS != 64 {
                        panic!("diskann is not compatible with non-64-bit systems");
                    }
                };

                self as usize
            }
        }
    };
}

impl_to_usize!(u8);
impl_to_usize!(u16);
impl_to_usize!(u32);
impl_to_usize!(u64);
impl_to_usize!(usize);

/// Return the name of a type.
///
/// Model this as a custom trait rather than using `std::any::type_name` to provide a stable
/// name (which is not provided by that function).
pub trait TypeStr {
    fn type_str() -> &'static str;
}

impl TypeStr for u32 {
    fn type_str() -> &'static str {
        "u32"
    }
}

impl TypeStr for u64 {
    fn type_str() -> &'static str {
        "u64"
    }
}

impl TypeStr for usize {
    fn type_str() -> &'static str {
        "usize"
    }
}

// For testing purposes, we need a type that is not losslessly convertible to `u64`.
#[cfg(test)]
impl TypeStr for u128 {
    fn type_str() -> &'static str {
        "u128"
    }
}

/// An error type indicating that conversion from an integer to a `VectorId` failed due
/// to a narrowing conversion.
///
/// We can have conversion errors both going from an integer to a `VectorId`, and from a
/// `VectorID` to an integer.
///
/// Since the only thing that changes during error reporting is the string describing the
/// conversion, the direction of the conversion is captured as the const generic `INT_TO_ID`.
#[derive(Debug, Clone, Copy)]
pub struct IdConversionError<const INT_TO_ID: bool, FromType, ToType>(
    FromType,
    std::marker::PhantomData<ToType>,
)
where
    FromType: TypeStr + Display,
    ToType: TypeStr;

impl<const INT_TO_ID: bool, FromType, ToType> IdConversionError<INT_TO_ID, FromType, ToType>
where
    FromType: TypeStr + Display,
    ToType: TypeStr,
{
    fn new(value: FromType) -> Self {
        Self(value, std::marker::PhantomData)
    }
}

impl<const INT_TO_ID: bool, FromType, ToType> Display
    for IdConversionError<INT_TO_ID, FromType, ToType>
where
    FromType: TypeStr + Display,
    ToType: TypeStr,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Converting from an integer to an ID.
        if INT_TO_ID {
            write!(
                f,
                "could not convert integer {}({}) to a VectorId of type {}",
                FromType::type_str(),
                self.0,
                ToType::type_str()
            )
        } else {
            // Converting from an ID to an integer.
            write!(
                f,
                "could not convert VectorId {}({}) to an integer of type {}",
                FromType::type_str(),
                self.0,
                ToType::type_str()
            )
        }
    }
}

// opt-in to error reporting.
impl<const INT_TO_ID: bool, FromType, ToType> std::error::Error
    for IdConversionError<INT_TO_ID, FromType, ToType>
where
    FromType: TypeStr + Display + Debug,
    ToType: TypeStr + Debug,
{
}

/// Allow conversion to `ANNError` for error propagation.
impl<const INT_TO_ID: bool, FromType, ToType> From<IdConversionError<INT_TO_ID, FromType, ToType>>
    for ANNError
where
    FromType: TypeStr + Display,
    ToType: TypeStr,
{
    #[track_caller]
    fn from(err: IdConversionError<INT_TO_ID, FromType, ToType>) -> Self {
        ANNError::log_index_error(err)
    }
}

/// Record that an conversion error occurred while trying to convert to a VectorId.
pub type ErrorToVectorId<FromType, ToType> = IdConversionError<true, FromType, ToType>;

/// Record that conversion error occurred while trying to convert to an Integer.
pub type ErrorToInt<FromType, ToType> = IdConversionError<false, FromType, ToType>;

/// Try to convert a type into a vector id. IF such a conversion can not be performed
/// losslessly, return an `ErrorToVectorId` error, recording the involved types and the
/// value that caused conversion to fail.
///
/// # Examples
///
/// The example below shows a failed conversion.
/// ```rust
/// use diskann::utils::VectorIdTryFrom;
///
/// let x = u32::vector_id_try_from(u64::MAX);
/// assert!(x.is_err());
/// assert_eq!(
///     x.unwrap_err().to_string(),
///     "could not convert integer u64(18446744073709551615) to a VectorId of type u32"
/// );
/// ```
///
/// A successful conversion will look like the example below.
/// ```rust
/// use diskann::utils::VectorIdTryFrom;
///
/// let input: u64 = 500;
/// let x = u32::vector_id_try_from(input).unwrap();
/// assert_eq!(u64::from(x), input);
/// ```
pub trait VectorIdTryFrom<T: TypeStr + Copy + Display>: TypeStr + Sized {
    fn vector_id_try_from(value: T) -> Result<Self, ErrorToVectorId<T, Self>>;
}

/// Implement `VectorIdTryFrom` for pairs of types that implement `TryFrom`.
///
/// Note that the implementation *does* overwrite whatever error is yielded from the
/// initial `try_from`.
impl<T, U> VectorIdTryFrom<T> for U
where
    T: Copy + TypeStr + Display,
    U: TryFrom<T> + TypeStr,
{
    fn vector_id_try_from(value: T) -> Result<Self, ErrorToVectorId<T, Self>> {
        <U as TryFrom<T>>::try_from(value).map_err(|_| ErrorToVectorId::new(value))
    }
}

/// Try to convert into a VectorId. If such a conversion can not be performed losslessly,
/// return an `ErrorToVectorId` recording the involved types and the value that caused
/// conversion to fail.
/// ```rust
/// use diskann::utils::TryIntoVectorId;
///
/// let x: Result<u32, _> = (u64::MAX).try_into_vector_id();
/// assert!(x.is_err());
/// assert_eq!(
///     x.unwrap_err().to_string(),
///     "could not convert integer u64(18446744073709551615) to a VectorId of type u32"
/// );
/// ```
///
/// Conversions that succeed can be processed like normal.
/// ```rust
/// use diskann::utils::TryIntoVectorId;
///
/// let input: u64 = 10;
/// let x: u32 = input.try_into_vector_id().unwrap();
/// assert_eq!(x, 10);
/// ```
///
/// Like the Rust trait [TryInto](https://doc.rust-lang.org/std/convert/trait.TryInto.html),
/// this trait should not be implemented directly. Rather, the trait `VectorIdTryFrom`
/// should be implemented instead.
pub trait TryIntoVectorId<T>: Copy + TypeStr + Display
where
    T: TypeStr,
{
    /// Perform the conversion.
    fn try_into_vector_id(self) -> Result<T, ErrorToVectorId<Self, T>>;
}

/// The trait `TryIntoVectorId<T>` is automatically implemented for `U` when
/// `U: VectorIdTryFrom<T>`.
impl<T, U> TryIntoVectorId<T> for U
where
    U: Copy + TypeStr + Display,
    T: VectorIdTryFrom<U>,
{
    fn try_into_vector_id(self) -> Result<T, ErrorToVectorId<Self, T>> {
        <T as VectorIdTryFrom<U>>::vector_id_try_from(self).map_err(|_| ErrorToVectorId::new(self))
    }
}

/// Try to convert a `VectorId` to an integer. If such a conversion cannot be performed
/// losslessly, return an `ErrorToInt` recording the involved types and the value that
/// caused the conversion to fail.
///
/// In many ways, this trait is the pair for `TryIntoVectorId`, but its corresponding error
/// message reports the direction of the conversion.
pub trait TryIntoInteger<To>: Copy + TypeStr + Display
where
    To: Sized + TypeStr,
{
    fn try_into_integer(self) -> Result<To, ErrorToInt<Self, To>>;
}

impl<To, From> TryIntoInteger<To> for From
where
    From: Copy + TypeStr + Display,
    To: Sized + TypeStr + TryFrom<From>,
{
    fn try_into_integer(self) -> Result<To, ErrorToInt<Self, To>> {
        <To as TryFrom<Self>>::try_from(self).map_err(|_| ErrorToInt::new(self))
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::ANNErrorKind;

    #[test]
    fn vecid_from_u32_test() {
        // success: small u32 -> u64
        let val: u32 = 12345;
        let got: u64 = vecid_from_u32(val).unwrap();
        assert_eq!(got, val as u64);

        // failure: u32::MAX cannot fit into u16 -> should return ANNError
        let res = vecid_from_u32::<u16>(u32::MAX);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), ANNErrorKind::IndexError);
    }

    #[test]
    fn vecid_from_usize_test() {
        // success: small usize -> u64
        let val: usize = 12345;
        let got: u64 = vecid_from_usize(val).unwrap();
        assert_eq!(got, val as u64);

        // failure: usize::MAX cannot fit into u32 -> should return ANNError
        let res = vecid_from_usize::<u32>(usize::MAX);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().kind(), ANNErrorKind::IndexError);
    }

    #[test]
    fn type_str() {
        assert_eq!(u32::type_str(), "u32");
        assert_eq!(u64::type_str(), "u64");
        assert_eq!(u128::type_str(), "u128");
        assert_eq!(usize::type_str(), "usize");
    }

    fn int_to_id_message<From: TypeStr + Display, To: TypeStr>(value: From) -> String {
        format!(
            "could not convert integer {}({}) to a VectorId of type {}",
            From::type_str(),
            value,
            To::type_str()
        )
    }

    fn id_to_int_message<From: TypeStr + Display, To: TypeStr>(value: From) -> String {
        format!(
            "could not convert VectorId {}({}) to an integer of type {}",
            From::type_str(),
            value,
            To::type_str()
        )
    }

    #[test]
    fn id_conversion_error_new_direct() {
        // direct constructor test for IdConversionError::new and Display
        let err = IdConversionError::<true, u32, usize>::new(42);
        assert_eq!(
            err.to_string(),
            "could not convert integer u32(42) to a VectorId of type usize"
        );

        let err = IdConversionError::<false, usize, u32>::new(7);
        assert_eq!(
            err.to_string(),
            "could not convert VectorId usize(7) to an integer of type u32"
        );
    }

    #[test]
    fn id_conversion_error_messages() {
        // Int to ID: From u32
        let err = IdConversionError::<true, u32, u32>::new(10);
        assert_eq!(err.to_string(), int_to_id_message::<u32, u32>(10));
        let err = IdConversionError::<true, u32, u64>::new(11);
        assert_eq!(err.to_string(), int_to_id_message::<u32, u64>(11));
        let err = IdConversionError::<true, u32, usize>::new(12);
        assert_eq!(err.to_string(), int_to_id_message::<u32, usize>(12));

        // Int to ID: From u64
        let err = IdConversionError::<true, u64, u32>::new(10);
        assert_eq!(err.to_string(), int_to_id_message::<u64, u32>(10));
        let err = IdConversionError::<true, u64, u64>::new(11);
        assert_eq!(err.to_string(), int_to_id_message::<u64, u64>(11));
        let err = IdConversionError::<true, u64, usize>::new(12);
        assert_eq!(err.to_string(), int_to_id_message::<u64, usize>(12));

        // Int to ID: From usize
        let err = IdConversionError::<true, usize, u32>::new(10);
        assert_eq!(err.to_string(), int_to_id_message::<usize, u32>(10));
        let err = IdConversionError::<true, usize, u64>::new(11);
        assert_eq!(err.to_string(), int_to_id_message::<usize, u64>(11));
        let err = IdConversionError::<true, usize, usize>::new(12);
        assert_eq!(err.to_string(), int_to_id_message::<usize, usize>(12));

        // ID to Int: From u32
        let err = IdConversionError::<false, u32, u32>::new(10);
        assert_eq!(err.to_string(), id_to_int_message::<u32, u32>(10));
        let err = IdConversionError::<false, u32, u64>::new(11);
        assert_eq!(err.to_string(), id_to_int_message::<u32, u64>(11));
        let err = IdConversionError::<false, u32, usize>::new(12);
        assert_eq!(err.to_string(), id_to_int_message::<u32, usize>(12));

        // ID to Int: From u64
        let err = IdConversionError::<false, u64, u32>::new(10);
        assert_eq!(err.to_string(), id_to_int_message::<u64, u32>(10));
        let err = IdConversionError::<false, u64, u64>::new(11);
        assert_eq!(err.to_string(), id_to_int_message::<u64, u64>(11));
        let err = IdConversionError::<false, u64, usize>::new(12);
        assert_eq!(err.to_string(), id_to_int_message::<u64, usize>(12));

        // ID to int: From usize
        let err = IdConversionError::<false, usize, u32>::new(10);
        assert_eq!(err.to_string(), id_to_int_message::<usize, u32>(10));
        let err = IdConversionError::<false, usize, u64>::new(11);
        assert_eq!(err.to_string(), id_to_int_message::<usize, u64>(11));
        let err = IdConversionError::<false, usize, usize>::new(12);
        assert_eq!(err.to_string(), id_to_int_message::<usize, usize>(12));
    }

    #[test]
    fn id_conversion_to_annerror() {
        // VectorId -> Int
        let x = ErrorToInt::<u64, u32>::new(500);
        let ann = ANNError::from(x);
        assert_eq!(ann.kind(), ANNErrorKind::IndexError);

        // Int -> VectorId
        let x = ErrorToVectorId::<usize, u32>::new(10);
        let ann = ANNError::from(x);
        assert_eq!(ann.kind(), ANNErrorKind::IndexError);
    }

    #[test]
    fn vector_id_try_from() {
        let convertible_u64: u64 = 321;
        let convertible_usize: usize = 1234;
        // A 128-bit integer that is convertible to u64 but not u32.
        let convertible_u128: u128 = (u64::MAX).into();
        assert!(u32::try_from(convertible_u128).is_err());
        assert!(u64::try_from(convertible_u128).is_ok());

        // to u32 - errors
        let x: Result<u32, _> = u32::vector_id_try_from(u64::MAX);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            int_to_id_message::<u64, u32>(u64::MAX)
        );

        let x: Result<u32, _> = u32::vector_id_try_from(usize::MAX);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            int_to_id_message::<usize, u32>(usize::MAX)
        );

        // to u32 - works
        let x: Result<u32, _> = u32::vector_id_try_from(convertible_u64);
        assert_eq!(
            x.unwrap(),
            <u64 as TryInto<u32>>::try_into(convertible_u64).unwrap()
        );

        let x: Result<u32, _> = u32::vector_id_try_from(convertible_usize);
        assert_eq!(
            x.unwrap(),
            <usize as TryInto<u32>>::try_into(convertible_usize).unwrap()
        );

        // to u64 - errors
        let x: Result<u64, _> = u64::vector_id_try_from(u128::MAX);
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            int_to_id_message::<u128, u64>(u128::MAX)
        );

        // to u64 - success
        let x: Result<u64, _> = u64::vector_id_try_from(convertible_u128);
        assert_eq!(
            x.unwrap(),
            <u128 as TryInto<u64>>::try_into(convertible_u128).unwrap()
        );

        let x: Result<u64, _> = u64::vector_id_try_from(u32::MAX);
        assert_eq!(x.unwrap(), <u32 as Into<u64>>::into(u32::MAX));
    }

    #[test]
    fn try_into_vector_id() {
        let convertible_u64: u64 = 321;
        let convertible_usize: usize = 1234;
        // A 128-bit integer that is convertible to u64 but not u32.
        let convertible_u128: u128 = (u64::MAX).into();
        assert!(u32::try_from(convertible_u128).is_err());
        assert!(u64::try_from(convertible_u128).is_ok());

        // to u32 - errors
        let x: Result<u32, _> = (u64::MAX).try_into_vector_id();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            int_to_id_message::<u64, u32>(u64::MAX)
        );

        let x: Result<u32, _> = (usize::MAX).try_into_vector_id();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            int_to_id_message::<usize, u32>(usize::MAX)
        );

        // to u32 - works
        let x: Result<u32, _> = convertible_u64.try_into_vector_id();
        assert_eq!(
            x.unwrap(),
            <u64 as TryInto<u32>>::try_into(convertible_u64).unwrap()
        );

        let x: Result<u32, _> = convertible_usize.try_into_vector_id();
        assert_eq!(
            x.unwrap(),
            <usize as TryInto<u32>>::try_into(convertible_usize).unwrap()
        );

        // to u64 - errors
        let x: Result<u64, _> = (u128::MAX).try_into_vector_id();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            int_to_id_message::<u128, u64>(u128::MAX)
        );

        // to u64 - success
        let x: Result<u64, _> = convertible_u128.try_into_vector_id();
        assert_eq!(
            x.unwrap(),
            <u128 as TryInto<u64>>::try_into(convertible_u128).unwrap()
        );

        let x: Result<u64, _> = (u32::MAX).try_into_vector_id();
        assert_eq!(x.unwrap(), <u32 as Into<u64>>::into(u32::MAX));
    }

    #[test]
    fn try_into_integer() {
        let convertible_u64: u64 = 321;
        let convertible_usize: usize = 1234;
        // A 128-bit integer that is convertible to u64 but not u32.
        let convertible_u128: u128 = (u64::MAX).into();
        assert!(u32::try_from(convertible_u128).is_err());
        assert!(u64::try_from(convertible_u128).is_ok());

        // to u32 - errors
        let x: Result<u32, _> = (u64::MAX).try_into_integer();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            id_to_int_message::<u64, u32>(u64::MAX)
        );

        let x: Result<u32, _> = (usize::MAX).try_into_integer();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            id_to_int_message::<usize, u32>(usize::MAX)
        );

        // to u32 - works
        let x: Result<u32, _> = convertible_u64.try_into_integer();
        assert_eq!(
            x.unwrap(),
            <u64 as TryInto<u32>>::try_into(convertible_u64).unwrap()
        );

        let x: Result<u32, _> = convertible_usize.try_into_integer();
        assert_eq!(
            x.unwrap(),
            <usize as TryInto<u32>>::try_into(convertible_usize).unwrap()
        );

        // to u64 - errors
        let x: Result<u64, _> = (u128::MAX).try_into_integer();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            id_to_int_message::<u128, u64>(u128::MAX)
        );

        // to u64 - success
        let x: Result<u64, _> = convertible_u128.try_into_integer();
        assert_eq!(
            x.unwrap(),
            <u128 as TryInto<u64>>::try_into(convertible_u128).unwrap()
        );

        let x: Result<u64, _> = (u32::MAX).try_into_integer();
        assert_eq!(x.unwrap(), <u32 as Into<u64>>::into(u32::MAX));

        // to usize - errors
        let x: Result<usize, _> = (u128::MAX).try_into_integer();
        assert!(x.is_err());
        assert_eq!(
            x.unwrap_err().to_string(),
            id_to_int_message::<u128, usize>(u128::MAX)
        );

        // to usize - success
        let x: Result<usize, _> = convertible_u128.try_into_integer();
        assert_eq!(
            x.unwrap(),
            <u128 as TryInto<usize>>::try_into(convertible_u128).unwrap()
        );

        let x: Result<usize, _> = (u32::MAX).try_into_integer();
        assert_eq!(
            x.unwrap(),
            <u32 as TryInto<usize>>::try_into(u32::MAX).unwrap()
        );
    }
}
