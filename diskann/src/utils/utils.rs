/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::num::NonZeroUsize;

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

#[cfg(test)]
mod test_utils {
    use super::*;

    #[test]
    fn type_str() {
        assert_eq!(u32::type_str(), "u32");
        assert_eq!(u64::type_str(), "u64");
        assert_eq!(usize::type_str(), "usize");
    }
}
