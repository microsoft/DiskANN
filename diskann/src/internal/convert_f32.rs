/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ops::Deref;

use diskann_vector::conversion::CastFromSlice;
use half::f16;

/// Convert self to something that dereferences to a slice of `f32` as efficiently as possible.
///
/// Some compressed distance implementations wish to consume an argument as a slice of `f32`.
///
/// When data is easily converted to such a slice, we wish to avoid any allocations and
/// instead return a reference to the underlying slice directly.
///
/// However, when conversion needs to happen, we must return something that owns the
/// converted data.
pub(crate) trait ConvertF32 {
    /// The intermediate type returned by `convert_f32`.
    ///
    /// We require that this dereferences to a `&[f32]`.
    /// This allows returning `&[f32]` directly, or a `Vec<f32>` if a slice cannot be
    /// readily obtained.
    type Returns<'a>: Deref<Target = [f32]>
    where
        Self: 'a;

    /// Efficiently represent `self` as something convertible to a `&[f32]`.
    fn convert_f32(&self) -> Self::Returns<'_>;
}

/// For `[f32]` the conversion is the identity operation.
impl ConvertF32 for [f32] {
    type Returns<'a>
        = &'a [f32]
    where
        Self: 'a;

    fn convert_f32(&self) -> Self::Returns<'_> {
        self
    }
}

macro_rules! bulk_conversion_allocating {
    ($T:ty) => {
        /// Convert by allocating and populating a vector with element-wise conversion.
        impl ConvertF32 for [$T] {
            type Returns<'a>
                = Vec<f32>
            where
                Self: 'a;

            fn convert_f32(&self) -> Self::Returns<'_> {
                self.iter().map(|i| (*i).into()).collect()
            }
        }
    };
}

bulk_conversion_allocating!(i8);
bulk_conversion_allocating!(u8);

impl ConvertF32 for [f16] {
    type Returns<'a> = Vec<f32>;

    fn convert_f32(&self) -> Self::Returns<'_> {
        let mut output = vec![f32::default(); self.len()];
        output.cast_from_slice(self);
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ensure that `[f32]` is the identity.
    #[test]
    fn f32_is_identity() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];
        let ptr = v.as_ptr();

        let u = v.convert_f32();
        assert_eq!(u, v);
        assert_eq!(
            u.as_ptr(),
            ptr,
            "conversion should be the identity and return the underlying span"
        );
    }

    fn test_allocating<T>(input: &[T])
    where
        T: Into<f32> + Copy,
        [T]: ConvertF32,
    {
        let converted = input.convert_f32();
        assert_eq!(converted.len(), input.len());
        for (i, (c, n)) in std::iter::zip(converted.iter(), input.iter()).enumerate() {
            assert_eq!(
                *c,
                <T as Into<f32>>::into(*n),
                "conversion failed for input {i}"
            );
        }
    }

    #[test]
    fn test_i8() {
        let v: Vec<i8> = vec![1, 2, 3];
        test_allocating(&v);
    }

    #[test]
    fn test_u8() {
        let v: Vec<u8> = vec![1, 2, 3];
        test_allocating(&v);
    }

    #[test]
    fn test_f16() {
        let v: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        test_allocating(&v);
    }
}
