/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

use diskann_vector::{
    DistanceFunction, PreprocessedDistanceFunction,
    conversion::CastFromSlice,
    distance::{DistanceProvider, Metric},
};
use half::f16;
use thiserror::Error;

use crate::{ANNError, internal::convert_f32::ConvertF32};

/// This is the data type for values stored in the graph. This type should implement the
/// following traits:
///
///   * `Sized`: This trait is necessary for the type to have a known size at compile time.
///
///   * `bytemuck::Pod`: The type must be as plain as possible, allowing all possible bit
///     patterns.
///
///   * `Send` & `Sync`: These traits are necessary for allowing safe access from multiple
///     threads, they are needed because the graph will be used in a multithreaded context.
///
///   * `Default`: Type can be deafult constructed.
///
///   * `FromPrimitive`: Constructible from integers.
///
pub trait VectorElement:
    Sized + bytemuck::Pod + num_traits::FromPrimitive + std::fmt::Debug + Default + Send + Sync
{
}

impl<T> VectorElement for T where
    T: Sized + bytemuck::Pod + num_traits::FromPrimitive + std::fmt::Debug + Default + Send + Sync
{
}

/// A common collection of behavior required for element types of vectors that
/// behave like full-precision vectors. It covers native types like `f32`, `f16`
/// `i8` and `u8` but also [`MinMaxElement`] which is an element type to represent
/// vectors quantized using [`quantization::minmax`] and can be used in-place of
/// full-precision vectors.
pub trait VectorRepr: VectorElement {
    /// An error type for implementations that throw-errors when converting to full-precision
    /// vectors; such as [`crate::MinMaxElement`]. For regular full-precision vectors this a
    /// null type.
    type Error: std::error::Error + Debug + Send + Sync + Into<ANNError>;

    /// An implementation of [`DistanceFunction`] for computing similarity between two
    /// equal sized slices of `Self`.
    type Distance: for<'a, 'b> DistanceFunction<&'a [Self], &'b [Self], f32> + Send + Sync + 'static;

    /// An implementation of [`PreprocessedDistanceFunciton`] for computing similarity
    /// between a fixed query and slices of `Self`.
    type QueryDistance: for<'a> PreprocessedDistanceFunction<&'a [Self], f32>
        + Send
        + Sync
        + 'static;

    /// Return the dimension of the vector when converted into a full-precision vector.
    ///
    /// For most implementations of `VectorRepr` this simply outputs the length of the input
    /// slice; however, for quantized vectors such as [`minmax::Data`] that can be used instead of
    /// flat full-precision vectors, the output of this might be different than the input length.
    fn full_dimension(vec: &[Self]) -> Result<usize, Self::Error>;

    /// Return a [`DistanceFunction`] that computes distances between equal sized slices
    /// of `Self`.
    ///
    /// If `dim` is provided, then the returned implementation *may* be
    /// specialized for this dimension. Invoking the returned `DistanceFunction` may panic
    /// if incorrectly sized slices are provided.
    ///
    /// If `dim` is not provided, then the resulting [`DistanceFunction`] can be invoked
    /// for all distances of slices.
    fn distance(metric: Metric, dim: Option<usize>) -> Self::Distance;

    /// Return a [`PreprocessedDistanceFunction`] for the provided `query`.
    ///
    /// This may perform some pre-processing on `query` to enable more efficient
    /// computations.
    fn query_distance(query: &[Self], metric: Metric) -> Self::QueryDistance;

    /// Return an object that efficiently dereferences to a `&[f32]`.
    ///
    /// When `Self` is `f32` - this function is a noop and references `data` directly.
    fn as_f32(data: &[Self]) -> Result<impl std::ops::Deref<Target = [f32]>, Self::Error>;

    /// Write the vector as a full-precision f32 vector into a destination slice.
    ///
    /// This function can throw an error if conversion to a full-precision f32 vector fails
    /// or if trhe dimensions do not match.
    fn as_f32_into(src: &[Self], dst: &mut [f32]) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Error)]
#[error("Unable to set full-precision vector of length {src} into slice of length {dst}")]
pub struct NativeTypeLengthError {
    src: usize,
    dst: usize,
}

impl From<NativeTypeLengthError> for ANNError {
    fn from(err: NativeTypeLengthError) -> ANNError {
        ANNError::log_index_error(format!(
            "Unable to set full-precision vector of length {} into slice of length {}",
            err.src, err.dst
        ))
    }
}

macro_rules! default_impl {
    (
        $T:ty,
        QueryDistance = $QueryDistance:ty,
        query_impl = $query_impl:expr,
        into_impl = $into_impl:expr
    ) => {
        impl VectorRepr for $T {
            type Error = NativeTypeLengthError;
            type Distance = diskann_vector::distance::Distance<$T, $T>;
            type QueryDistance = $QueryDistance;

            fn distance(metric: Metric, dim: Option<usize>) -> Self::Distance {
                <$T>::distance_comparer(metric, dim)
            }

            fn query_distance(query: &[$T], metric: Metric) -> Self::QueryDistance {
                ($query_impl)(query, metric)
            }

            fn full_dimension(v: &[Self]) -> Result<usize, Self::Error> {
                Ok(v.len())
            }

            fn as_f32(data: &[$T]) -> Result<impl std::ops::Deref<Target = [f32]>, Self::Error> {
                Ok(data.convert_f32())
            }

            fn as_f32_into(src: &[Self], dst: &mut [f32]) -> Result<(), Self::Error> {
                if dst.len() != src.len() {
                    return Err(NativeTypeLengthError{src: src.len(), dst: dst.len()});
                }
                ($into_impl)(src, dst);
                Ok(())
            }
        }
    };
    ($T:ty) => {
    default_impl!(
        $T,
        QueryDistance = BufferedDistance<$T>,
        query_impl = |query : &[$T], metric| {
            BufferedDistance::new(query.into(), metric)
        },
        into_impl = |src : &[$T], dst : &mut [f32]| {
            for (d, x) in dst.iter_mut().zip(src.iter()) {
                *d = (*x).into();
            }
        }
    );
    };
}

default_impl!(i8);
default_impl!(u8);

default_impl!(
    f32,
    QueryDistance = BufferedDistance<f32>,
    query_impl = |query: &[f32], metric| BufferedDistance::new(query.into(), metric),
    into_impl = |src : &[f32], dst: &mut [f32]| {
        dst.copy_from_slice(src);
    }
);

default_impl!(
    f16,
    QueryDistance = BufferedDistance<f16, f32>,
    query_impl = |query: &[f16], metric| {
        let mut converted: Box<[f32]> = (0..query.len()).map(|_| f32::default()).collect();
        converted.cast_from_slice(query);
        BufferedDistance::new(converted, metric)
    },
    into_impl = |src: &[f16], dst: &mut [f32]| {
        dst.cast_from_slice(src);
    }
);

///////////////
// Distances //
///////////////

/// An implementation of [`diskann_vector::PreprocessedDistanceFunction`] for full-precision
/// distances.
///
/// As a [`diskann_vector::PreprocessedDistanceFunction`], this implementation needs to work
/// in a standalone manner, meaning that we need to keep a copy of the query.
pub struct BufferedDistance<T, U = T>
where
    U: 'static,
    T: 'static,
{
    query: Box<[U]>,
    f: diskann_vector::distance::Distance<U, T>,
}

impl<T, U> BufferedDistance<T, U> {
    /// A short-cut constructor to create a new `BufferedDistance` when `T` has an appropriate
    /// [`diskann_vector::distance::DistanceProvider`].
    pub fn new(query: Box<[U]>, metric: Metric) -> Self
    where
        U: DistanceProvider<T>,
    {
        let dim = query.len();
        Self {
            query,
            f: U::distance_comparer(metric, Some(dim)),
        }
    }
}

impl<T, U> PreprocessedDistanceFunction<&[T]> for BufferedDistance<T, U> {
    #[inline(always)]
    fn evaluate_similarity(&self, x: &[T]) -> f32 {
        self.f.call(&self.query, x)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use diskann_vector::Half;

    use super::*;

    fn implements_vector_element<T: VectorElement>() -> bool {
        let _ = PhantomData::<T>;
        true
    }

    // Ensure we can construct our types of interest as `VectorElement`s.
    // If they cannot, then the call to `implements_vector_element` will fail and we don't
    // compile.
    #[test]
    fn test_vector_element() {
        assert!(implements_vector_element::<Half>());
        assert!(implements_vector_element::<f32>());
        // assert!(implements_vector_element::<f64>());

        assert!(implements_vector_element::<i8>());
        assert!(implements_vector_element::<i16>());
        // assert!(implements_vector_element::<i32>());
        // assert!(implements_vector_element::<i64>());

        assert!(implements_vector_element::<u8>());
        assert!(implements_vector_element::<u16>());
        // assert!(implements_vector_element::<u32>());
        // assert!(implements_vector_element::<u64>());
    }
}
