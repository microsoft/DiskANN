/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_utils::{views::MatrixView, ReborrowMut};
use diskann_vector::{
    distance::InnerProduct, norm::FastL2Norm, MathematicalValue, Norm, PureDistanceFunction,
};
#[cfg(feature = "flatbuffers")]
use flatbuffers::{FlatBufferBuilder, WIPOffset};
use rand::{Rng, RngCore};
use thiserror::Error;

use super::{
    CompensatedCosine, CompensatedIP, CompensatedSquaredL2, DataMeta, DataMetaError, DataMut,
    FullQueryMeta, FullQueryMut, QueryMeta, QueryMut, SupportedMetric,
};
#[cfg(feature = "flatbuffers")]
use crate::{
    algorithms::transforms::TransformError, flatbuffers::spherical, spherical::InvalidMetric,
};
use crate::{
    algorithms::{
        heap::SliceHeap,
        transforms::{NewTransformError, Transform, TransformFailed, TransformKind},
    },
    alloc::{Allocator, AllocatorError, GlobalAllocator, Poly, ScopedAllocator, TryClone},
    bits::{PermutationStrategy, Representation, Unsigned},
    num::Positive,
    utils::{compute_means_and_average_norm, compute_normalized_means, CannotBeEmpty},
    AsFunctor, CompressIntoWith,
};

///////////////
// Quantizer //
///////////////

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct SphericalQuantizer<A = GlobalAllocator>
where
    A: Allocator,
{
    /// The offset to apply to each vector.
    shift: Poly<[f32], A>,

    /// The [`SphericalQuantizer`] supports several different strategies for performing the
    /// distance-preserving transformation on dataset vectors, which may be applicable in
    /// different scenarios.
    ///
    /// The different transformations may have restrictions on the number of supported dimensions.
    /// While we will accept all non-zero input dimensions, the output dimension of a transform
    /// may be higher or lower, depending on the configuration.
    transform: Transform<A>,

    /// The metric meant to be used by the quantizer.
    metric: SupportedMetric,

    /// When processing queries, it may be beneficial to modify the query norm to match the
    /// dataset norm.
    ///
    /// This is only applicable when `InnerProduct` and `Cosine` are used, but serves to
    /// move the query into the dynamic range of the quantization.
    ///
    /// You would think that the normalization step in RabitQ would mitigate this, but
    /// that is not always right since range-adjustment happens before centering.
    mean_norm: Positive<f32>,

    /// To support 16-bit constants which have a limited dynamic range, we allow a
    /// pre-scaling parameter that is multiplied to each value in compressed vectors.
    ///
    /// This allows to transparent handling of compressing integral data, which can
    /// otherwise easily overflow `f16`.
    pre_scale: Positive<f32>,
}

impl<A> TryClone for SphericalQuantizer<A>
where
    A: Allocator,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        Ok(Self {
            shift: self.shift.try_clone()?,
            transform: self.transform.try_clone()?,
            metric: self.metric,
            mean_norm: self.mean_norm,
            pre_scale: self.pre_scale,
        })
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
pub enum TrainError {
    #[error("data dim cannot be zero")]
    DimCannotBeZero,
    #[error("data cannot be empty")]
    DataCannotBeEmpty,
    #[error("pre-scale must be positive")]
    PrescaleNotPositive,
    #[error("norm must be positive")]
    NormNotPositive,
    #[error("computed norm contains infinity or NaN")]
    NormNotFinite,
    #[error("reciprocal norm contains infinity or NaN")]
    ReciprocalNormNotFinite,
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

impl<A> SphericalQuantizer<A>
where
    A: Allocator,
{
    /// Return the number dimensions this quantizer has been trained for.
    pub fn input_dim(&self) -> usize {
        self.shift.len()
    }

    /// Return the dimension of the post-transformed vector.
    ///
    /// Output storage vectors should use this dimension instead of `self.dim()` because
    /// in general, the output dim **may** be different from the input dimension.
    pub fn output_dim(&self) -> usize {
        self.transform.output_dim()
    }

    /// Return the per-dimension shift vector.
    ///
    /// This vector is meant to accomplish two goals:
    ///
    /// 1. Centers the data around the training dataset mean.
    /// 2. Offsets each dimension into a range that can be encoded in unsigned values.
    pub fn shift(&self) -> &[f32] {
        &self.shift
    }

    /// Return the approximate mean norm of the training data.
    pub fn mean_norm(&self) -> Positive<f32> {
        self.mean_norm
    }

    /// Return the pre-scaling parameter for data. This value is multiplied to every
    /// compressed vector to adjust its dynamic range.
    ///
    /// A value of 1.0 means that no scaling is occurring.
    pub fn pre_scale(&self) -> Positive<f32> {
        self.pre_scale
    }

    /// Return a reference to the allocator used by this data structure.
    pub fn allocator(&self) -> &A {
        self.shift.allocator()
    }

    /// A lower-level constructor that accepts a centroid, mean norm, and pre-scale directly.
    pub fn generate(
        mut centroid: Poly<[f32], A>,
        mean_norm: f32,
        transform: TransformKind,
        metric: SupportedMetric,
        pre_scale: Option<f32>,
        rng: &mut dyn RngCore,
        allocator: A,
    ) -> Result<Self, TrainError> {
        let pre_scale = match pre_scale {
            Some(v) => Positive::new(v).map_err(|_| TrainError::PrescaleNotPositive)?,
            None => crate::num::POSITIVE_ONE_F32,
        };

        let dim = match NonZeroUsize::new(centroid.len()) {
            Some(dim) => dim,
            None => {
                return Err(TrainError::DimCannotBeZero);
            }
        };

        let mean_norm = Positive::new(mean_norm).map_err(|_| TrainError::NormNotPositive)?;

        // We passed in a 'rng' so `Transform::new` will not fail.
        let transform = match Transform::new(transform, dim, Some(rng), allocator.clone()) {
            Ok(v) => v,
            Err(NewTransformError::RngMissing(_)) => unreachable!("An Rng was provided"),
            Err(NewTransformError::AllocatorError(err)) => {
                return Err(TrainError::AllocatorError(err));
            }
        };

        // Transform the centroid by the pre-scale.
        centroid
            .iter_mut()
            .for_each(|v| *v *= pre_scale.into_inner());

        Ok(SphericalQuantizer {
            shift: centroid,
            transform,
            metric,
            mean_norm,
            pre_scale,
        })
    }

    /// Return the metric used by this quantizer.
    pub fn metric(&self) -> SupportedMetric {
        self.metric
    }

    /// Construct a quantizer for vectors in the distribution of `data`.
    ///
    /// The type of distance-preserving transform to use is selected by the [`TransformKind`].
    ///
    /// Vectors compressed with this quantizer will be **metric specific** and optimized for
    /// distance computations rather than reconstruction. This means that vectors compressed
    /// targeting the inner-product distance will not return meaningful results if used for
    /// L2 distance computations.
    ///
    /// Additionally, vectors compressed when using the [`SupportedMetric::Cosine`] distance
    /// will be implicitly normalized before being compressed to enable better compression.
    ///
    /// If argument `pre_scale` is given, then all vectors compressed by this quantizer will
    /// first be scaled by this value. Note that if given, `pre_scale` **must** be positive.
    pub fn train<T, R>(
        data: MatrixView<T>,
        transform: TransformKind,
        metric: SupportedMetric,
        pre_scale: PreScale,
        rng: &mut R,
        allocator: A,
    ) -> Result<Self, TrainError>
    where
        T: Copy + Into<f64> + Into<f32>,
        R: Rng,
    {
        // An inner implementation erasing the type of the random number generator to
        // cut down on excess monomorphization.
        #[inline(never)]
        fn train<T, A>(
            data: MatrixView<T>,
            transform: TransformKind,
            metric: SupportedMetric,
            pre_scale: PreScale,
            rng: &mut dyn RngCore,
            allocator: A,
        ) -> Result<SphericalQuantizer<A>, TrainError>
        where
            T: Copy + Into<f64> + Into<f32>,
            A: Allocator,
        {
            // This check is repeated in `Self::generate`, but we prefer to bail as early
            // as possible if we detect an error.
            if data.ncols() == 0 {
                return Err(TrainError::DimCannotBeZero);
            }

            let (centroid, mean_norm) = match metric {
                SupportedMetric::SquaredL2 | SupportedMetric::InnerProduct => {
                    compute_means_and_average_norm(data)
                }
                SupportedMetric::Cosine => (
                    compute_normalized_means(data)
                        .map_err(|_: CannotBeEmpty| TrainError::DataCannotBeEmpty)?,
                    1.0,
                ),
            };

            let mean_norm = mean_norm as f32;

            if mean_norm <= 0.0 {
                return Err(TrainError::NormNotPositive);
            }

            if !mean_norm.is_finite() {
                return Err(TrainError::NormNotFinite);
            }

            // Determining if (and how) the pre-scaling term will be calculated.
            let pre_scale: Positive<f32> = match pre_scale {
                PreScale::None => crate::num::POSITIVE_ONE_F32,
                PreScale::Some(v) => v,
                PreScale::ReciprocalMeanNorm => {
                    // We've checked that `mean_norm` is both positive and finite.
                    //
                    // It's possible that when converted to `f32`,
                    //
                    // Taking the reciprocal is well defined. However, since the norms
                    // and scales in `compute_means_and_average_norm` are done using `f64`,
                    // it's possible that the computed `mean_norm` is subnormal, leading
                    // to the reciprocal being infinity.
                    let pre_scale = Positive::new(1.0 / mean_norm)
                        .map_err(|_| TrainError::ReciprocalNormNotFinite)?;

                    if !pre_scale.into_inner().is_finite() {
                        return Err(TrainError::ReciprocalNormNotFinite);
                    }

                    pre_scale
                }
            };

            // Allow the pre-scaling to take place inside `Self::generate`.
            let centroid =
                Poly::from_iter(centroid.into_iter().map(|i| i as f32), allocator.clone())?;

            SphericalQuantizer::generate(
                centroid,
                mean_norm,
                transform,
                metric,
                Some(pre_scale.into_inner()),
                rng,
                allocator,
            )
        }

        train(data, transform, metric, pre_scale, rng, allocator)
    }

    /// Rescale the argument `v` to be in the rough dynamic range of the training dataset.
    pub fn rescale(&self, v: &mut [f32]) {
        let norm = FastL2Norm.evaluate(&*v);
        let m = self.mean_norm.into_inner() / norm;
        v.iter_mut().for_each(|i| *i *= m);
    }

    /// Private helper function to do common data pre-processing.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != self.dim()`.
    fn preprocess<'a>(
        &self,
        data: &[f32],
        allocator: ScopedAllocator<'a>,
    ) -> Result<Preprocessed<'a>, CompressionError> {
        assert_eq!(data.len(), self.input_dim(), "Data dimension is incorrect.");

        // Fold in pre-scaling with the potential norm corretion for cosine.
        //
        // NOTE: When we're computing Cosine Similarity, we normalize the vector. As such,
        // the `pre_scale` parameter become irrelvant since it just gets normalized away.
        let scale = self.pre_scale.into_inner();
        let mul: f32 = match self.metric {
            SupportedMetric::Cosine => {
                let norm: f32 = (FastL2Norm).evaluate(data);
                if norm == 0.0 {
                    1.0
                } else {
                    1.0 / norm
                }
            }
            SupportedMetric::SquaredL2 | SupportedMetric::InnerProduct => scale,
        };

        // Center the vector and compute the squared norm of the shifted vector.
        let shifted = Poly::from_iter(
            std::iter::zip(data.iter(), self.shift.iter()).map(|(&f, &s)| mul * f - s),
            allocator,
        )?;

        let shifted_norm = FastL2Norm.evaluate(&*shifted);
        if !shifted_norm.is_finite() {
            return Err(CompressionError::InputContainsNaN);
        }
        let inner_product_with_centroid = match self.metric {
            SupportedMetric::SquaredL2 => None,
            SupportedMetric::InnerProduct | SupportedMetric::Cosine => {
                let ip: MathematicalValue<f32> = InnerProduct::evaluate(&*shifted, &*self.shift);
                Some(ip.into_inner())
            }
        };

        Ok(Preprocessed {
            shifted,
            shifted_norm,
            inner_product_with_centroid,
        })
    }
}

/// Pre-scaling selector for spherical quantization training. Pre-scaling adjusts the
/// dynamic range of the data (usually decreasing it uniformly) to keep the correction terms
/// within the range expressible by 16-bit floating point numbers.
#[derive(Debug, Clone, Copy)]
pub enum PreScale {
    /// Do not use any pre-scaling.
    None,
    /// Pre-scale all data by the specified amount.
    Some(Positive<f32>),
    /// Heuristically estimate a pre-scaling parameter by using the inverse approximate
    /// mean norm. This will nearly normalize in-distribution vectors.
    ReciprocalMeanNorm,
}

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
#[derive(Debug, Clone, Error, PartialEq)]
#[non_exhaustive]
pub enum DeserializationError {
    #[error(transparent)]
    TransformError(#[from] TransformError),
    #[error("unrecognized flatbuffer identifier")]
    UnrecognizedIdentifier,
    #[error("transform length not equal to centroid")]
    DimMismatch,
    #[error("norm is missing or is not positive")]
    MissingNorm,
    #[error("pre-scale is missing or is not positive")]
    PreScaleNotPositive,

    #[error(transparent)]
    InvalidFlatBuffer(#[from] flatbuffers::InvalidFlatbuffer),

    #[error(transparent)]
    InvalidMetric(#[from] InvalidMetric),

    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
impl<A> SphericalQuantizer<A>
where
    A: Allocator + Clone,
{
    /// Pack `self` into `buf` using the [`spherical::SphericalQuantizer`] serialized
    /// representation.
    pub(crate) fn pack<'a, FA>(
        &self,
        buf: &mut FlatBufferBuilder<'a, FA>,
    ) -> WIPOffset<spherical::SphericalQuantizer<'a>>
    where
        FA: flatbuffers::Allocator + 'a,
    {
        // Save the centroid vector.
        let centroid = buf.create_vector(&self.shift);

        // Save the transform.
        let transform = self.transform.pack(buf);

        // Finish up.
        spherical::SphericalQuantizer::create(
            buf,
            &spherical::SphericalQuantizerArgs {
                centroid: Some(centroid),
                transform: Some(transform),
                metric: self.metric.into(),
                mean_norm: self.mean_norm.into_inner(),
                pre_scale: self.pre_scale.into_inner(),
            },
        )
    }

    /// Attempt to unpack `self` from a serialized [`spherical::SphericalQuantizer`]
    /// serialized representation, returning any encountered error.
    pub(crate) fn try_unpack(
        alloc: A,
        proto: spherical::SphericalQuantizer<'_>,
    ) -> Result<Self, DeserializationError> {
        let metric: SupportedMetric = proto.metric().try_into()?;

        // Unpack the centroid.
        let shift = Poly::from_iter(proto.centroid().into_iter(), alloc.clone())?;

        // Unpack the transform.
        let transform = Transform::try_unpack(alloc, proto.transform())?;

        // Ensure consistency between the shift dimensions and the transform.
        if shift.len() != transform.input_dim() {
            return Err(DeserializationError::DimMismatch);
        }

        // Make sure we get a sane value for the mean norm.
        let mean_norm =
            Positive::new(proto.mean_norm()).map_err(|_| DeserializationError::MissingNorm)?;

        let pre_scale = Positive::new(proto.pre_scale())
            .map_err(|_| DeserializationError::PreScaleNotPositive)?;

        Ok(Self {
            shift,
            transform,
            metric,
            mean_norm,
            pre_scale,
        })
    }
}

struct Preprocessed<'a> {
    shifted: Poly<[f32], ScopedAllocator<'a>>,
    shifted_norm: f32,
    inner_product_with_centroid: Option<f32>,
}

impl Preprocessed<'_> {
    /// Return the metric specific correction term as sumamrized below:
    ///
    /// * Inner Product: The inner product between the shifted vector and the centroid.
    /// * Squared L2: The squared norm of the shifted vector.
    fn metric_specific(&self) -> f32 {
        match self.inner_product_with_centroid {
            Some(ip) => ip,
            None => self.shifted_norm * self.shifted_norm,
        }
    }
}

///////////////////////
// Distance Functors //
///////////////////////

impl<A> AsFunctor<CompensatedSquaredL2> for SphericalQuantizer<A>
where
    A: Allocator,
{
    fn as_functor(&self) -> CompensatedSquaredL2 {
        CompensatedSquaredL2::new(self.output_dim())
    }
}

impl<A> AsFunctor<CompensatedIP> for SphericalQuantizer<A>
where
    A: Allocator,
{
    fn as_functor(&self) -> CompensatedIP {
        CompensatedIP::new(&self.shift, self.output_dim())
    }
}

impl<A> AsFunctor<CompensatedCosine> for SphericalQuantizer<A>
where
    A: Allocator,
{
    fn as_functor(&self) -> CompensatedCosine {
        CompensatedCosine::new(self.as_functor())
    }
}

/////////////////
// Compression //
/////////////////

#[derive(Debug, Error, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CompressionError {
    #[error("input contains NaN")]
    InputContainsNaN,

    #[error("expected source vector to have length {expected}")]
    SourceDimensionMismatch { expected: usize },

    #[error("expected destination vector to have length {expected}")]
    DestinationDimensionMismatch { expected: usize },

    #[error(
        "encoding error - you may need to scale the entire dataset to reduce its dynamic range"
    )]
    EncodingError(#[from] DataMetaError),

    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

fn check_dims(
    input: usize,
    output: usize,
    from: usize,
    into: usize,
) -> Result<(), CompressionError> {
    if from != input {
        return Err(CompressionError::SourceDimensionMismatch { expected: input });
    }
    if into != output {
        return Err(CompressionError::DestinationDimensionMismatch { expected: output });
    }
    Ok(())
}

/// Helper trait to dispatch to a faster 1-bit implementation and use the slower
/// maximum-cosine algorithm when more than 1 bit is used.
trait FinishCompressing {
    fn finish_compressing(
        &mut self,
        preprocessed: &Preprocessed<'_>,
        transformed: &[f32],
        transformed_norm: f32,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError>;
}

impl FinishCompressing for DataMut<'_, 1> {
    fn finish_compressing(
        &mut self,
        preprocessed: &Preprocessed<'_>,
        transformed: &[f32],
        transformed_norm: f32,
        _: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError> {
        // Compute signed quantized vector (-1 or 1)
        // and also populate the unsigned bit representation in `into` output vector.
        let mut quant_raw_inner_product = 0.0f32;
        let mut bit_sum = 0u32;
        transformed.iter().enumerate().for_each(|(i, &r)| {
            let bit: u8 = if r > 0.0 { 1 } else { 0 };

            quant_raw_inner_product += r.abs();
            bit_sum += <u8 as Into<u32>>::into(bit);

            // SAFETY: From check 1, we know that `i < into.len()`.
            unsafe { self.vector_mut().set_unchecked(i, bit) };
        });

        // The value we just computed for `quant_raw_inner_product` is:
        // ```
        // Y = <x', x> * sqrt(D)                        [1]
        // ```
        // The inner product correction term is
        // ```
        //       2 |X|
        // -----------------                            [2]
        // <x', x> * sqrt(D)
        // ```
        // [1] substitutes directly into [2] and we get
        // ```
        // 2 |X|
        // -----
        //   Y
        // ```
        // Therefore, the inner product correction term is
        // ```
        // 2.0 * shifted_norm / quant_raw_inner_product
        // ```
        let inner_product_correction =
            2.0 * transformed_norm * preprocessed.shifted_norm / quant_raw_inner_product;
        self.set_meta(DataMeta::new(
            inner_product_correction,
            preprocessed.metric_specific(),
            bit_sum,
        )?);
        Ok(())
    }
}

impl FinishCompressing for DataMut<'_, 2> {
    fn finish_compressing(
        &mut self,
        preprocessed: &Preprocessed<'_>,
        transformed: &[f32],
        transformed_norm: f32,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError> {
        compress_via_maximum_cosine(
            self.reborrow_mut(),
            preprocessed,
            transformed,
            transformed_norm,
            allocator,
        )
    }
}

impl FinishCompressing for DataMut<'_, 4> {
    fn finish_compressing(
        &mut self,
        preprocessed: &Preprocessed<'_>,
        transformed: &[f32],
        transformed_norm: f32,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError> {
        compress_via_maximum_cosine(
            self.reborrow_mut(),
            preprocessed,
            transformed,
            transformed_norm,
            allocator,
        )
    }
}

impl FinishCompressing for DataMut<'_, 8> {
    fn finish_compressing(
        &mut self,
        preprocessed: &Preprocessed<'_>,
        transformed: &[f32],
        transformed_norm: f32,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError> {
        compress_via_maximum_cosine(
            self.reborrow_mut(),
            preprocessed,
            transformed,
            transformed_norm,
            allocator,
        )
    }
}

//////////////////////
// Data Compression //
//////////////////////

impl<A> CompressIntoWith<&[f32], FullQueryMut<'_>, ScopedAllocator<'_>> for SphericalQuantizer<A>
where
    A: Allocator,
{
    type Error = CompressionError;

    /// Compress the input vector `from` into the bitslice `into`.
    ///
    /// # Error
    ///
    /// Returns an error if
    /// * The input contains `NaN`.
    /// * `from.len() != self.dim()`: Vector to be compressed must have the same
    ///   dimensionality as the quantizer.
    /// * `into.len() != self.output_dim()`: Compressed vector must have the same
    ///   dimensionality as the output of the distance-preserving transform. Importantely,
    ///   this **may** be different than `self.dim()` and should be retrieved from
    ///   `self.output_dim()`.
    fn compress_into_with(
        &self,
        from: &[f32],
        mut into: FullQueryMut<'_>,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), Self::Error> {
        let input_dim = self.shift.len();
        let output_dim = self.output_dim();
        check_dims(input_dim, output_dim, from.len(), into.len())?;

        let mut preprocessed = self.preprocess(from, allocator)?;

        // If the preprocessed norm is zero, then we tried to compress the center directly.
        // In this case, we can get the correct behavior by setting `into` to all zeros.
        if preprocessed.shifted_norm == 0.0 {
            into.vector_mut().fill(0.0);
            *into.meta_mut() = Default::default();
            return Ok(());
        }

        preprocessed
            .shifted
            .iter_mut()
            .for_each(|v| *v /= preprocessed.shifted_norm);

        // Transformation can fail due to OOM - we want to handle that gracefully.
        //
        // If the transformation fails because we provided the wrong sizes, that is a hard
        // program bug.
        #[expect(clippy::panic, reason = "the dimensions should already be as expected")]
        match self
            .transform
            .transform_into(into.vector_mut(), &preprocessed.shifted, allocator)
        {
            Ok(()) => {}
            Err(TransformFailed::AllocatorError(err)) => {
                return Err(CompressionError::AllocatorError(err))
            }
            Err(TransformFailed::SourceMismatch { .. })
            | Err(TransformFailed::DestinationMismatch { .. }) => {
                panic!(
                    "The sizes of these arrays should already be checked - this is a logic error"
                );
            }
        }

        *into.meta_mut() = FullQueryMeta {
            sum: into.vector().iter().sum::<f32>(),
            shifted_norm: preprocessed.shifted_norm,
            metric_specific: preprocessed.metric_specific(),
        };
        Ok(())
    }
}

impl<const NBITS: usize, A> CompressIntoWith<&[f32], DataMut<'_, NBITS>, ScopedAllocator<'_>>
    for SphericalQuantizer<A>
where
    A: Allocator,
    Unsigned: Representation<NBITS>,
    for<'a> DataMut<'a, NBITS>: FinishCompressing,
{
    type Error = CompressionError;

    /// Compress the input vector `from` into the bitslice `into`.
    ///
    /// # Error
    ///
    /// Returns an error if
    /// * The input contains `NaN`.
    /// * `from.len() != self.dim()`: Vector to be compressed must have the same
    ///   dimensionality as the quantizer.
    /// * `into.len() != self.output_dim()`: Compressed vector must have the same
    ///   dimensionality as the output of the distance-preserving transform. Importantely,
    ///   this **may** be different than `self.dim()` and should be retrieved from
    ///   `self.output_dim()`.
    fn compress_into_with(
        &self,
        from: &[f32],
        mut into: DataMut<'_, NBITS>,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), Self::Error> {
        let input_dim = self.shift.len();
        let output_dim = self.output_dim();
        check_dims(input_dim, output_dim, from.len(), into.len())?;

        let mut preprocessed = self.preprocess(from, allocator)?;

        if preprocessed.shifted_norm == 0.0 {
            into.set_meta(DataMeta::default());
            return Ok(());
        }

        let mut transformed = Poly::broadcast(0.0f32, output_dim, allocator)?;
        preprocessed
            .shifted
            .iter_mut()
            .for_each(|v| *v /= preprocessed.shifted_norm);

        // Transformation can fail due to OOM - we want to handle that gracefully.
        //
        // If the transformation fails because we provided the wrong sizes, that is a hard
        // program bug.
        #[expect(clippy::panic, reason = "the dimensions should already be as expected")]
        match self
            .transform
            .transform_into(&mut transformed, &preprocessed.shifted, allocator)
        {
            Ok(()) => {}
            Err(TransformFailed::AllocatorError(err)) => {
                return Err(CompressionError::AllocatorError(err))
            }
            Err(TransformFailed::SourceMismatch { .. })
            | Err(TransformFailed::DestinationMismatch { .. }) => {
                panic!(
                    "The sizes of these arrays should already be checked - this is a logic error"
                );
            }
        }

        let transformed_norm = if self.transform.preserves_norms() {
            1.0
        } else {
            (FastL2Norm).evaluate(&*transformed)
        };

        into.finish_compressing(&preprocessed, &transformed, transformed_norm, allocator)?;
        Ok(())
    }
}

struct AsNonZero<const NBITS: usize>;
impl<const NBITS: usize> AsNonZero<NBITS> {
    // Lint: Unwrap is being used in a const-context.
    #[allow(clippy::unwrap_used)]
    const NON_ZERO: NonZeroUsize = NonZeroUsize::new(NBITS).unwrap();
}

fn compress_via_maximum_cosine<const NBITS: usize>(
    mut data: DataMut<'_, NBITS>,
    preprocessed: &Preprocessed<'_>,
    transformed: &[f32],
    transformed_norm: f32,
    allocator: ScopedAllocator<'_>,
) -> Result<(), CompressionError>
where
    Unsigned: Representation<NBITS>,
{
    assert_eq!(data.len(), transformed.len());

    // Find the value we will use to multiply `transformed` to round it to the lattice
    // element that has the maximum cosine-similarity.
    let optimal_scale =
        maximize_cosine_similarity(transformed, AsNonZero::<NBITS>::NON_ZERO, allocator)?;

    let domain = Unsigned::domain_const::<NBITS>();
    let min = *domain.start() as f32;
    let max = *domain.end() as f32;
    let offset = max / 2.0;

    let mut self_inner_product = 0.0f32;
    let mut bit_sum = 0u32;
    for (i, t) in transformed.iter().enumerate() {
        let v = (*t * optimal_scale + offset).clamp(min, max).round();
        let dv = v - offset;
        self_inner_product = dv.mul_add(*t, self_inner_product);

        let v = v as u8;
        bit_sum += <u8 as Into<u32>>::into(v);

        // SAFETY: We have checked that `data.len() == transformed.len()`, so this access
        // is in-bounds.
        //
        // Further, by construction, `v` is encodable by the `Unsigned`.
        unsafe { data.vector_mut().set_unchecked(i, v) };
    }

    let shifted_norm = preprocessed.shifted_norm;
    let inner_product_correction = (transformed_norm * shifted_norm) / self_inner_product;
    data.set_meta(DataMeta::new(
        inner_product_correction,
        preprocessed.metric_specific(),
        bit_sum,
    )?);
    Ok(())
}

// This struct does 2 things:
//
// 1. Records the index in `v` and `rounded` and the scaling parameter so that
//   `value * v[position]` gets rounded to `rounded[position] + 1` while
//   `(value - epsilon) * v[position]` is rounded to `rounded[position]` for some small
//   epsilon.
//
//   Informally, "what's the smallest scaling factor so `v[position]` gets rounded to
//   the next value.
//
// 2. Imposes a total ordering on `f32` values so it can be used in a `BinaryHeap`.
//   Additionally, ordering is reverse so that `BinaryHeap` models a min-heap instead
//   of a max-heap.
#[derive(Debug, Clone, Copy)]
struct Pair {
    value: f32,
    position: u32,
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl Eq for Pair {}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .value
            .partial_cmp(&self.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// This is a tricky function - please read carefully.
///
/// Given a vector `v` compute the scaling factor `s` such that the cosine similarity
/// betwen `v` and `r` is maximized where `r` is defined as
/// ```math
/// let offset = (2^(num_bits) - 1) / 2;
/// let r = (s * v + offset).round().clamp(0, 2^num_bits - 1) - offset
/// ```
///
/// More informally, maximize the inner product between `v` and the points in a square
/// lattice with 2^num_bits values in each dimension, centered around zero. This latice
/// takes the values (+0.5, -0.5, +1.5, -1.5 ...) to give equal weight above and below zero.
///
/// It works by slowly increasing the factor `s` such that the rounding of only one
/// dimension in `v` is changed at a time. A running tally of the cosine similarity is
/// computed for each scaling factor until we've processed `D * 2^(num_bits - 1)` possible
/// scaling factors, where `D` is the length of `v`.
///
/// The best scaling factor is returned.
///
/// Refer to algorithm 1 in <https://arxiv.org/pdf/2409.09913>.
///
/// # Panics
///
/// Panics is `v.is_empty()`.
///
/// # Implementation details
///
/// We work with the absolute value of the elements in the vector `v`.
/// This does not affect the final result as the scaling works the same in both the
/// positive and negative directions but simplifies the book keeping.
fn maximize_cosine_similarity(
    v: &[f32],
    num_bits: NonZeroUsize,
    allocator: ScopedAllocator<'_>,
) -> Result<f32, AllocatorError> {
    // Initially, the lattice element has the value `0.5` for all dimensions.
    // This means the initial inner product between `v` and the rounded term is simply
    // `0.5 * sum(abs.(v))`. The absolute value is used because the latice element is
    // always in the direction of the components in `v`.
    let mut current_ip = 0.5 * v.iter().map(|i| i.abs() as f64).sum::<f64>();
    let mut current_square_norm = 0.25 * (v.len() as f64);

    // Book keeping for the current value of the rounded vector.
    // The true numeric value is 0.5 less than this (in the direction of `v`), but we use
    // integers for a smaller memory footprint.
    let mut rounded = Poly::broadcast(1u16, v.len(), allocator)?;

    // Compute the critical values and store them on a heap.
    //
    // The binary heap will keep track of the minimum critical value. Multiplying `v` by the
    // minimum critical value `s` means that `s * v` will only change `rounded` from its
    // current value at a single index (the position associated with `s`).
    let eps = 0.0001f32;
    let one_and_change = 1.0 + eps;
    let mut base = Poly::from_iter(
        v.iter().enumerate().map(|(position, value)| {
            let value = one_and_change / value.abs();
            Pair {
                value,
                position: position as u32,
            }
        }),
        allocator,
    )?;

    // Lint: This is a private method and all the callers have an invariant that they check
    // for non-empty inputs.
    #[allow(clippy::expect_used)]
    let mut critical_values =
        SliceHeap::new(&mut base).expect("calling code should not allow the slice to be empty");

    let mut max_similarity = f64::NEG_INFINITY;
    let mut optimal_scale = f32::default();
    let stop = (2usize).pow(num_bits.get() as u32 - 1) as u16;

    loop {
        let mut should_break = false;
        critical_values.update_root(|pair| {
            let Pair { value, position } = *pair;
            if value == f32::MAX {
                should_break = true;
                return;
            }

            let r = &mut rounded[position as usize];
            let vp = &v[position as usize];

            let old_r = *r;
            // By the nature of cricital values, only `r` will change in `rounded` when
            // multiplying by `value`. And that change will be to increase by 1.
            *r += 1;

            // The inner product estimate simply increases by `vp.abs()` because:
            //
            // * `r` is the only value in `rounded` that changes.
            // * `r` is increased by 1.
            current_ip += vp.abs() as f64;

            // This uses the formula
            // ```math
            // (x + 1)^2 - x^2 = x^2 + 2x + 1 - x^2
            //                 = 2x + 1
            // ```
            // substitute `x = y - 1/2` to obtain the true value associated with rounded and
            // we get
            // ```math
            // 2 ( y - 1/2 ) + 1 = 2y - 1 + 1
            //                   = 2y
            // ```
            // Therefore, the change in the estimate for the square norm of `rounded` is
            // `2 * old_r`.
            current_square_norm += (2 * old_r) as f64;

            // Compute the current cosine similarity and update max if needed.
            let similarity = current_ip / current_square_norm.sqrt();
            if similarity > max_similarity {
                max_similarity = similarity;
                optimal_scale = value;
            }

            // Compute the scaling factor that will change this dimension to the next value.
            if *r < stop {
                *pair = Pair {
                    value: (*r as f32 + eps) / vp.abs(),
                    position,
                };
            } else {
                *pair = Pair {
                    value: f32::MAX,
                    position,
                };
            }
        });
        if should_break {
            break;
        }
    }

    Ok(optimal_scale)
}

///////////////////////
// Query Compression //
///////////////////////

impl<const NBITS: usize, Perm, A>
    CompressIntoWith<&[f32], QueryMut<'_, NBITS, Perm>, ScopedAllocator<'_>>
    for SphericalQuantizer<A>
where
    Unsigned: Representation<NBITS>,
    Perm: PermutationStrategy<NBITS>,
    A: Allocator,
{
    type Error = CompressionError;

    /// Compress the input vector `from` into the bitslice `into`.
    ///
    /// # Error
    ///
    /// Returns an error if
    /// * The input contains `NaN`.
    /// * `from.len() != self.dim()`: Vector to be compressed must have the same
    ///   dimensionality as the quantizer.
    /// * `into.len() != self.output_dim()`: Compressed vector must have the same
    ///   dimensionality as the output of the distance-preserving transform. Importantely,
    ///   this **may** be different than `self.dim()` and should be retrieved from
    ///   `self.output_dim()`.
    fn compress_into_with(
        &self,
        from: &[f32],
        mut into: QueryMut<'_, NBITS, Perm>,
        allocator: ScopedAllocator<'_>,
    ) -> Result<(), Self::Error> {
        let input_dim = self.shift.len();
        let output_dim = self.output_dim();
        check_dims(input_dim, output_dim, from.len(), into.len())?;

        let mut preprocessed = self.preprocess(from, allocator)?;

        if preprocessed.shifted_norm == 0.0 {
            into.set_meta(QueryMeta::default());
            return Ok(());
        }

        preprocessed
            .shifted
            .iter_mut()
            .for_each(|v| *v /= preprocessed.shifted_norm);

        let mut transformed = Poly::broadcast(0.0f32, output_dim, allocator)?;

        // Transformation can fail due to OOM - we want to handle that gracefully.
        //
        // If the transformation fails because we provided the wrong sizes, that is a hard
        // program bug.
        #[expect(clippy::panic, reason = "the dimensions should already be as expected")]
        match self
            .transform
            .transform_into(&mut transformed, &preprocessed.shifted, allocator)
        {
            Ok(()) => {}
            Err(TransformFailed::AllocatorError(err)) => {
                return Err(CompressionError::AllocatorError(err))
            }
            Err(TransformFailed::SourceMismatch { .. })
            | Err(TransformFailed::DestinationMismatch { .. }) => {
                panic!(
                    "The sizes of these arrays should already be checked - this is a logic error"
                );
            }
        }

        // Compute the minimum and maximum values of the transformed vector.
        let (min, max) = transformed
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), i| {
                (i.min(min), i.max(max))
            });

        let domain = Unsigned::domain_const::<NBITS>();
        let lo = (*domain.start()) as f32;
        let hi = (*domain.end()) as f32;

        let scale = (max - min) / hi;
        let mut bit_sum: f32 = 0.0;
        transformed.iter().enumerate().for_each(|(i, v)| {
            let c = ((v - min) / scale).round().clamp(lo, hi);
            bit_sum += c;

            // Lint: We have verified that `into.len() == transformed.len()`, so the index
            // `i` is in bounds.
            //
            // Further, `c` has beem clamped to `[0, 2^NBITS - 1]` and is thus encodable
            // with the NBITS-bit unsigned representation.
            #[allow(clippy::unwrap_used)]
            into.vector_mut().set(i, c as i64).unwrap();
        });

        // Finish up the compensation terms.
        into.set_meta(QueryMeta {
            inner_product_correction: preprocessed.shifted_norm * scale,
            bit_sum,
            offset: min / scale,
            metric_specific: preprocessed.metric_specific(),
        });

        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(not(miri))]
#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Display;

    use diskann_utils::{
        lazy_format,
        views::{self, Matrix},
        ReborrowMut,
    };
    use diskann_vector::{norm::FastL2NormSquared, PureDistanceFunction};
    use diskann_wide::ARCH;
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };
    use rand_distr::StandardNormal;

    use crate::{
        algorithms::transforms::TargetDim,
        alloc::GlobalAllocator,
        bits::{BitTranspose, Dense},
        spherical::{Data, DataMetaF32, FullQuery, Query},
        test_util,
    };

    // Test cosine-similarity maximizer
    #[test]
    fn test_cosine_similarity_maximizer() {
        let mut rng = StdRng::seed_from_u64(0x070d9ff8cf5e0f8c);
        let num_trials = 10000;
        let num_bits = NonZeroUsize::new(3).unwrap();

        let scale_distribution = Uniform::new(0.5f32, 10.0f32).unwrap();

        let run_test = |target: [f32; 4]| {
            let scale =
                maximize_cosine_similarity(&target, num_bits, ScopedAllocator::global()).unwrap();

            let mut best: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
            let mut best_similarity: f32 = f32::NEG_INFINITY;

            // This crazy series of nested loops performs an exhaustive search over the
            // encoding space.
            let min = -3.5;
            for i0 in 0..8 {
                for i1 in 0..8 {
                    for i2 in 0..8 {
                        for i3 in 0..8 {
                            let p: [f32; 4] = [
                                min + (i0 as f32),
                                min + (i1 as f32),
                                min + (i2 as f32),
                                min + (i3 as f32),
                            ];

                            let sim: MathematicalValue<f32> =
                                diskann_vector::distance::Cosine::evaluate(&p, &target);
                            let sim = sim.into_inner();
                            if sim > best_similarity {
                                best_similarity = sim;
                                // Transform into an integer starting at zero.
                                best = p.map(|i| i - min);
                            }
                        }
                    }
                }
            }

            // Now, rescale the input vector, clamp, and round.
            // Check if they agree.
            let clamped = target.map(|i| (i * scale - min).round().clamp(0.0, 7.0));
            let clamped_cosine: MathematicalValue<f32> =
                diskann_vector::distance::Cosine::evaluate(&clamped.map(|i| i + min), &target);

            // We expect to either get the best value found via exhaustive search, or some
            // scalar multiple of it (since that will have the same cosine similarity).
            let passed = if best == clamped {
                true
            } else {
                let ratio: Vec<f32> = std::iter::zip(best, clamped)
                    .map(|(b, c)| {
                        let ratio = (b + min) / (c + min);
                        assert_ne!(
                            ratio, 0.0,
                            "ratio should never be zero because `b` is an integer and \
                             `min` is not"
                        );
                        ratio
                    })
                    .collect();

                ratio.iter().all(|i| *i == ratio[0])
            };

            if !passed {
                panic!(
                    "failed for input {:?}.\
                     Best = {:?}, Found = {:?}\
                     Best similarity = {}, similarity with clamped = {}",
                    target,
                    best,
                    clamped,
                    best_similarity,
                    clamped_cosine.into_inner()
                );
            }
        };

        // Run targeted tests.
        let min = -3.5;
        for i0 in (0..8).step_by(2) {
            for i1 in (1..9).step_by(2) {
                for i2 in (0..8).step_by(2) {
                    for i3 in (1..9).step_by(2) {
                        let p: [f32; 4] = [
                            min + (i0 as f32),
                            min + (i1 as f32),
                            min + (i2 as f32),
                            min + (i3 as f32),
                        ];
                        run_test(p)
                    }
                }
            }
        }

        for _ in 0..num_trials {
            let this_scale: f32 = scale_distribution.sample(&mut rng);
            let v: [f32; 4] = [(); 4].map(|_| {
                let v: f32 = StandardNormal {}.sample(&mut rng);
                this_scale * v
            });
            run_test(v);
        }
    }

    #[test]
    #[should_panic(expected = "calling code should not allow the slice to be empty")]
    fn empty_slice_panics() {
        maximize_cosine_similarity(
            &[],
            NonZeroUsize::new(4).unwrap(),
            ScopedAllocator::global(),
        )
        .unwrap();
    }

    struct Setup {
        transform: TransformKind,
        nrows: usize,
        ncols: usize,
        num_trials: usize,
    }

    fn get_scale(scale: PreScale, quantizer: &SphericalQuantizer) -> f32 {
        match scale {
            PreScale::None => 1.0,
            PreScale::Some(v) => v.into_inner(),
            PreScale::ReciprocalMeanNorm => 1.0 / quantizer.mean_norm().into_inner(),
        }
    }

    fn test_l2<const Q: usize, const D: usize, Perm>(
        setup: &Setup,
        problem: &test_util::TestProblem,
        computed_means: &[f32],
        pre_scale: PreScale,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<Q>,
        Unsigned: Representation<D>,
        Perm: PermutationStrategy<Q>,
        for<'a> SphericalQuantizer:
            CompressIntoWith<&'a [f32], DataMut<'a, D>, ScopedAllocator<'a>>,
        for<'a> SphericalQuantizer:
            CompressIntoWith<&'a [f32], QueryMut<'a, Q, Perm>, ScopedAllocator<'a>>,
    {
        assert_eq!(setup.nrows, problem.data.nrows());
        assert_eq!(setup.ncols, problem.data.ncols());

        let scoped_global = ScopedAllocator::global();
        let distribution = Uniform::new(0, setup.nrows).unwrap();
        let quantizer = SphericalQuantizer::train(
            problem.data.as_view(),
            setup.transform,
            SupportedMetric::SquaredL2,
            pre_scale,
            rng,
            GlobalAllocator,
        )
        .unwrap();

        let scale = get_scale(pre_scale, &quantizer);

        let mut b = Data::<D, _>::new_boxed(quantizer.output_dim());
        let mut q = Query::<Q, Perm, _>::new_boxed(quantizer.output_dim());
        let mut f = FullQuery::empty(quantizer.output_dim(), GlobalAllocator).unwrap();

        assert_eq!(
            quantizer.mean_norm.into_inner(),
            problem.mean_norm as f32,
            "computed mean norm should not apply scale"
        );
        let scaled_means: Vec<_> = computed_means.iter().map(|i| scale * i).collect();
        assert_eq!(&*scaled_means, quantizer.shift());

        let l2: CompensatedSquaredL2 = quantizer.as_functor();
        assert_eq!(l2.dim, quantizer.output_dim() as f32);

        for _ in 0..setup.num_trials {
            let i = distribution.sample(rng);
            let v = problem.data.row(i);

            quantizer
                .compress_into_with(v, b.reborrow_mut(), scoped_global)
                .unwrap();
            quantizer
                .compress_into_with(v, q.reborrow_mut(), scoped_global)
                .unwrap();
            quantizer
                .compress_into_with(v, f.reborrow_mut(), scoped_global)
                .unwrap();

            let shifted: Vec<f32> = std::iter::zip(v.iter(), quantizer.shift().iter())
                .map(|(a, b)| scale * a - b)
                .collect();

            // Check that the compensation coefficient were chosen correctly.
            {
                let DataMetaF32 {
                    inner_product_correction,
                    bit_sum,
                    metric_specific,
                } = b.meta().to_full(ARCH);

                let shifted_square_norm = metric_specific;

                // Check that the bit-count is correct. let bv = b.vector();
                let bv = b.vector();
                let s: usize = (0..bv.len()).map(|i| bv.get(i).unwrap() as usize).sum();
                assert_eq!(s, bit_sum as usize);

                // Check that the shifted norm is correct.
                {
                    let expected = FastL2NormSquared.evaluate(&*shifted);
                    let err = (shifted_square_norm - expected).abs() / expected.abs();
                    assert!(
                        err < 5.0e-4, // twice the minimum normal f16 value.
                        "failed diff check, got {}, expected {} - relative error = {}",
                        shifted_square_norm,
                        expected,
                        err
                    );
                }

                // Finaly, verify that the self-inner-product is clustered around 0.8 as
                // the RaBitQ paper suggests.
                if const { D == 1 } {
                    let self_inner_product = 2.0 * shifted_square_norm.sqrt()
                        / (inner_product_correction * (bv.len() as f32).sqrt());
                    assert!(
                        (self_inner_product - 0.8).abs() < 0.13,
                        "self inner-product should be close to 0.8. Instead, it's {}",
                        self_inner_product
                    );
                }
            }

            {
                let QueryMeta {
                    inner_product_correction,
                    bit_sum,
                    offset,
                    metric_specific,
                } = q.meta();

                let shifted_square_norm = metric_specific;
                let mut preprocessed = quantizer.preprocess(v, scoped_global).unwrap();
                preprocessed
                    .shifted
                    .iter_mut()
                    .for_each(|i| *i /= preprocessed.shifted_norm);

                let mut transformed = vec![0.0f32; quantizer.output_dim()];
                quantizer
                    .transform
                    .transform_into(&mut transformed, &preprocessed.shifted, scoped_global)
                    .unwrap();

                let min = transformed.iter().fold(f32::MAX, |min, &i| min.min(i));
                let max = transformed.iter().fold(f32::MIN, |max, &i| max.max(i));

                let scale = (max - min) / ((2usize.pow(Q as u32) - 1) as f32);

                // Shifted Norm
                {
                    let expected = FastL2NormSquared.evaluate(&*shifted);
                    let err = (shifted_square_norm - expected).abs() / expected.abs();
                    assert!(
                        err < 2e-7,
                        "failed diff check, got {}, expected {} - relative error = {}",
                        shifted_square_norm,
                        expected,
                        err
                    );
                }

                // Inner product correction
                {
                    let expected = shifted_square_norm.sqrt() * scale;
                    let got = inner_product_correction;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"innerproduct_scale\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Offset
                {
                    let expected = min / scale;
                    let got = offset;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"sum_scale\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Bit Sum
                {
                    let expected = (0..q.len())
                        .map(|i| q.vector().get(i).unwrap())
                        .sum::<i64>() as f32;

                    let got = bit_sum;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"offset\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }
            }

            // Check that the compensation coefficient were chosen correctly.
            {
                // Check that the bit-count is correct.
                let s: f32 = f.data.iter().sum::<f32>();
                assert_eq!(s, f.meta.sum);

                // Check that the shifted norm is correct.
                {
                    let expected = FastL2Norm.evaluate(&*shifted);
                    let err = (f.meta.shifted_norm - expected).abs() / expected.abs();
                    assert!(
                        err < 2e-7,
                        "failed diff check, got {}, expected {} - relative error = {}",
                        f.meta.shifted_norm,
                        expected,
                        err
                    );
                }

                assert_eq!(
                    f.meta.metric_specific,
                    f.meta.shifted_norm * f.meta.shifted_norm,
                    "metric specific data for squared l2 is the square shifted norm",
                );
            }
        }

        // Finally - test that if we compress the centroid, the metadata coefficients get
        // zeroed correctly.
        quantizer
            .compress_into_with(computed_means, b.reborrow_mut(), scoped_global)
            .unwrap();
        assert_eq!(b.meta(), DataMeta::default());

        quantizer
            .compress_into_with(computed_means, q.reborrow_mut(), scoped_global)
            .unwrap();
        assert_eq!(q.meta(), QueryMeta::default());

        f.data.fill(f32::INFINITY);
        quantizer
            .compress_into_with(computed_means, f.reborrow_mut(), scoped_global)
            .unwrap();
        assert!(f.data.iter().all(|&i| i == 0.0));
        assert_eq!(f.meta.sum, 0.0);
        assert_eq!(f.meta.metric_specific, 0.0);
    }

    fn test_ip<const Q: usize, const D: usize, Perm>(
        setup: &Setup,
        problem: &test_util::TestProblem,
        computed_means: &[f32],
        pre_scale: PreScale,
        rng: &mut StdRng,
        ctx: &dyn Display,
    ) where
        Unsigned: Representation<Q>,
        Unsigned: Representation<D>,
        Perm: PermutationStrategy<Q>,
        for<'a> SphericalQuantizer:
            CompressIntoWith<&'a [f32], DataMut<'a, D>, ScopedAllocator<'a>>,
        for<'a> SphericalQuantizer:
            CompressIntoWith<&'a [f32], QueryMut<'a, Q, Perm>, ScopedAllocator<'a>>,
    {
        assert_eq!(setup.nrows, problem.data.nrows());
        assert_eq!(setup.ncols, problem.data.ncols());

        let scoped_global = ScopedAllocator::global();
        let distribution = Uniform::new(0, setup.nrows).unwrap();
        let quantizer = SphericalQuantizer::train(
            problem.data.as_view(),
            setup.transform,
            SupportedMetric::InnerProduct,
            pre_scale,
            rng,
            GlobalAllocator,
        )
        .unwrap();

        let scale = get_scale(pre_scale, &quantizer);

        let mut b = Data::<D, _>::new_boxed(quantizer.output_dim());
        let mut q = Query::<Q, Perm, _>::new_boxed(quantizer.output_dim());
        let mut f = FullQuery::empty(quantizer.output_dim(), GlobalAllocator).unwrap();

        assert_eq!(
            quantizer.mean_norm.into_inner(),
            problem.mean_norm as f32,
            "computed mean norm should not apply scale"
        );
        let scaled_means: Vec<_> = computed_means.iter().map(|i| scale * i).collect();
        assert_eq!(&*scaled_means, quantizer.shift());

        let ip: CompensatedIP = quantizer.as_functor();

        assert_eq!(ip.dim, quantizer.output_dim() as f32);
        assert_eq!(
            ip.squared_shift_norm,
            FastL2NormSquared.evaluate(quantizer.shift())
        );

        for _ in 0..setup.num_trials {
            let i = distribution.sample(rng);
            let v = problem.data.row(i);

            quantizer
                .compress_into_with(v, b.reborrow_mut(), scoped_global)
                .unwrap();
            quantizer
                .compress_into_with(v, q.reborrow_mut(), scoped_global)
                .unwrap();
            quantizer
                .compress_into_with(v, f.reborrow_mut(), scoped_global)
                .unwrap();

            let shifted: Vec<f32> = std::iter::zip(v.iter(), quantizer.shift().iter())
                .map(|(a, b)| scale * a - b)
                .collect();

            // Check that the compensation coefficient were chosen correctly.
            {
                let DataMetaF32 {
                    inner_product_correction,
                    bit_sum,
                    metric_specific,
                } = b.meta().to_full(ARCH);

                let inner_product_with_centroid = metric_specific;

                // Check that the bit-count is correct.
                let bv = b.vector();
                let s: usize = (0..bv.len()).map(|i| bv.get(i).unwrap() as usize).sum();
                assert_eq!(s, bit_sum as usize);

                // Check that the shifted norm is correct.
                let inner_product: MathematicalValue<f32> =
                    InnerProduct::evaluate(&*shifted, quantizer.shift());

                let diff = (inner_product.into_inner() - inner_product_with_centroid).abs();
                assert!(
                    diff < 1.53e-5,
                    "got a diff of {}. Expected = {}, got = {} -- context: {}",
                    diff,
                    inner_product.into_inner(),
                    inner_product_with_centroid,
                    ctx,
                );

                // Finaly, verify that the self-inner-product is clustered around 0.8 as
                // the RaBitQ paper suggests.
                if const { D == 1 } {
                    let self_inner_product = 2.0 * (FastL2Norm).evaluate(&*shifted)
                        / (inner_product_correction * (bv.len() as f32).sqrt());
                    assert!(
                        (self_inner_product - 0.8).abs() < 0.12,
                        "self inner-product should be close to 0.8. Instead, it's {}",
                        self_inner_product
                    );
                }
            }

            {
                let QueryMeta {
                    inner_product_correction,
                    bit_sum,
                    offset,
                    metric_specific,
                } = q.meta();

                let inner_product_with_centroid = metric_specific;
                let mut preprocessed = quantizer.preprocess(v, scoped_global).unwrap();
                preprocessed
                    .shifted
                    .iter_mut()
                    .for_each(|i| *i /= preprocessed.shifted_norm);

                let mut transformed = vec![0.0f32; quantizer.output_dim()];
                quantizer
                    .transform
                    .transform_into(&mut transformed, &preprocessed.shifted, scoped_global)
                    .unwrap();

                let min = transformed.iter().fold(f32::MAX, |min, &i| min.min(i));
                let max = transformed.iter().fold(f32::MIN, |max, &i| max.max(i));

                let scale = (max - min) / ((2usize.pow(Q as u32) - 1) as f32);

                // Inner product correction
                {
                    let expected = (FastL2Norm).evaluate(&*shifted) * scale;
                    let got = inner_product_correction;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"innerproduct_scale\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Offset
                {
                    let expected = min / scale;
                    let got = offset;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"sum_scale\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Bit Sum
                {
                    let expected = (0..q.len())
                        .map(|i| q.vector().get(i).unwrap())
                        .sum::<i64>() as f32;

                    let got = bit_sum;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"offset\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Inner Product with Centroid
                {
                    // Check that the shifted norm is correct.
                    let inner_product: MathematicalValue<f32> =
                        InnerProduct::evaluate(&*shifted, quantizer.shift());
                    assert_eq!(inner_product.into_inner(), inner_product_with_centroid);
                }
            }

            // Check that the compensation coefficient were chosen correctly.
            {
                // Check that the bit-count is correct.
                let s: f32 = f.data.iter().sum::<f32>();
                assert_eq!(s, f.meta.sum);

                // Check that the shifted norm is correct.
                {
                    let expected = FastL2Norm.evaluate(&*shifted);
                    let err = (f.meta.shifted_norm - expected).abs() / expected.abs();
                    assert!(
                        err < 2e-7,
                        "failed diff check, got {}, expected {} - relative error = {}",
                        f.meta.shifted_norm,
                        expected,
                        err
                    );
                }

                // Check that the shifted norm is correct. s
                let inner_product: MathematicalValue<f32> =
                    InnerProduct::evaluate(&*shifted, quantizer.shift());
                assert_eq!(inner_product.into_inner(), f.meta.metric_specific,);
            }
        }

        // Finally - test that if we compress the centroid, the metadata coefficients get
        // zeroed correctly.
        quantizer
            .compress_into_with(computed_means, b.reborrow_mut(), scoped_global)
            .unwrap();
        assert_eq!(b.meta(), DataMeta::default());

        quantizer
            .compress_into_with(computed_means, q.reborrow_mut(), scoped_global)
            .unwrap();
        assert_eq!(q.meta(), QueryMeta::default());

        f.data.fill(f32::INFINITY);
        quantizer
            .compress_into_with(computed_means, f.reborrow_mut(), scoped_global)
            .unwrap();
        assert!(f.data.iter().all(|&i| i == 0.0));
        assert_eq!(f.meta.sum, 0.0);
        assert_eq!(f.meta.metric_specific, 0.0);
    }

    fn test_cosine<const Q: usize, const D: usize, Perm>(
        setup: &Setup,
        problem: &test_util::TestProblem,
        pre_scale: PreScale,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<Q>,
        Unsigned: Representation<D>,
        Perm: PermutationStrategy<Q>,
        for<'a> SphericalQuantizer:
            CompressIntoWith<&'a [f32], DataMut<'a, D>, ScopedAllocator<'a>>,
        for<'a> SphericalQuantizer:
            CompressIntoWith<&'a [f32], QueryMut<'a, Q, Perm>, ScopedAllocator<'a>>,
    {
        assert_eq!(setup.nrows, problem.data.nrows());
        assert_eq!(setup.ncols, problem.data.ncols());

        let scoped_global = ScopedAllocator::global();
        let distribution = Uniform::new(0, setup.nrows).unwrap();
        let quantizer = SphericalQuantizer::train(
            problem.data.as_view(),
            setup.transform,
            SupportedMetric::Cosine,
            pre_scale,
            rng,
            GlobalAllocator,
        )
        .unwrap();

        let mut b = Data::<D, _>::new_boxed(quantizer.output_dim());
        let mut q = Query::<Q, Perm, _>::new_boxed(quantizer.output_dim());
        let mut f = FullQuery::empty(quantizer.output_dim(), GlobalAllocator).unwrap();

        let cosine: CompensatedCosine = quantizer.as_functor();

        assert_eq!(cosine.inner.dim, quantizer.output_dim() as f32);
        assert_eq!(
            cosine.inner.squared_shift_norm,
            FastL2NormSquared.evaluate(quantizer.shift())
        );

        const IP_BOUND: f32 = 2.6e-3;

        let mut test_row = |v: &[f32]| {
            let vnorm = (FastL2Norm).evaluate(v);
            let v_normalized: Vec<f32> = v
                .iter()
                .map(|i| if vnorm == 0.0 { 0.0 } else { *i / vnorm })
                .collect();

            quantizer
                .compress_into_with(v, b.reborrow_mut(), scoped_global)
                .unwrap();

            quantizer
                .compress_into_with(v, q.reborrow_mut(), scoped_global)
                .unwrap();

            quantizer
                .compress_into_with(v, f.reborrow_mut(), scoped_global)
                .unwrap();

            let shifted: Vec<f32> = std::iter::zip(v_normalized.iter(), quantizer.shift().iter())
                .map(|(a, b)| a - b)
                .collect();

            // Check that the compensation coefficient were chosen correctly.
            {
                let DataMetaF32 {
                    inner_product_correction,
                    bit_sum,
                    metric_specific,
                } = b.meta().to_full(ARCH);

                let inner_product_with_centroid = metric_specific;

                // Check that the bit-count is correct.
                let bv = b.vector();
                let s: usize = (0..bv.len()).map(|i| bv.get(i).unwrap() as usize).sum();
                assert_eq!(s, bit_sum as usize);

                // Check that the shifted norm is correct. Since they are computed slightly
                // differnetly, allow a small amount of error.
                let inner_product: MathematicalValue<f32> =
                    InnerProduct::evaluate(&*shifted, quantizer.shift());

                let abs = (inner_product.into_inner() - inner_product_with_centroid).abs();
                let relative = abs / inner_product.into_inner().abs();

                assert!(
                    abs < 1e-7 || relative < IP_BOUND,
                    "got an abs/rel of {}/{} with a bound of {}/{}",
                    abs,
                    relative,
                    1e-7,
                    IP_BOUND
                );

                // Finaly, verify that the self-inner-product is clustered around 0.8 as
                // the RaBitQ paper suggests.
                if const { D == 1 } {
                    let self_inner_product = 2.0 * (FastL2Norm).evaluate(&*shifted)
                        / (inner_product_correction * (bv.len() as f32).sqrt());
                    assert!(
                        (self_inner_product - 0.8).abs() < 0.11,
                        "self inner-product should be close to 0.8. Instead, it's {}",
                        self_inner_product
                    );
                }
            }

            {
                let QueryMeta {
                    inner_product_correction,
                    bit_sum,
                    offset,
                    metric_specific,
                } = q.meta();

                let inner_product_with_centroid = metric_specific;
                let mut preprocessed = quantizer.preprocess(v, scoped_global).unwrap();
                preprocessed
                    .shifted
                    .iter_mut()
                    .for_each(|i| *i /= preprocessed.shifted_norm);

                let mut transformed = vec![0.0f32; quantizer.output_dim()];
                quantizer
                    .transform
                    .transform_into(&mut transformed, &preprocessed.shifted, scoped_global)
                    .unwrap();

                let min = transformed.iter().fold(f32::MAX, |min, &i| min.min(i));
                let max = transformed.iter().fold(f32::MIN, |max, &i| max.max(i));

                let scale = (max - min) / ((2usize.pow(Q as u32) - 1) as f32);

                // Inner product correction
                {
                    let expected = (FastL2Norm).evaluate(&*shifted) * scale;
                    let got = inner_product_correction;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"innerproduct_scale\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Offset
                {
                    let expected = min / scale;
                    let got = offset;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"sum_scale\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Bit Sum
                {
                    let expected = (0..q.len())
                        .map(|i| q.vector().get(i).unwrap())
                        .sum::<i64>() as f32;

                    let got = bit_sum;

                    let err = (expected - got).abs();
                    assert!(
                        err < 1.0e-7,
                        "\"offset\": expected {}, got {}, error = {}",
                        expected,
                        got,
                        err
                    );
                }

                // Inner Product with Centroid
                {
                    // Check that the shifted norm is correct.
                    let inner_product: MathematicalValue<f32> =
                        InnerProduct::evaluate(&*shifted, quantizer.shift());

                    let err = (inner_product.into_inner() - inner_product_with_centroid).abs()
                        / inner_product.into_inner().abs();
                    assert!(
                        err < IP_BOUND,
                        "\"offset\": expected {}, got {}, error = {}",
                        inner_product.into_inner(),
                        inner_product_with_centroid,
                        err
                    );
                }
            }

            // Check that the compensation coefficient were chosen correctly.
            {
                // Check that the bit-count is correct.
                let s: f32 = f.data.iter().sum::<f32>();
                assert_eq!(s, f.meta.sum);

                // Check that the shifted norm is correct.
                {
                    let expected = FastL2Norm.evaluate(&*shifted);
                    let err = (f.meta.shifted_norm - expected).abs() / expected.abs();
                    assert!(
                        err < 2e-7,
                        "failed diff check, got {}, expected {} - relative error = {}",
                        f.meta.shifted_norm,
                        expected,
                        err
                    );
                }

                // Check that the shifted norm is correct. s
                let inner_product: MathematicalValue<f32> =
                    InnerProduct::evaluate(&*shifted, quantizer.shift());
                let err = (inner_product.into_inner() - f.meta.metric_specific).abs()
                    / inner_product.into_inner().abs();
                assert!(
                    err < IP_BOUND,
                    "\"offset\": expected {}, got {}, error = {}",
                    inner_product.into_inner(),
                    f.meta.metric_specific,
                    err
                );
            }
        };

        for _ in 0..setup.num_trials {
            let i = distribution.sample(rng);
            let v = problem.data.row(i);
            test_row(v);
        }

        // Ensure that if a zero vector is provided that we do not divide by zero.
        let zero = vec![0.0f32; quantizer.input_dim()];
        test_row(&zero);
    }

    fn _test_oom_resiliance<T>(quantizer: &SphericalQuantizer, data: &[f32], dst: &mut T)
    where
        for<'a> T: ReborrowMut<'a>,
        for<'a> SphericalQuantizer: CompressIntoWith<
            &'a [f32],
            <T as ReborrowMut<'a>>::Target,
            ScopedAllocator<'a>,
            Error = CompressionError,
        >,
    {
        let mut succeeded = false;
        let mut failed = false;
        for max_allocations in 0..10 {
            match quantizer.compress_into_with(
                data,
                dst.reborrow_mut(),
                ScopedAllocator::new(&test_util::LimitedAllocator::new(max_allocations)),
            ) {
                Ok(()) => {
                    succeeded = true;
                }
                Err(CompressionError::AllocatorError(_)) => {
                    failed = true;
                }
                Err(other) => {
                    panic!("received an unexpected error: {:?}", other);
                }
            }
        }
        assert!(succeeded);
        assert!(failed);
    }

    fn test_oom_resiliance<const Q: usize, const D: usize, Perm>(
        setup: &Setup,
        problem: &test_util::TestProblem,
        pre_scale: PreScale,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<Q>,
        Unsigned: Representation<D>,
        Perm: PermutationStrategy<Q>,
        for<'a> SphericalQuantizer: CompressIntoWith<
            &'a [f32],
            DataMut<'a, D>,
            ScopedAllocator<'a>,
            Error = CompressionError,
        >,
        for<'a> SphericalQuantizer: CompressIntoWith<
            &'a [f32],
            QueryMut<'a, Q, Perm>,
            ScopedAllocator<'a>,
            Error = CompressionError,
        >,
    {
        assert_eq!(setup.nrows, problem.data.nrows());
        assert_eq!(setup.ncols, problem.data.ncols());

        let quantizer = SphericalQuantizer::train(
            problem.data.as_view(),
            setup.transform,
            SupportedMetric::SquaredL2,
            pre_scale,
            rng,
            GlobalAllocator,
        )
        .unwrap();

        // Data.
        let data = problem.data.row(0);
        _test_oom_resiliance::<Data<D, _>>(
            &quantizer,
            data,
            &mut Data::new_boxed(quantizer.output_dim()),
        );
        _test_oom_resiliance::<Query<Q, Perm, _>>(
            &quantizer,
            data,
            &mut Query::new_boxed(quantizer.output_dim()),
        );
        _test_oom_resiliance::<FullQuery<_>>(
            &quantizer,
            data,
            &mut FullQuery::empty(quantizer.output_dim(), GlobalAllocator).unwrap(),
        );
    }

    fn test_quantizer<const Q: usize, const D: usize, Perm>(setup: &Setup, rng: &mut StdRng)
    where
        Unsigned: Representation<Q>,
        Unsigned: Representation<D>,
        Perm: PermutationStrategy<Q>,
        for<'a> SphericalQuantizer: CompressIntoWith<
            &'a [f32],
            DataMut<'a, D>,
            ScopedAllocator<'a>,
            Error = CompressionError,
        >,
        for<'a> SphericalQuantizer: CompressIntoWith<
            &'a [f32],
            QueryMut<'a, Q, Perm>,
            ScopedAllocator<'a>,
            Error = CompressionError,
        >,
    {
        let problem = test_util::create_test_problem(setup.nrows, setup.ncols, rng);
        let computed_means_f32: Vec<_> = problem.means.iter().map(|i| *i as f32).collect();

        let scales = [
            PreScale::Some(Positive::new(1.0 / 1024.0).unwrap()),
            PreScale::Some(Positive::new(1.0 / 1024.0).unwrap()),
            PreScale::ReciprocalMeanNorm,
        ];

        for scale in scales {
            let ctx = &lazy_format!("dim = {}, scale = {:?}", setup.ncols, scale);

            test_l2::<Q, D, Perm>(setup, &problem, &computed_means_f32, scale, rng);
            test_ip::<Q, D, Perm>(setup, &problem, &computed_means_f32, scale, rng, ctx);
            test_cosine::<Q, D, Perm>(setup, &problem, scale, rng);
        }

        test_oom_resiliance::<Q, D, Perm>(setup, &problem, PreScale::ReciprocalMeanNorm, rng);
    }

    #[test]
    fn test_spherical_quantizer() {
        let mut rng = StdRng::seed_from_u64(0xab516aef1ce61640);
        for dim in [56, 72, 128, 255] {
            let setup = Setup {
                transform: TransformKind::PaddingHadamard {
                    target_dim: TargetDim::Same,
                },
                nrows: 64,
                ncols: dim,
                num_trials: 10,
            };

            test_quantizer::<4, 1, BitTranspose>(&setup, &mut rng);
            test_quantizer::<2, 2, Dense>(&setup, &mut rng);
            test_quantizer::<4, 4, Dense>(&setup, &mut rng);
            test_quantizer::<8, 8, Dense>(&setup, &mut rng);

            let setup = Setup {
                transform: TransformKind::DoubleHadamard {
                    target_dim: TargetDim::Same,
                },
                nrows: 64,
                ncols: dim,
                num_trials: 10,
            };
            test_quantizer::<4, 1, BitTranspose>(&setup, &mut rng);
            test_quantizer::<2, 2, Dense>(&setup, &mut rng);
            test_quantizer::<4, 4, Dense>(&setup, &mut rng);
            test_quantizer::<8, 8, Dense>(&setup, &mut rng);
        }
    }

    ////////////
    // Errors //
    ////////////

    #[test]
    fn err_dim_cannot_be_zero() {
        let data = Matrix::new(0.0f32, 10, 0);
        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let err = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            PreScale::None,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "data dim cannot be zero");
    }

    #[test]
    fn err_norm_must_be_positive() {
        let data = Matrix::new(0.0f32, 10, 10);
        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let err = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            PreScale::None,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "norm must be positive");
    }

    #[test]
    fn err_norm_cannot_be_infinity() {
        let mut data = Matrix::new(0.0f32, 10, 10);
        data[(2, 5)] = f32::INFINITY;

        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let err = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            PreScale::None,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "computed norm contains infinity or NaN");
    }

    #[test]
    fn err_reciprocal_norm_cannot_be_infinity() {
        let mut data = Matrix::new(0.0f32, 10, 10);
        data[(2, 5)] = 2.93863e-39;

        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let err = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            PreScale::ReciprocalMeanNorm,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "reciprocal norm contains infinity or NaN");
    }

    #[test]
    fn err_mean_norm_cannot_be_zero_generate() {
        let centroid = Poly::broadcast(0.0f32, 10, GlobalAllocator).unwrap();
        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let err = SphericalQuantizer::generate(
            centroid,
            0.0,
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            None,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "norm must be positive");
    }

    #[test]
    fn err_scale_cannot_be_zero_generate() {
        let centroid = Poly::broadcast(0.0f32, 10, GlobalAllocator).unwrap();
        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let err = SphericalQuantizer::generate(
            centroid,
            1.0,
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            Some(0.0),
            &mut rng,
            GlobalAllocator,
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "pre-scale must be positive");
    }

    #[test]
    fn compression_errors_data() {
        let mut rng = StdRng::seed_from_u64(0xe3e9f42ed9f15883);
        let data = Matrix::<f32>::new(views::Init(|| StandardNormal {}.sample(&mut rng)), 16, 12);

        let quantizer = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::PaddingHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            PreScale::None,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap();

        let scoped_global = ScopedAllocator::global();

        // Input contains NaN.
        {
            let mut query: Vec<f32> = quantizer.shift().to_vec();
            let mut d = Data::<1, _>::new_boxed(quantizer.output_dim());
            let mut q = Query::<4, BitTranspose, _>::new_boxed(quantizer.output_dim());

            for i in 0..query.len() {
                let last = query[i];
                for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    query[i] = v;

                    let err = quantizer
                        .compress_into_with(&*query, d.reborrow_mut(), scoped_global)
                        .unwrap_err();

                    assert_eq!(err.to_string(), "input contains NaN", "failed for {}", v);

                    let err = quantizer
                        .compress_into_with(&*query, q.reborrow_mut(), scoped_global)
                        .unwrap_err();

                    assert_eq!(err.to_string(), "input contains NaN", "failed for {}", v);
                }
                query[i] = last;
            }
        }

        // Input has a large value.
        {
            let query: Vec<f32> = vec![1000000.0; quantizer.input_dim()];
            let mut d = Data::<1, _>::new_boxed(quantizer.output_dim());

            let err = quantizer
                .compress_into_with(&*query, d.reborrow_mut(), scoped_global)
                .unwrap_err();

            let expected = "encoding error - you may need to scale the entire dataset to reduce its dynamic range";

            assert_eq!(err.to_string(), expected, "failed for {:?}", query);
        }

        // Input length
        for len in [quantizer.input_dim() - 1, quantizer.input_dim() + 1] {
            let query = vec![0.0f32; len];
            let mut d = Data::<1, _>::new_boxed(quantizer.output_dim());
            let mut q = Query::<4, BitTranspose, _>::new_boxed(quantizer.output_dim());

            let err = quantizer
                .compress_into_with(&*query, d.reborrow_mut(), scoped_global)
                .unwrap_err();
            assert_eq!(
                err,
                CompressionError::SourceDimensionMismatch {
                    expected: quantizer.input_dim(),
                }
            );

            let err = quantizer
                .compress_into_with(&*query, q.reborrow_mut(), scoped_global)
                .unwrap_err();
            assert_eq!(
                err,
                CompressionError::SourceDimensionMismatch {
                    expected: quantizer.input_dim(),
                }
            );
        }

        for len in [quantizer.output_dim() - 1, quantizer.output_dim() + 1] {
            let query = vec![0.0f32; quantizer.input_dim()];
            let mut d = Data::<1, _>::new_boxed(len);
            let mut q = Query::<4, BitTranspose, _>::new_boxed(len);

            let err = quantizer
                .compress_into_with(&*query, d.reborrow_mut(), scoped_global)
                .unwrap_err();
            assert_eq!(
                err,
                CompressionError::DestinationDimensionMismatch {
                    expected: quantizer.output_dim(),
                }
            );

            let err = quantizer
                .compress_into_with(&*query, q.reborrow_mut(), scoped_global)
                .unwrap_err();
            assert_eq!(
                err,
                CompressionError::DestinationDimensionMismatch {
                    expected: quantizer.output_dim(),
                }
            );
        }
    }

    #[test]
    fn centroid_scaling_happens_in_generate() {
        let centroid = Poly::from_iter(
            [1088.6732f32, 1393.32, 1547.877].into_iter(),
            GlobalAllocator,
        )
        .unwrap();
        let mean_norm = 2359.27;
        let pre_scale = 1.0 / mean_norm;

        let quantizer = SphericalQuantizer::generate(
            centroid,
            mean_norm,
            TransformKind::Null,
            SupportedMetric::InnerProduct,
            Some(pre_scale),
            &mut StdRng::seed_from_u64(10),
            GlobalAllocator,
        )
        .unwrap();

        let mut v = Data::<4, _>::new_boxed(quantizer.input_dim());
        let data: &[f32] = &[1000.34, 1456.32, 1234.5446];
        assert!(quantizer
            .compress_into_with(data, v.reborrow_mut(), ScopedAllocator::global())
            .is_ok(),
            "if this failed, the likely culprit is exceeding the value of the 16-bit correction terms"
            );
    }
}

#[cfg(feature = "flatbuffers")]
#[cfg(test)]
mod test_serialization {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algorithms::transforms::TargetDim,
        flatbuffers::{self as fb, to_flatbuffer},
        poly, test_util,
    };

    #[test]
    fn test_serialization_happy_path() {
        let mut rng = StdRng::seed_from_u64(0x070d9ff8cf5e0f8c);
        let problem = test_util::create_test_problem(10, 128, &mut rng);

        let low = NonZeroUsize::new(100).unwrap();
        let high = NonZeroUsize::new(150).unwrap();

        let kinds = [
            // Null
            TransformKind::Null,
            // Double Hadamard
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Natural,
            },
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Override(low),
            },
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Override(high),
            },
            // Padding Hadamard
            TransformKind::PaddingHadamard {
                target_dim: TargetDim::Same,
            },
            TransformKind::PaddingHadamard {
                target_dim: TargetDim::Natural,
            },
            TransformKind::PaddingHadamard {
                target_dim: TargetDim::Override(low),
            },
            TransformKind::PaddingHadamard {
                target_dim: TargetDim::Override(high),
            },
            // Random Rotation
            #[cfg(all(not(miri), feature = "linalg"))]
            TransformKind::RandomRotation {
                target_dim: TargetDim::Same,
            },
            #[cfg(all(not(miri), feature = "linalg"))]
            TransformKind::RandomRotation {
                target_dim: TargetDim::Natural,
            },
            #[cfg(all(not(miri), feature = "linalg"))]
            TransformKind::RandomRotation {
                target_dim: TargetDim::Override(low),
            },
            #[cfg(all(not(miri), feature = "linalg"))]
            TransformKind::RandomRotation {
                target_dim: TargetDim::Override(high),
            },
        ];

        let pre_scales = [
            PreScale::None,
            PreScale::Some(Positive::new(0.5).unwrap()),
            PreScale::Some(Positive::new(1.0).unwrap()),
            PreScale::Some(Positive::new(1.5).unwrap()),
            PreScale::ReciprocalMeanNorm,
        ];

        let alloc = GlobalAllocator;
        for kind in kinds.into_iter() {
            for metric in SupportedMetric::all() {
                for pre_scale in pre_scales {
                    let quantizer = SphericalQuantizer::train(
                        problem.data.as_view(),
                        kind,
                        metric,
                        pre_scale,
                        &mut rng,
                        alloc,
                    )
                    .unwrap();

                    let data = to_flatbuffer(|buf| quantizer.pack(buf));
                    let proto =
                        flatbuffers::root::<fb::spherical::SphericalQuantizer>(&data).unwrap();
                    let reloaded = SphericalQuantizer::try_unpack(alloc, proto).unwrap();
                    assert_eq!(quantizer, reloaded, "failed on transform {:?}", kind);
                }
            }
        }
    }

    #[test]
    fn test_error_checking() {
        let mut rng = StdRng::seed_from_u64(0x070d9ff8cf5e0f8c);
        let problem = test_util::create_test_problem(10, 128, &mut rng);

        let transform = TransformKind::DoubleHadamard {
            target_dim: TargetDim::Same,
        };

        let alloc = GlobalAllocator;
        let mut make_quantizer = || {
            SphericalQuantizer::train(
                problem.data.as_view(),
                transform,
                SupportedMetric::SquaredL2,
                PreScale::None,
                &mut rng,
                alloc,
            )
            .unwrap()
        };

        type E = DeserializationError;

        // Missing norm: 0.0
        {
            let mut quantizer = make_quantizer();
            // SAFETY: We do not do anything with the created value and the compiler
            // does not know about the layout of `Positive`, so we don't need to worry
            // about violating layout restrictions.
            quantizer.mean_norm = unsafe { Positive::new_unchecked(0.0) };

            let data = to_flatbuffer(|buf| quantizer.pack(buf));
            let proto = flatbuffers::root::<fb::spherical::SphericalQuantizer>(&data).unwrap();
            let err = SphericalQuantizer::try_unpack(alloc, proto).unwrap_err();
            assert_eq!(err, E::MissingNorm);
        }

        // Missing norm: negative
        {
            let mut quantizer = make_quantizer();

            // SAFETY: We do not do anything with the created value and the compiler
            // does not know about the layout of `Positive`, so we don't need to worry
            // about violating layout restrictions.
            quantizer.mean_norm = unsafe { Positive::new_unchecked(-1.0) };

            let data = to_flatbuffer(|buf| quantizer.pack(buf));
            let proto = flatbuffers::root::<fb::spherical::SphericalQuantizer>(&data).unwrap();
            let err = SphericalQuantizer::try_unpack(alloc, proto).unwrap_err();
            assert_eq!(err, E::MissingNorm);
        }

        // PreScaleNotPositive
        {
            let mut quantizer = make_quantizer();

            // SAFETY: This really isn't safe, but we are not using the improper value in a
            // way that will trigger undefined behavior.
            quantizer.pre_scale = unsafe { Positive::new_unchecked(0.0) };

            let data = to_flatbuffer(|buf| quantizer.pack(buf));
            let proto = flatbuffers::root::<fb::spherical::SphericalQuantizer>(&data).unwrap();
            let err = SphericalQuantizer::try_unpack(alloc, proto).unwrap_err();
            assert_eq!(err, E::PreScaleNotPositive);
        }

        // Dim Mismatch.
        {
            let mut quantizer = make_quantizer();
            quantizer.shift = poly!([1.0, 2.0, 3.0], alloc).unwrap();

            let data = to_flatbuffer(|buf| quantizer.pack(buf));
            let proto = flatbuffers::root::<fb::spherical::SphericalQuantizer>(&data).unwrap();
            let err = SphericalQuantizer::try_unpack(alloc, proto).unwrap_err();
            assert_eq!(err, E::DimMismatch);
        }
    }
}
