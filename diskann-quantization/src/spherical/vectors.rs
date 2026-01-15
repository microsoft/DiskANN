/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Spherical Quantization Distance Functions
//!
//! ## Inner Product - 1-bit Symmetric
//!
//! Key:
//! * `X'` (Upper case with "prime"): The original, full-precision vectors.
//! * `C`: The dataset centroid.
//! * `X` (Upper case, no "prime"): The result of `X' - C`. That is, the centered vectors.
//! * `|X|`: The L2 norm of a vector.
//! * `x` (Lower case): The normalized version of a vector `X` respectively.
//! * `x'`: The quantized reconstruction of `x`, computed as `x = T(x!)` where
//!
//!   - `x!` is the binary encoded vector in `{-1/sqrt(dim), +1/sqrt(dim)}^dim`.
//!   - `x -> T(x)` is the distance-preserving transformation.
//!
//! ```math
//! <X', Y'> = <X + C, Y + C>
//!          = <X, Y> + <X, C> + <Y, C> + <C, C>
//!          = |X| |Y| <x, y> + <X, C> + <Y, C> + |C|^2
//!                    ------
//!                      |
//!                 Normalized
//!                 Components
//! ```
//!
//! Now, working with the normalized components:
//! ```math
//! <x, y> \approx <x', y'> / (<x', x> * <y', y>)         [From the RabitQ Paper]
//!                            -------   -------
//!                               |         |
//!                            Self Dot  Self Dot
//!                            Product    Product
//! ```
//! Where `x'` and `y'` are transformed vectors in the domain `{-1/sqrt(D), 1/sqrt(D)}^D`.
//!
//! This is the result from the RabitQ paper (though modified to work on two symmetrically
//! compressed vectors).
//!
//! NOTE: The symmetric correction factor gives incorrect estimates for estimating the
//! distance between a vector and itself because the term `<x', x>` is strictly less
//! than one, bringing the estimate for the inner product `<x, x>` to a value greater than 1.
//! In practice, this still yields better recall (both exhaustive and via graph build) than
//! no correction, so we keep it.
//!
//! Finally, to compute the inner product `<x', y'>` we use the following general approach:
//! ```math
//! <x', y'> = <a * (bx + b), c * (by + d)>
//!          = (a * b) ( <bx, by> + b*sum(by) + d*sum(bx) + b*d )
//!            -------   --------   - -------   - -------
//!               |         |       |    |      |    |
//!            Scaling      |       | Bit Sum   | Bit Sum
//!             Terms       |       |           |
//!                         |       |        y offset
//!                    Bit Inner    |
//!                     Product     |
//!                              x offset
//! ```
//!
//! When the vectors `x` and `y` use the same scaling or offset terms, some of this
//! computation cam be simplified. However, spherical quantization allows queries to use
//! a different compression (i.e., scalar quantization) and so this term reflects the
//! general strategy.
//!
//! Thus, for each vector `X`, we need the following compensation values:
//!
//! 1. `|X| * a / <x', x>`: The norm of `X'` after it has been shifted by the centroid
//!    multiplied by the quantization scaling parameter, divided by the correction term.
//!    This whole expression is multiplied to the the result of the inner product term to
//!    obtain the full-norm estimate of the shifted inner product.
//!
//! 2. `<X, C>`: The inner product between the shifted vector and the centroid.
//!
//! 3. `sum(bx)`: The sum of the bits in the binary vector representation of `x'`.
//!
//! 4. `|X|`: The norm of the shifted vector - used to computed L2 distances.
//!
//! ## Squared L2 - 1-bit Symmetric
//!
//! ```math
//! |X' - Y'| = | (X' - C) - (Y' - C) |
//!           = | X - Y |
//!           = |X|^2 + |Y|^2 - 2 <X, Y>
//!           = |X|^2 + |Y|^2 - 2 |X| |Y| <x, y>
//!                                       ------
//!                                         |
//!                              Reuse from Inner Product
//! ```
//!
//! The compensation terms used here are the same as the same.
//!
//! # Full Precision Queries
//!
//! When the vector `Y` is full-precision, the expression for the inner product becomes
//! ```math
//! <a(X + b), Y> = a (<X, Y> + b * sum(Y))
//! ```
//!
//! # Dev Notes
//!
//! The functions implemented here use the [`diskann_wide::arch::Target2`] interface to
//! propagate micro-architecture defails from the caller.
//!
//! When calling implementations in [`crate::bits::distances]`, be sure to use
//! [`diskann_wide::Architecture::run2`] instead to invoke the distance functions. This will
//! architecture specific
//! [target features](https://rust-lang.github.io/rfcs/2045-target-feature.html) are
//! inhereted properly, even if these functions are not inlined.

use diskann_utils::{Reborrow, ReborrowMut};
use diskann_vector::{norm::FastL2NormSquared, Norm};
use diskann_wide::{arch::Target2, Architecture};
use half::f16;
use thiserror::Error;

#[cfg(feature = "flatbuffers")]
use crate::flatbuffers as fb;
use crate::{
    alloc::{AllocatorCore, AllocatorError, Poly},
    bits::{BitSlice, Dense, PermutationStrategy, Representation, Unsigned},
    distances::{self, InnerProduct, MV},
    meta,
};

//////////////////////
// Supported Metric //
//////////////////////

/// The metrics that are supported by [`crate::spherical::SphericalQuantizer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedMetric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

#[cfg(test)]
impl SupportedMetric {
    fn pick(self, shifted_norm: f32, inner_product_with_centroid: f32) -> f32 {
        match self {
            Self::SquaredL2 => shifted_norm * shifted_norm,
            Self::InnerProduct | Self::Cosine => inner_product_with_centroid,
        }
    }

    #[cfg(feature = "flatbuffers")]
    pub(super) fn all() -> [Self; 3] {
        [Self::SquaredL2, Self::InnerProduct, Self::Cosine]
    }
}

impl TryFrom<diskann_vector::distance::Metric> for SupportedMetric {
    type Error = UnsupportedMetric;
    fn try_from(metric: diskann_vector::distance::Metric) -> Result<Self, Self::Error> {
        use diskann_vector::distance::Metric;
        match metric {
            Metric::L2 => Ok(Self::SquaredL2),
            Metric::InnerProduct => Ok(Self::InnerProduct),
            Metric::Cosine => Ok(Self::Cosine),
            unsupported => Err(UnsupportedMetric(unsupported)),
        }
    }
}

impl PartialEq<diskann_vector::distance::Metric> for SupportedMetric {
    fn eq(&self, metric: &diskann_vector::distance::Metric) -> bool {
        match Self::try_from(*metric) {
            Ok(m) => *self == m,
            Err(_) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("metric {0:?} is not supported for spherical quantization")]
pub struct UnsupportedMetric(pub(crate) diskann_vector::distance::Metric);

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[error("the value {0} is not recognized as a supported metric")]
pub struct InvalidMetric(i8);

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
impl TryFrom<fb::spherical::SupportedMetric> for SupportedMetric {
    type Error = InvalidMetric;
    fn try_from(value: fb::spherical::SupportedMetric) -> Result<Self, Self::Error> {
        match value {
            fb::spherical::SupportedMetric::SquaredL2 => Ok(Self::SquaredL2),
            fb::spherical::SupportedMetric::InnerProduct => Ok(Self::InnerProduct),
            fb::spherical::SupportedMetric::Cosine => Ok(Self::Cosine),
            unsupported => Err(InvalidMetric(unsupported.0)),
        }
    }
}

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
impl From<SupportedMetric> for fb::spherical::SupportedMetric {
    fn from(value: SupportedMetric) -> Self {
        match value {
            SupportedMetric::SquaredL2 => fb::spherical::SupportedMetric::SquaredL2,
            SupportedMetric::InnerProduct => fb::spherical::SupportedMetric::InnerProduct,
            SupportedMetric::Cosine => fb::spherical::SupportedMetric::Cosine,
        }
    }
}

//////////
// Data //
//////////

/// Metadata for correcting quantization for computing distances among quant vectors.
#[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct DataMeta {
    /// This is the whole term
    /// ```math
    /// |X| * a / <x', x>
    /// ```
    /// and represents the entires correction factor for computing inner products on the
    /// representation
    /// ```math
    /// bx + b
    /// ```
    /// where `bx` is unsigned binary encoding of the vector and `b` (obtained from
    /// `Self::offset_term`) is the compression offset.
    pub inner_product_correction: f16,

    /// A metric-specific correction term. Refer to the module level documentation to
    /// understand the implication of the terms outlined here.
    ///
    /// | Squared L2    |  `|X|^2`  |
    /// | Inner Product | `<X', C>` |
    pub metric_specific: f16,

    /// Two times the sum of the ones in the binary representation of the transformed
    /// unit vector.
    ///
    /// This is the term `sum(bx)` in the module level documentation.
    pub bit_sum: u16,
}

#[derive(Debug, Error, Clone, Copy, PartialEq)]
pub enum DataMetaError {
    #[error("inner product correction {value} cannot fit in a 16-bit floating point number")]
    InnerProductCorrection { value: f32 },

    #[error("metric specific correction {value} cannot fit in a 16-bit floating point number")]
    MetricSpecific { value: f32 },

    #[error("bit sum {value} cannot fit in a 16-bit unsigned integer")]
    BitSum { value: u32 },
}

impl DataMeta {
    /// Construct a new metadata from components.
    ///
    /// This will internally convert the `f32` values to `f16`.
    pub fn new(
        inner_product_correction: f32,
        metric_specific: f32,
        bit_sum: u32,
    ) -> Result<Self, DataMetaError> {
        let inner_product_correction_f16 = diskann_wide::cast_f32_to_f16(inner_product_correction);
        if !inner_product_correction_f16.is_finite() {
            return Err(DataMetaError::InnerProductCorrection {
                value: inner_product_correction,
            });
        }

        let metric_specific_f16 = diskann_wide::cast_f32_to_f16(metric_specific);
        if !metric_specific_f16.is_finite() {
            return Err(DataMetaError::MetricSpecific {
                value: metric_specific,
            });
        }

        let bit_sum_u16: u16 = bit_sum
            .try_into()
            .map_err(|_| DataMetaError::BitSum { value: bit_sum })?;

        Ok(Self {
            inner_product_correction: inner_product_correction_f16,
            metric_specific: metric_specific_f16,
            bit_sum: bit_sum_u16,
        })
    }

    /// Compute the term `b` for a binary compression of a vector so the reconstruction can
    /// be expressed as
    /// ```math
    /// a (bx + b)
    /// ```
    /// where
    ///
    /// * `a` is the scaling term to achieve the correct dynamic range.
    /// * `bx` is the unsigned binary encoded vector.
    ///
    /// This value is computed as
    /// ```math
    /// 2 ^ NBITS - 1
    /// -------------
    ///      2
    /// ```
    /// and ensures equal coverage above and below 0.
    const fn offset_term<const NBITS: usize>() -> f32 {
        ((2usize).pow(NBITS as u32) as f32 - 1.0) / 2.0
    }

    /// Convert the values in `self` to their full precision representation for computation.
    #[inline(always)]
    pub fn to_full<A>(self, arch: A) -> DataMetaF32
    where
        A: Architecture,
    {
        use diskann_wide::SIMDVector;

        // Relying on `diskann_wide::cast_f16_to_f32` to correctly propagation `target_features`
        // correction does not seem to completely work.
        //
        // We take matters into our own hand and use the architecture's conversion routines
        // directly.
        let pre = [
            self.metric_specific,
            self.inner_product_correction,
            half::f16::default(),
            half::f16::default(),
            half::f16::default(),
            half::f16::default(),
            half::f16::default(),
            half::f16::default(),
        ];

        let post: <A as Architecture>::f32x8 =
            <A as Architecture>::f16x8::from_array(arch, pre).into();
        let post = post.to_array();

        DataMetaF32 {
            metric_specific: post[0],
            inner_product_correction: post[1],
            bit_sum: self.bit_sum.into(),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct DataMetaF32 {
    pub inner_product_correction: f32,
    pub metric_specific: f32,
    pub bit_sum: f32,
}

/// A borrowed `ComptensatedVector`.
pub type DataRef<'a, const NBITS: usize> = meta::VectorRef<'a, NBITS, Unsigned, DataMeta>;

/// A mutably borrowed `ComptensatedVector`.
pub type DataMut<'a, const NBITS: usize> = meta::VectorMut<'a, NBITS, Unsigned, DataMeta>;

/// An owning data vector.
pub type Data<const NBITS: usize, A> = meta::PolyVector<NBITS, Unsigned, DataMeta, Dense, A>;

///////////
// Query //
///////////

/// Scalar quantization correction factors for computing distances between scalar quantized
/// queries and spherically quantized data elements.
///
/// Computing the distance between a query and a data vector uses the same forumla derived
/// in the module level documentation.
///
/// The one difference is that the query must explicitly carry the "offset" term as it
/// cannot be derived from the number of bits used for the compression.
#[derive(Copy, Clone, Default, Debug, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct QueryMeta {
    /// The value with which to scale the bit-level inner product with the 1-bit data data
    /// vectors.
    pub inner_product_correction: f32,

    /// Scaling factor for the `DataMeta::twice_contraction`. Applied separately to
    /// still allow 1-bit vectors using `DataMeta` to compute distances with eachother
    /// efficiently.
    pub bit_sum: f32,

    /// The query-specific offset, taking into account the scaling factor for the query as
    /// well as its minimum value. See the struct-level documentation for an explanation.
    pub offset: f32,

    /// The corresponding metric specific term as [`DataMeta`].
    pub metric_specific: f32,
}

/// A specialized type for computing higher-precision inner products with data vectors.
pub type Query<const NBITS: usize, Perm, A> = meta::PolyVector<NBITS, Unsigned, QueryMeta, Perm, A>;

/// A reference-like version of `Query`.
pub type QueryRef<'a, const NBITS: usize, Perm> =
    meta::VectorRef<'a, NBITS, Unsigned, QueryMeta, Perm>;

/// A mutable reference-like version of `Query`.
pub type QueryMut<'a, const NBITS: usize, Perm> =
    meta::VectorMut<'a, NBITS, Unsigned, QueryMeta, Perm>;

////////////////////
// Full Precision //
////////////////////

#[derive(Debug, Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct FullQueryMeta {
    /// The sum of `data`.
    pub sum: f32,
    /// The norm of the shifted vector.
    pub shifted_norm: f32,
    /// Metric specific correction term. See [`DataMeta`].
    pub metric_specific: f32,
}

/// A full-precision query.
#[derive(Debug)]
pub struct FullQuery<A>
where
    A: AllocatorCore,
{
    /// The data after centering, normalization, and transformation.
    pub data: Poly<[f32], A>,
    pub meta: FullQueryMeta,
}

impl<A> FullQuery<A>
where
    A: AllocatorCore,
{
    /// Construct an empty `FullQuery` for `dim` dimensional data.
    pub fn empty(dim: usize, allocator: A) -> Result<Self, AllocatorError> {
        Ok(Self {
            data: Poly::broadcast(0.0f32, dim, allocator)?,
            meta: Default::default(),
        })
    }
}

pub type FullQueryRef<'a> = meta::slice::SliceRef<'a, f32, FullQueryMeta>;

pub type FullQueryMut<'a> = meta::slice::SliceMut<'a, f32, FullQueryMeta>;

impl<'short, A> Reborrow<'short> for FullQuery<A>
where
    A: AllocatorCore,
{
    type Target = FullQueryRef<'short>;
    fn reborrow(&'short self) -> Self::Target {
        FullQueryRef::new(&self.data, &self.meta)
    }
}

impl<'short, A> ReborrowMut<'short> for FullQuery<A>
where
    A: AllocatorCore,
{
    type Target = FullQueryMut<'short>;
    fn reborrow_mut(&'short mut self) -> Self::Target {
        FullQueryMut::new(&mut self.data, &mut self.meta)
    }
}

/////////////
// Helpers //
/////////////

/// This is a workaround to the error `Can't use generic parameters from outer function.` by
/// forcing constant evaluation of expressions involving offset terms.
struct ConstOffset<const NBITS: usize>;

impl<const NBITS: usize> ConstOffset<NBITS> {
    const OFFSET: f32 = DataMeta::offset_term::<NBITS>();
    const OFFSET_SQUARED: f32 = DataMeta::offset_term::<NBITS>() * DataMeta::offset_term::<NBITS>();
}

/// This represents the computation
/// ```math
/// |X'| |Y'| <x, y>
/// ```
/// from the module-level docstring.
#[inline(always)]
fn kernel<A, const NBITS: usize>(
    arch: A,
    x: DataRef<'_, NBITS>,
    y: DataRef<'_, NBITS>,
    dim: f32,
) -> distances::Result<f32>
where
    A: Architecture,
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a> Target2<
        A,
        distances::MathematicalResult<u32>,
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'a, NBITS, Unsigned>,
    >,
{
    // NOTE: `Target2<_, _, _, _>` is used instead of `Architecture::run2` to ensure that
    // the kernel is inlined into this callsize.
    //
    // Even using `Architecture::run2_inline` is not sufficient to guarantee inlining.
    let ip: distances::MathematicalResult<u32> =
        <_ as Target2<_, _, _, _>>::run(InnerProduct, arch, x.vector(), y.vector());

    let ip = ip?.into_inner() as f32;

    let offset = ConstOffset::<NBITS>::OFFSET;
    let offset_squared = ConstOffset::<NBITS>::OFFSET_SQUARED;

    let xc = x.meta().to_full(arch);
    let yc = y.meta().to_full(arch);

    Ok(xc.inner_product_correction
        * yc.inner_product_correction
        * (ip - offset * (xc.bit_sum + yc.bit_sum) + offset_squared * dim))
}

////////////////////////////
// Compensated Squared L2 //
////////////////////////////

/// A `DistanceFunction` containing scaling parameters to enable distance the SquaredL2
/// distance function over `CompensatedVectors` belonging to the same quantization space.
#[derive(Debug, Clone, Copy)]
pub struct CompensatedSquaredL2 {
    pub(super) dim: f32,
}

impl CompensatedSquaredL2 {
    /// Construct a new `CompensatedSquaredL2` with the given scaling factor.
    pub fn new(dim: usize) -> Self {
        Self { dim: dim as f32 }
    }
}

/// A blanket implementation for applying the identity transformation from
/// `MathematicalValue` to `f32` for Euclidean distance computations.
impl<A, T, U> Target2<A, distances::Result<f32>, T, U> for CompensatedSquaredL2
where
    A: Architecture,
    Self: Target2<A, distances::MathematicalResult<f32>, T, U>,
{
    #[inline(always)]
    fn run(self, arch: A, x: T, y: U) -> distances::Result<f32> {
        self.run(arch, x, y).map(|r| r.into_inner())
    }
}

/// Compute the squared euclidean distance between the two compensated vectors.
///
/// The value returned by this function is scaled properly, meaning that distances returned
/// by this method are compatible with full-precision distances.
///
/// # Validity
///
/// The results of this function are only meaningful if both `x`, `y`, and `Self` belong to
/// the same quantizer.
impl<A, const NBITS: usize>
    Target2<A, distances::MathematicalResult<f32>, DataRef<'_, NBITS>, DataRef<'_, NBITS>>
    for CompensatedSquaredL2
where
    A: Architecture,
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a> Target2<
        A,
        distances::MathematicalResult<u32>,
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'a, NBITS, Unsigned>,
    >,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        x: DataRef<'_, NBITS>,
        y: DataRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        let xc = x.meta().to_full(arch);
        let yc = y.meta().to_full(arch);
        let result = xc.metric_specific + yc.metric_specific - 2.0 * kernel(arch, x, y, self.dim)?;
        Ok(MV::new(result))
    }
}

impl<A, const Q: usize, const D: usize, Perm>
    Target2<A, distances::MathematicalResult<f32>, QueryRef<'_, Q, Perm>, DataRef<'_, D>>
    for CompensatedSquaredL2
where
    A: Architecture,
    Unsigned: Representation<Q>,
    Unsigned: Representation<D>,
    Perm: PermutationStrategy<Q>,
    for<'a> InnerProduct: Target2<
        A,
        distances::MathematicalResult<u32>,
        BitSlice<'a, Q, Unsigned, Perm>,
        BitSlice<'a, D, Unsigned>,
    >,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        x: QueryRef<'_, Q, Perm>,
        y: DataRef<'_, D>,
    ) -> distances::MathematicalResult<f32> {
        let ip: distances::MathematicalResult<u32> =
            arch.run2_inline(InnerProduct, x.vector(), y.vector());
        let ip = ip?.into_inner() as f32;

        let yc = y.meta().to_full(arch);
        let xc = x.meta();

        let y_offset: f32 = DataMeta::offset_term::<D>();

        let corrected_ip = yc.inner_product_correction
            * xc.inner_product_correction
            * (ip - y_offset * xc.bit_sum + xc.offset * yc.bit_sum
                - y_offset * xc.offset * self.dim);

        Ok(MV::new(
            yc.metric_specific + xc.metric_specific - 2.0 * corrected_ip,
        ))
    }
}

/// Compute the inner product between a full-precision query and a spherically quantized
/// data vector.
///
/// Returns an error if the arguments have different lengths.
impl<A, const NBITS: usize>
    Target2<A, distances::MathematicalResult<f32>, FullQueryRef<'_>, DataRef<'_, NBITS>>
    for CompensatedSquaredL2
where
    A: Architecture,
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a> Target2<
        A,
        distances::MathematicalResult<f32>,
        &'a [f32],
        BitSlice<'a, NBITS, Unsigned>,
    >,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        x: FullQueryRef<'_>,
        y: DataRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        let s = arch
            .run2(InnerProduct, x.vector(), y.vector())?
            .into_inner();

        let xc = x.meta();
        let yc = y.meta().to_full(arch);

        let offset = ConstOffset::<NBITS>::OFFSET;
        let ip = s - xc.sum * offset;

        // NOTE: `xc.metric_specific` already carries the square norm, so we can save
        // a multiple by using it directly.
        let r = xc.metric_specific + yc.metric_specific
            - 2.0 * xc.shifted_norm * yc.inner_product_correction * ip;
        Ok(MV::new(r))
    }
}

////////////////////
// Compensated IP //
////////////////////

/// A `DistanceFunction` containing scaling parameters to enable distance the SquaredL2
/// distance function over `CompensatedVectors` belonging to the same quantization space.
#[derive(Debug, Clone, Copy)]
pub struct CompensatedIP {
    pub(super) squared_shift_norm: f32,
    pub(super) dim: f32,
}

impl CompensatedIP {
    /// Construct a new `CompensatedIP` with the given scaling factor and shift norm.
    pub fn new(shift: &[f32], dim: usize) -> Self {
        Self {
            squared_shift_norm: FastL2NormSquared.evaluate(shift),
            dim: dim as f32,
        }
    }
}

/// A blanket implementation for applying the negating transformation
/// ```text
/// x -> -x
/// ```
/// from `MathematicalValue` to `f32` for inner product distance computations.
impl<A, T, U> Target2<A, distances::Result<f32>, T, U> for CompensatedIP
where
    A: Architecture,
    Self: Target2<A, distances::MathematicalResult<f32>, T, U>,
{
    #[inline(always)]
    fn run(self, arch: A, x: T, y: U) -> distances::Result<f32> {
        arch.run2(self, x, y).map(|r| -r.into_inner())
    }
}

/// Compute the inner product between the two compensated vectors.
///
/// Returns an error if the arguments have different lengths.
///
/// The value returned by this function is scaled properly, meaning that distances returned
/// by this method are compatible with full-precision computations.
///
/// # Validity
///
/// The results of this function are only meaningful if both `x`, `y`, and `Self` belong to
/// the same quantizer.
impl<A, const NBITS: usize>
    Target2<A, distances::MathematicalResult<f32>, DataRef<'_, NBITS>, DataRef<'_, NBITS>>
    for CompensatedIP
where
    A: Architecture,
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a> Target2<
        A,
        distances::MathematicalResult<u32>,
        BitSlice<'a, NBITS, Unsigned>,
        BitSlice<'a, NBITS, Unsigned>,
    >,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        x: DataRef<'_, NBITS>,
        y: DataRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        let xc = x.meta().to_full(arch);
        let yc = y.meta().to_full(arch);

        let result = xc.metric_specific
            + yc.metric_specific
            + kernel(arch, x, y, self.dim)?
            + self.squared_shift_norm;
        Ok(MV::new(result))
    }
}

impl<A, const Q: usize, const D: usize, Perm>
    Target2<A, distances::MathematicalResult<f32>, QueryRef<'_, Q, Perm>, DataRef<'_, D>>
    for CompensatedIP
where
    A: Architecture,
    Unsigned: Representation<Q>,
    Unsigned: Representation<D>,
    Perm: PermutationStrategy<Q>,
    for<'a> InnerProduct: Target2<
        A,
        distances::MathematicalResult<u32>,
        BitSlice<'a, Q, Unsigned, Perm>,
        BitSlice<'a, D, Unsigned>,
    >,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        x: QueryRef<'_, Q, Perm>,
        y: DataRef<'_, D>,
    ) -> distances::MathematicalResult<f32> {
        // The inner product of the bit-level data.
        let ip: MV<u32> = arch.run2_inline(InnerProduct, x.vector(), y.vector())?;

        let yc = y.meta().to_full(arch);
        let xc = x.meta();

        // Rely on constant propagation to pre-compute these terms.
        let y_offset: f32 = DataMeta::offset_term::<D>();

        let corrected_ip = xc.inner_product_correction
            * yc.inner_product_correction
            * (ip.into_inner() as f32 - y_offset * xc.bit_sum + xc.offset * yc.bit_sum
                - y_offset * xc.offset * self.dim);

        // Finally, reassemble the remaining compensation terms.
        Ok(MV::new(
            corrected_ip + yc.metric_specific + xc.metric_specific + self.squared_shift_norm,
        ))
    }
}

/// Compute the inner product between a full-precision query and a spherically quantized
/// data vector.
///
/// Returns an error if the arguments have different lengths.
impl<A, const NBITS: usize>
    Target2<A, distances::MathematicalResult<f32>, FullQueryRef<'_>, DataRef<'_, NBITS>>
    for CompensatedIP
where
    A: Architecture,
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a> Target2<
        A,
        distances::MathematicalResult<f32>,
        &'a [f32],
        BitSlice<'a, NBITS, Unsigned>,
    >,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        x: FullQueryRef<'_>,
        y: DataRef<'_, NBITS>,
    ) -> distances::MathematicalResult<f32> {
        let s = arch
            .run2(InnerProduct, x.vector(), y.vector())?
            .into_inner();

        let yc = y.meta().to_full(arch);
        let xc = x.meta();

        let offset = ConstOffset::<NBITS>::OFFSET;
        let ip = xc.shifted_norm * yc.inner_product_correction * (s - xc.sum * offset);

        Ok(MV::new(
            ip + xc.metric_specific + yc.metric_specific + self.squared_shift_norm,
        ))
    }
}

////////////////////////
// Compensated Cosine //
////////////////////////

/// A `DistanceFunction` containing scaling parameters to enable distance the Cosine
/// distance function over vectors belonging to the same quantization space.
///
/// This distance function works by assuming input vectors were normalized **prior** to
/// compression and therefore cosine may be computed by delegating to inner product
/// computations. The [`crate::spherical::SphericalQuantizer`] will ensure this
/// pre-normalization when constructed with [`SupportedMetric::Cosine`].
#[derive(Debug, Clone, Copy)]
pub struct CompensatedCosine {
    pub(super) inner: CompensatedIP,
}

impl CompensatedCosine {
    /// Construct a new `CompensatedCosine` around the [`CompensatedIP`].
    pub fn new(inner: CompensatedIP) -> Self {
        Self { inner }
    }
}

impl<A, T, U> Target2<A, distances::MathematicalResult<f32>, T, U> for CompensatedCosine
where
    A: Architecture,
    CompensatedIP: Target2<A, distances::MathematicalResult<f32>, T, U>,
{
    #[inline(always)]
    fn run(self, arch: A, x: T, y: U) -> distances::MathematicalResult<f32> {
        self.inner.run(arch, x, y)
    }
}

/// A blanket implementation for applying the transformation
/// ```text
/// x -> 1-x
/// ```
/// from `MathematicalValue` to `f32` for cosine distance computations.
impl<A, T, U> Target2<A, distances::Result<f32>, T, U> for CompensatedCosine
where
    A: Architecture,
    Self: Target2<A, distances::MathematicalResult<f32>, T, U>,
{
    #[inline(always)]
    fn run(self, arch: A, x: T, y: U) -> distances::Result<f32> {
        let r: MV<f32> = self.run(arch, x, y)?;
        Ok(1.0 - r.into_inner())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, Reborrow};
    use diskann_vector::{distance::Metric, norm::FastL2Norm, PureDistanceFunction};
    use diskann_wide::ARCH;
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };
    use rand_distr::StandardNormal;

    use super::*;
    use crate::{
        alloc::GlobalAllocator,
        bits::{BitTranspose, Dense},
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Approx {
        absolute: f32,
        relative: f32,
    }

    impl Approx {
        const fn new(absolute: f32, relative: f32) -> Self {
            assert!(absolute >= 0.0);
            assert!(relative >= 0.0);
            Self { absolute, relative }
        }

        fn check(&self, got: f32, expected: f32, ctx: Option<&dyn std::fmt::Display>) -> bool {
            struct Ctx<'a>(Option<&'a dyn std::fmt::Display>);

            impl std::fmt::Display for Ctx<'_> {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    match self.0 {
                        None => write!(f, "none"),
                        Some(d) => write!(f, "{}", d),
                    }
                }
            }

            let absolute = (got - expected).abs();
            if absolute <= self.absolute {
                true
            } else {
                let relative = absolute / expected.abs();
                if relative <= self.relative {
                    true
                } else {
                    panic!(
                        "got {}, expected {}. Abs/Rel = {}/{} with bounds {}/{}: Ctx: {}",
                        got,
                        expected,
                        absolute,
                        relative,
                        self.absolute,
                        self.relative,
                        Ctx(ctx)
                    );
                }
            }
        }
    }

    //////////////
    // DataMeta //
    //////////////

    #[test]
    fn test_data_meta() {
        // Test constructor happy path.
        let meta = DataMeta::new(1.0, 2.0, 10).unwrap();
        let expected = DataMetaF32 {
            inner_product_correction: 1.0,
            metric_specific: 2.0,
            bit_sum: 10.0,
        };
        assert_eq!(meta.to_full(ARCH), expected);

        // Test constructor errors.
        let err = DataMeta::new(65600.0, 2.0, 10).unwrap_err();
        assert_eq!(
            err.to_string(),
            "inner product correction 65600 cannot fit in a 16-bit floating point number"
        );

        let err = DataMeta::new(2.0, 65600.0, 10).unwrap_err();
        assert_eq!(
            err.to_string(),
            "metric specific correction 65600 cannot fit in a 16-bit floating point number"
        );

        let err = DataMeta::new(2.0, 2.0, 65536).unwrap_err();
        assert_eq!(
            err.to_string(),
            "bit sum 65536 cannot fit in a 16-bit unsigned integer",
        );
    }

    //////////////////////
    // Supported Metric //
    //////////////////////

    #[test]
    fn supported_metric() {
        assert_eq!(
            SupportedMetric::try_from(Metric::L2).unwrap(),
            SupportedMetric::SquaredL2
        );
        assert_eq!(
            SupportedMetric::try_from(Metric::InnerProduct).unwrap(),
            SupportedMetric::InnerProduct
        );
        assert_eq!(
            SupportedMetric::try_from(Metric::Cosine).unwrap(),
            SupportedMetric::Cosine
        );
        assert!(matches!(
            SupportedMetric::try_from(Metric::CosineNormalized),
            Err(UnsupportedMetric(Metric::CosineNormalized))
        ));

        assert_eq!(SupportedMetric::SquaredL2, Metric::L2);
        assert_ne!(SupportedMetric::SquaredL2, Metric::InnerProduct);
        assert_ne!(SupportedMetric::SquaredL2, Metric::Cosine);
        assert_ne!(SupportedMetric::SquaredL2, Metric::CosineNormalized);

        assert_ne!(SupportedMetric::InnerProduct, Metric::L2);
        assert_eq!(SupportedMetric::InnerProduct, Metric::InnerProduct);
        assert_ne!(SupportedMetric::SquaredL2, Metric::Cosine);
        assert_ne!(SupportedMetric::SquaredL2, Metric::CosineNormalized);
    }

    ///////////////
    // Distances //
    ///////////////

    struct Reference<T> {
        compressed: T,
        reconstructed: Vec<f32>,
        norm: f32,
        center_ip: f32,
        self_ip: Option<f32>,
    }

    trait GenerateReference: Sized {
        fn generate_reference(
            center: &[f32],
            metric: SupportedMetric,
            rng: &mut StdRng,
        ) -> Reference<Self>;
    }

    impl<const NBITS: usize> GenerateReference for Data<NBITS, GlobalAllocator>
    where
        Unsigned: Representation<NBITS>,
    {
        fn generate_reference(
            center: &[f32],
            metric: SupportedMetric,
            rng: &mut StdRng,
        ) -> Reference<Self> {
            let dim = center.len();

            let mut reconstructed = vec![0.0f32; dim];
            let mut compressed = Data::<NBITS, _>::new_boxed(dim);

            let mut bit_sum = 0;
            let dist = Uniform::try_from(Unsigned::domain_const::<NBITS>()).unwrap();
            let offset = (2usize.pow(NBITS as u32) as f32 - 1.0) / 2.0;
            for (i, r) in reconstructed.iter_mut().enumerate() {
                let b: i64 = dist.sample(rng);
                bit_sum += b;
                compressed.vector_mut().set(i, b).unwrap();
                *r = (b as f32) - offset;
            }

            let r_norm = FastL2Norm.evaluate(reconstructed.as_slice());
            reconstructed.iter_mut().for_each(|i| *i /= r_norm);

            let norm: f32 = Uniform::new(0.0, 2.0).unwrap().sample(rng);
            let center_ip: f32 = Uniform::new(0.5, 2.5).unwrap().sample(rng);
            let self_ip: f32 = Uniform::new(0.5, 1.5).unwrap().sample(rng);

            compressed.set_meta(
                DataMeta::new(
                    norm / (self_ip * r_norm),
                    metric.pick(norm, center_ip),
                    bit_sum.try_into().unwrap(),
                )
                .unwrap(),
            );

            Reference {
                compressed,
                reconstructed,
                norm,
                center_ip,
                self_ip: Some(self_ip),
            }
        }
    }

    impl<const NBITS: usize, Perm> GenerateReference for Query<NBITS, Perm, GlobalAllocator>
    where
        Unsigned: Representation<NBITS>,
        Perm: PermutationStrategy<NBITS>,
    {
        fn generate_reference(
            center: &[f32],
            metric: SupportedMetric,
            rng: &mut StdRng,
        ) -> Reference<Self> {
            let dim = center.len();

            let mut reconstructed = vec![0.0f32; dim];
            let mut compressed = Query::<NBITS, Perm, _>::new_boxed(dim);

            let distribution = Uniform::try_from(Unsigned::domain_const::<NBITS>()).unwrap();

            let base: f32 = StandardNormal {}.sample(rng);
            let scale: f32 = {
                let scale: f32 = StandardNormal {}.sample(rng);
                scale.abs()
            };

            let mut bit_sum = 0;
            for (i, r) in reconstructed.iter_mut().enumerate() {
                let b = distribution.sample(rng);
                compressed.vector_mut().set(i, b).unwrap();
                *r = base + scale * (b as f32);
                bit_sum += b;
            }

            let norm: f32 = Uniform::new(0.0, 2.0).unwrap().sample(rng);
            let center_ip: f32 = Uniform::new(-2.0, 2.0).unwrap().sample(rng);

            compressed.set_meta(QueryMeta {
                inner_product_correction: norm * scale,
                bit_sum: bit_sum as f32,
                offset: base / scale,
                metric_specific: metric.pick(norm, center_ip),
            });

            Reference {
                compressed,
                reconstructed,
                norm,
                center_ip,
                self_ip: None,
            }
        }
    }

    impl GenerateReference for FullQuery<GlobalAllocator> {
        fn generate_reference(
            center: &[f32],
            metric: SupportedMetric,
            rng: &mut StdRng,
        ) -> Reference<Self> {
            let dim = center.len();

            let mut query = FullQuery::empty(dim, GlobalAllocator).unwrap();

            let mut sum = 0.0;
            let dist = StandardNormal {};
            for r in query.data.iter_mut() {
                let b: f32 = dist.sample(rng);
                sum += b;
                *r = b;
            }

            let r_norm = FastL2Norm.evaluate(&*query.data);
            query.data.iter_mut().for_each(|i| *i /= r_norm);

            let norm: f32 = Uniform::new(0.0, 2.0).unwrap().sample(rng);
            let center_ip: f32 = Uniform::new(-2.0, 2.0).unwrap().sample(rng);

            query.meta = FullQueryMeta {
                sum: sum / r_norm,
                shifted_norm: norm,
                metric_specific: metric.pick(norm, center_ip),
            };

            let reconstructed = query.data.to_vec();
            Reference {
                compressed: query,
                reconstructed,
                norm,
                center_ip,
                self_ip: None,
            }
        }
    }

    /// Refer to the module level documentation for some insight into what these components
    /// mean.
    ///
    /// The gist of these tests are that we generate the binary vectors `bx` and `by`
    /// (along with their code-book representation), the center, and the shifted versions
    /// of the target vectors.
    ///
    /// From those components, we computed the compensation terms and compute the expected
    /// result manually, verifying that the compensated computation works as expected.
    fn test_compensated_distance<const NBITS: usize>(
        dim: usize,
        ntrials: usize,
        err_l2: Approx,
        err_ip: Approx,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<NBITS>,
        for<'a> CompensatedIP: Target2<
                diskann_wide::arch::Current,
                distances::Result<f32>,
                DataRef<'a, NBITS>,
                DataRef<'a, NBITS>,
            > + Target2<
                diskann_wide::arch::Current,
                distances::MathematicalResult<f32>,
                DataRef<'a, NBITS>,
                DataRef<'a, NBITS>,
            >,
        for<'a> CompensatedSquaredL2: Target2<
                diskann_wide::arch::Current,
                distances::Result<f32>,
                DataRef<'a, NBITS>,
                DataRef<'a, NBITS>,
            > + Target2<
                diskann_wide::arch::Current,
                distances::MathematicalResult<f32>,
                DataRef<'a, NBITS>,
                DataRef<'a, NBITS>,
            >,
    {
        let mut center = vec![0.0f32; dim];
        for trial in 0..ntrials {
            // Sample the center.
            center
                .iter_mut()
                .for_each(|c| *c = StandardNormal {}.sample(rng));

            let c_square_norm = FastL2NormSquared.evaluate(&*center);

            // Inner Product
            {
                let x = Data::<NBITS, _>::generate_reference(
                    &center,
                    SupportedMetric::InnerProduct,
                    rng,
                );
                let y = Data::<NBITS, _>::generate_reference(
                    &center,
                    SupportedMetric::InnerProduct,
                    rng,
                );

                let kernel_result = {
                    let xy: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                        &*x.reconstructed,
                        &*y.reconstructed,
                    );
                    x.norm * y.norm * xy.into_inner() / (x.self_ip.unwrap() * y.self_ip.unwrap())
                };

                let reference_ip = kernel_result + x.center_ip + y.center_ip + c_square_norm;
                let ip = CompensatedIP::new(&center, center.len());
                let got_ip: distances::MathematicalResult<f32> =
                    ARCH.run2(ip, x.compressed.reborrow(), y.compressed.reborrow());
                let got_ip = got_ip.unwrap();

                let ctx = &lazy_format!(
                    "Inner Product, trial {} of {}, dim = {}",
                    trial,
                    ntrials,
                    dim
                );
                assert!(err_ip.check(got_ip.into_inner(), reference_ip, Some(ctx)));

                let got_ip_f32: distances::Result<f32> =
                    ARCH.run2(ip, x.compressed.reborrow(), y.compressed.reborrow());

                let got_ip_f32 = got_ip_f32.unwrap();

                assert_eq!(got_ip_f32, -got_ip.into_inner());

                // Cosine (very similary to inner-product).
                let cosine = CompensatedCosine::new(ip);
                let got_cosine: distances::MathematicalResult<f32> =
                    ARCH.run2(cosine, x.compressed.reborrow(), y.compressed.reborrow());
                let got_cosine = got_cosine.unwrap();
                assert_eq!(
                    got_cosine.into_inner(),
                    got_ip.into_inner(),
                    "cosine and IP should be the same"
                );

                let got_cosine_f32: distances::Result<f32> =
                    ARCH.run2(cosine, x.compressed.reborrow(), y.compressed.reborrow());

                let got_cosine_f32 = got_cosine_f32.unwrap();

                assert_eq!(
                    got_cosine_f32,
                    1.0 - got_cosine.into_inner(),
                    "incorrect transform performed"
                );
            }

            // Squared L2
            {
                let x =
                    Data::<NBITS, _>::generate_reference(&center, SupportedMetric::SquaredL2, rng);
                let y =
                    Data::<NBITS, _>::generate_reference(&center, SupportedMetric::SquaredL2, rng);

                // Compute the expected value for the quantity `|X'| |Y'| <x, y>`.
                let kernel_result = {
                    let xy: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                        &*x.reconstructed,
                        &*y.reconstructed,
                    );
                    x.norm * y.norm * xy.into_inner() / (x.self_ip.unwrap() * y.self_ip.unwrap())
                };

                let reference_l2 = x.norm * x.norm + y.norm * y.norm - 2.0 * kernel_result;
                let l2 = CompensatedSquaredL2::new(dim);
                let got_l2: distances::MathematicalResult<f32> =
                    ARCH.run2(l2, x.compressed.reborrow(), y.compressed.reborrow());
                let got_l2 = got_l2.unwrap();

                let ctx =
                    &lazy_format!("Squared L2, trial {} of {}, dim = {}", trial, ntrials, dim);
                assert!(err_l2.check(got_l2.into_inner(), reference_l2, Some(ctx)));

                let got_l2_f32: distances::Result<f32> =
                    ARCH.run2(l2, x.compressed.reborrow(), y.compressed.reborrow());
                let got_l2_f32 = got_l2_f32.unwrap();

                assert_eq!(got_l2_f32, got_l2.into_inner());
            }
        }
    }

    /// This works similarly to the 1-bit compensated distances, but checks the 4-bit query
    /// path.
    fn test_mixed_compensated_distance<const Q: usize, const D: usize, Perm>(
        dim: usize,
        ntrials: usize,
        err_l2: Approx,
        err_ip: Approx,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<Q>,
        Unsigned: Representation<D>,
        Perm: PermutationStrategy<Q>,
        for<'a> CompensatedIP: Target2<
            diskann_wide::arch::Current,
            distances::MathematicalResult<f32>,
            QueryRef<'a, Q, Perm>,
            DataRef<'a, D>,
        >,
        for<'a> CompensatedSquaredL2: Target2<
            diskann_wide::arch::Current,
            distances::MathematicalResult<f32>,
            QueryRef<'a, Q, Perm>,
            DataRef<'a, D>,
        >,
        for<'a> CompensatedCosine: Target2<
            diskann_wide::arch::Current,
            distances::MathematicalResult<f32>,
            QueryRef<'a, Q, Perm>,
            DataRef<'a, D>,
        >,
        for<'a> CompensatedIP: Target2<
            diskann_wide::arch::Current,
            distances::Result<f32>,
            QueryRef<'a, Q, Perm>,
            DataRef<'a, D>,
        >,
        for<'a> CompensatedSquaredL2: Target2<
            diskann_wide::arch::Current,
            distances::Result<f32>,
            QueryRef<'a, Q, Perm>,
            DataRef<'a, D>,
        >,
        for<'a> CompensatedCosine: Target2<
            diskann_wide::arch::Current,
            distances::Result<f32>,
            QueryRef<'a, Q, Perm>,
            DataRef<'a, D>,
        >,
    {
        // The center
        let mut center = vec![0.0f32; dim];
        for trial in 0..ntrials {
            // Sample the center.
            center
                .iter_mut()
                .for_each(|c| *c = StandardNormal {}.sample(rng));

            let c_square_norm = FastL2NormSquared.evaluate(&*center);

            // Inner Product
            {
                let x = Query::<Q, Perm, _>::generate_reference(
                    &center,
                    SupportedMetric::InnerProduct,
                    rng,
                );
                let y =
                    Data::<D, _>::generate_reference(&center, SupportedMetric::InnerProduct, rng);

                // The expected scaled dot-product between the normalized vectors.
                let xy = {
                    let xy: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                        &*x.reconstructed,
                        &*y.reconstructed,
                    );
                    x.norm * y.norm * xy.into_inner() / y.self_ip.unwrap()
                };

                let reference_ip = -(xy + x.center_ip + y.center_ip + c_square_norm);
                let ip = CompensatedIP::new(&center, center.len());
                let got_ip: distances::Result<f32> =
                    ARCH.run2(ip, x.compressed.reborrow(), y.compressed.reborrow());
                let got_ip = got_ip.unwrap();

                let ctx = &lazy_format!(
                    "Inner Product, trial = {} of {}, dim = {}",
                    trial,
                    ntrials,
                    dim
                );

                assert!(err_ip.check(got_ip, reference_ip, Some(ctx)));

                // Cosine (very similary to inner-product).
                let cosine = CompensatedCosine::new(ip);
                let got_cosine: distances::MathematicalResult<f32> =
                    ARCH.run2(cosine, x.compressed.reborrow(), y.compressed.reborrow());

                let got_cosine = got_cosine.unwrap();
                assert_eq!(
                    got_cosine.into_inner(),
                    -got_ip,
                    "cosine and IP should be the same"
                );

                let got_cosine_f32: distances::Result<f32> =
                    ARCH.run2(cosine, x.compressed.reborrow(), y.compressed.reborrow());

                let got_cosine_f32 = got_cosine_f32.unwrap();
                assert_eq!(
                    got_cosine_f32,
                    1.0 - got_cosine.into_inner(),
                    "incorrect transform performed"
                );
            }

            // Squared L2
            {
                let x = Query::<Q, Perm, _>::generate_reference(
                    &center,
                    SupportedMetric::SquaredL2,
                    rng,
                );
                let y = Data::<D, _>::generate_reference(&center, SupportedMetric::SquaredL2, rng);

                // The expected scaled dot-product between the normalized vectors.
                let xy = {
                    let xy: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                        &*x.reconstructed,
                        &*y.reconstructed,
                    );
                    x.norm * y.norm * xy.into_inner() / y.self_ip.unwrap()
                };
                let reference_l2 = x.norm * x.norm + y.norm * y.norm - 2.0 * xy;
                let l2 = CompensatedSquaredL2::new(dim);
                let got_l2: distances::Result<f32> =
                    ARCH.run2(l2, x.compressed.reborrow(), y.compressed.reborrow());
                let got_l2 = got_l2.unwrap();

                let ctx = &lazy_format!(
                    "Squared L2, trial = {} of {}, dim = {}",
                    trial,
                    ntrials,
                    dim
                );

                assert!(err_l2.check(got_l2, reference_l2, Some(ctx)));
            }
        }
    }

    fn test_full_distances<const NBITS: usize>(
        dim: usize,
        ntrials: usize,
        err_l2: Approx,
        err_ip: Approx,
        rng: &mut StdRng,
    ) where
        Unsigned: Representation<NBITS>,
        for<'a> CompensatedIP: Target2<
            diskann_wide::arch::Current,
            distances::MathematicalResult<f32>,
            FullQueryRef<'a>,
            DataRef<'a, NBITS>,
        >,
        for<'a> CompensatedSquaredL2: Target2<
            diskann_wide::arch::Current,
            distances::MathematicalResult<f32>,
            FullQueryRef<'a>,
            DataRef<'a, NBITS>,
        >,
        for<'a> CompensatedCosine: Target2<
            diskann_wide::arch::Current,
            distances::MathematicalResult<f32>,
            FullQueryRef<'a>,
            DataRef<'a, NBITS>,
        >,
        for<'a> CompensatedIP: Target2<
            diskann_wide::arch::Current,
            distances::Result<f32>,
            FullQueryRef<'a>,
            DataRef<'a, NBITS>,
        >,
        for<'a> CompensatedSquaredL2: Target2<
            diskann_wide::arch::Current,
            distances::Result<f32>,
            FullQueryRef<'a>,
            DataRef<'a, NBITS>,
        >,
        for<'a> CompensatedCosine: Target2<
            diskann_wide::arch::Current,
            distances::Result<f32>,
            FullQueryRef<'a>,
            DataRef<'a, NBITS>,
        >,
    {
        // The center
        let mut center = vec![0.0f32; dim];
        for trial in 0..ntrials {
            // Sample the center.
            center
                .iter_mut()
                .for_each(|c| *c = StandardNormal {}.sample(rng));

            let c_square_norm = FastL2NormSquared.evaluate(&*center);

            // Inner Product
            {
                let x = FullQuery::generate_reference(&center, SupportedMetric::InnerProduct, rng);
                let y = Data::<NBITS, _>::generate_reference(
                    &center,
                    SupportedMetric::InnerProduct,
                    rng,
                );

                // The expected scaled dot-product between the normalized vectors.
                let xy = {
                    let xy: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                        &*x.reconstructed,
                        &*y.reconstructed,
                    );
                    x.norm * y.norm * xy.into_inner() / y.self_ip.unwrap()
                };

                let reference_ip = -(xy + x.center_ip + y.center_ip + c_square_norm);
                let ip = CompensatedIP::new(&center, center.len());
                let got_ip: distances::Result<f32> =
                    ARCH.run2(ip, x.compressed.reborrow(), y.compressed.reborrow());
                let got_ip = got_ip.unwrap();

                let ctx = &lazy_format!(
                    "Inner Product, trial = {} of {}, dim = {}",
                    trial,
                    ntrials,
                    dim
                );

                assert!(err_ip.check(got_ip, reference_ip, Some(ctx)));

                // Cosine (very similary to inner-product).
                let cosine = CompensatedCosine::new(ip);
                let got_cosine: distances::MathematicalResult<f32> =
                    ARCH.run2(cosine, x.compressed.reborrow(), y.compressed.reborrow());
                let got_cosine = got_cosine.unwrap();
                assert_eq!(
                    got_cosine.into_inner(),
                    -got_ip,
                    "cosine and IP should be the same"
                );

                let got_cosine_f32: distances::Result<f32> =
                    ARCH.run2(cosine, x.compressed.reborrow(), y.compressed.reborrow());

                let got_cosine_f32 = got_cosine_f32.unwrap();
                assert_eq!(
                    got_cosine_f32,
                    1.0 - got_cosine.into_inner(),
                    "incorrect transform performed"
                );
            }

            // Squared L2
            {
                let x = FullQuery::generate_reference(&center, SupportedMetric::SquaredL2, rng);
                let y =
                    Data::<NBITS, _>::generate_reference(&center, SupportedMetric::SquaredL2, rng);

                // The expected scaled dot-product between the normalized vectors.
                let xy = {
                    let xy: MV<f32> = diskann_vector::distance::InnerProduct::evaluate(
                        &*x.reconstructed,
                        &*y.reconstructed,
                    );
                    x.norm * y.norm * xy.into_inner() / y.self_ip.unwrap()
                };

                let reference_l2 = x.norm * x.norm + y.norm * y.norm - 2.0 * xy;
                let l2 = CompensatedSquaredL2::new(dim);
                let got_l2: distances::Result<f32> =
                    ARCH.run2(l2, x.compressed.reborrow(), y.compressed.reborrow());
                let got_l2 = got_l2.unwrap();

                let ctx = &lazy_format!(
                    "Squared L2, trial = {} of {}, dim = {}",
                    trial,
                    ntrials,
                    dim
                );
                assert!(err_l2.check(got_l2, reference_l2, Some(ctx)));
            }
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // The max dim does not need to be as high for these vectors because they
            // defer their distance function implementation to `BitSlice`, which is more
            // heavily tested.
            const MAX_DIM: usize = 37;
            const TRIALS_PER_DIM: usize = 1;
        } else {
            const MAX_DIM: usize = 256;
            const TRIALS_PER_DIM: usize = 20;
        }
    }

    #[test]
    fn test_symmetric_distances_1bit() {
        let mut rng = StdRng::seed_from_u64(0x2a5f79a2469218f6);
        for dim in 1..MAX_DIM {
            test_compensated_distance::<1>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(4.0e-3, 3.0e-3),
                Approx::new(1.0e-3, 5.0e-4),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_symmetric_distances_2bit() {
        let mut rng = StdRng::seed_from_u64(0x68f8f52057f94399);
        for dim in 1..MAX_DIM {
            test_compensated_distance::<2>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(3.5e-3, 2.0e-3),
                Approx::new(2.0e-3, 5.0e-4),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_symmetric_distances_4bit() {
        let mut rng = StdRng::seed_from_u64(0xb88d76ac4c58e923);
        for dim in 1..MAX_DIM {
            test_compensated_distance::<4>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(2.0e-3, 2.0e-3),
                Approx::new(2.0e-3, 5.0e-4),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_symmetric_distances_8bit() {
        let mut rng = StdRng::seed_from_u64(0x1c2b79873ee32626);
        for dim in 1..MAX_DIM {
            test_compensated_distance::<8>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(2.0e-3, 2.0e-3),
                Approx::new(2.0e-3, 4.0e-4),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_mixed_distances_4x1() {
        let mut rng = StdRng::seed_from_u64(0x1efb4d87ed0a8ada);
        for dim in 1..MAX_DIM {
            test_mixed_compensated_distance::<4, 1, BitTranspose>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(4.0e-3, 3.0e-3),
                Approx::new(1.3e-2, 8.3e-3),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_mixed_distances_4x4() {
        let mut rng = StdRng::seed_from_u64(0x508554264eb7a51b);
        for dim in 1..MAX_DIM {
            test_mixed_compensated_distance::<4, 4, Dense>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(4.0e-3, 3.0e-3),
                Approx::new(3.0e-4, 8.3e-2),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_mixed_distances_8x8() {
        let mut rng = StdRng::seed_from_u64(0x8acd8e4224c76c43);
        for dim in 1..MAX_DIM {
            test_mixed_compensated_distance::<8, 8, Dense>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(2.0e-3, 6.0e-3),
                Approx::new(1.0e-2, 3.0e-2),
                &mut rng,
            );
        }
    }

    // Full
    #[test]
    fn test_full_distances_1bit() {
        let mut rng = StdRng::seed_from_u64(0x7f93530559f42d66);
        for dim in 1..MAX_DIM {
            test_full_distances::<1>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(1.0e-3, 2.0e-3),
                Approx::new(0.0, 5.0e-3),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_full_distances_2bit() {
        let mut rng = StdRng::seed_from_u64(0xa3ad61d3d03a0c5a);
        for dim in 1..MAX_DIM {
            test_full_distances::<2>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(2.0e-3, 1.1e-3),
                Approx::new(7.0e-4, 1.0e-3),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_full_distances_4bit() {
        let mut rng = StdRng::seed_from_u64(0x3e2f50ed7c64f0c2);
        for dim in 1..MAX_DIM {
            test_full_distances::<4>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(2.0e-3, 1.0e-2),
                Approx::new(1.0e-3, 5.0e-4),
                &mut rng,
            );
        }
    }

    #[test]
    fn test_full_distances_8bit() {
        let mut rng = StdRng::seed_from_u64(0x95705070e415c6d3);
        for dim in 1..MAX_DIM {
            test_full_distances::<8>(
                dim,
                TRIALS_PER_DIM,
                Approx::new(1.0e-3, 1.0e-3),
                Approx::new(2.0e-3, 1.0e-4),
                &mut rng,
            );
        }
    }
}
