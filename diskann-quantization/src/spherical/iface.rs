/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The main export of this module is the dyn compatible [`Quantizer`] trait, which provides
//! a common interface for interacting with bit-width specific [`SphericalQuantizer`]s,
//! compressing vectors, and computing distances between compressed vectors.
//!
//! This is offered as a convenience interface for interacting with the myriad of generics
//! associated with the [`SphericalQuantizer`]. Better performance can be achieved by using
//! the generic types directly if desired.
//!
//! The [`Quantizer`] uses the [`Opaque`] and [`OpaqueMut`] types for its compressed data
//! representations. These are thin wrappers around raw byte slices.
//!
//! Distance computation is performed using [`DistanceComputer`] and [`QueryComputer`]
//!
//! Concrete implementations of [`Quantizer`] are available via the generic struct [`Impl`].
//!
//! ## Compatibility Table
//!
//! Multiple [`QueryLayout`]s are supported when constructing a [`QueryComputer`], but not
//! all layouts are supported for each back end. This table lists the valid instantiations
//! of [`Impl`] (parameterized by data vector bit-width) and their supported query layouts.
//!
//! | Bits | Same As Data | Full Precision | Four-Bit Transposed | Scalar Quantized |
//! |------|--------------|----------------|---------------------|------------------|
//! |    1 |     Yes      |      Yes       |         Yes         |       No         |
//! |    2 |     Yes      |      Yes       |          No         |       Yes        |
//! |    4 |     Yes      |      Yes       |          No         |       Yes        |
//! |    8 |     Yes      |      Yes       |          No         |       Yes        |
//!
//! # Example
//!
//! ```
//! use diskann_quantization::{
//!     alloc::{Poly, ScopedAllocator, AlignedAllocator, GlobalAllocator},
//!     algorithms::TransformKind,
//!     spherical::{iface, SupportedMetric, SphericalQuantizer, PreScale},
//!     num::PowerOfTwo,
//! };
//! use diskann_utils::views::Matrix;
//!
//! // For illustration purposes, the dataset consists of just a single vector.
//! let mut data = Matrix::new(1.0, 1, 4);
//! let quantizer = SphericalQuantizer::train(
//!     data.as_view(),
//!     TransformKind::Null,
//!     SupportedMetric::SquaredL2,
//!     PreScale::None,
//!     &mut rand::rng(),
//!     GlobalAllocator
//! ).unwrap();
//!
//! let quantizer: Box<dyn iface::Quantizer> = Box::new(
//!     iface::Impl::<1>::new(quantizer).unwrap()
//! );
//!
//! let alloc = AlignedAllocator::new(PowerOfTwo::new(1).unwrap());
//! let mut buf = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();
//!
//! quantizer.compress(
//!     data.row(0),
//!     iface::OpaqueMut::new(&mut buf),
//!     ScopedAllocator::new(&alloc),
//! ).unwrap();
//!
//! assert!(quantizer.is_supported(iface::QueryLayout::FullPrecision));
//! assert!(!quantizer.is_supported(iface::QueryLayout::ScalarQuantized));
//! ```

/// # DevDocs
///
/// This section provides developer documentation for the structure of the structs and
/// traits inside this file. The main goal of the [`DistanceComputer`] and [`QueryComputer`]
/// implementations are to introduce just a single level of indirection.
///
/// ## Distance Computer Philosophy
///
/// The goal of this code (and in large part the reason for the somewhat spaghetti nature)
/// is to do all the the dispatches:
///
/// * Number of bits.
/// * Query layout.
/// * Distance Type (L2, Inner Product, Cosine).
/// * Micro-architecture specific code-generation
///
/// Behind a **single** level of dynamic dispatch. This means we need to bake all of this
/// information into a single type, facilitated through a combination of the `Reify` and
/// `Curried` private structs.
///
/// ## Anatomy of the [`DistanceComputer`]
///
/// To provide a single level of indirection for all distance function implementations,
/// the [`DistanceComputer`] is a thin wrapper around the [`DynDistanceComputer`] trait.
///
/// Concrete implementations of this trait consist of a base distance function like
/// [`CompensatedIP`] or [`CompensatedSquaredL2`]. Because the data is passed through the
/// [`Opaque`] type, these distance functions are embedded inside a [`Reify`] which first
/// converts the [`Opaque`] to the appropriate fully-typed object before calling the inner
/// distance function.
///
/// This full typing is supplied by the private [`FromOpaque`] helper trait.
///
/// When returned from the [`Quantizer::distance_computer`] or
/// [`Quantizer::distance_computer_ref`] traits, the resulting computer will be specialized
/// to work solely on data vectors compressed through [`Quantizer::compress`].
///
/// When returned from [`Quantizer::query_computer`], the expected type of the query will
/// depend on the [`QueryLayout`] supplied to that method.
///
/// The method [`QueryComputer::layout`] is provided to inspect at run time the query layout
/// the object is meant for.
///
/// If at all possible, the [`QueryComputer`] should be preferred as it removes the
/// possibility of providing an incorrect query layout and will be slightly faster since
/// it does not require argument reification.
///
/// ## Anatomy of the [`QueryComputer`]
///
/// This is similar to the [`DistanceComputer`] but has the extra duty of supporting
/// multiple different [`iface::QueryLayouts`] (compressions for the query). To that end,
/// the stack of types used to implement the underlying [`DynDistanceComputer`] trait is:
///
/// * Base [`DistanceFunction`] (e.g. [`CompensatedIP`]).
///
/// * Embedded inside [`Curried`] - which also contains a heap-allocated representation of
///   the query using the selected layout. For example, this could be one of.
///
///   - [`diskann_quantization::spherical::Query`]
///   - [`diskann_quantization::spherical::FullQuery`]
///   - [`diskann_quantization::sphericasl::Data`]
///
/// * Embedded inside [`Reify`] to convert [`Opaque`] to the correct type.
use std::marker::PhantomData;

use diskann_utils::{Reborrow, ReborrowMut};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};
use diskann_wide::{
    arch::{Scalar, Target1, Target2},
    Architecture,
};
#[cfg(feature = "flatbuffers")]
use flatbuffers::FlatBufferBuilder;
use thiserror::Error;

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::{
    quantizer, CompensatedCosine, CompensatedIP, CompensatedSquaredL2, Data, DataMut, DataRef,
    FullQuery, FullQueryMut, FullQueryRef, Query, QueryMut, QueryRef, SphericalQuantizer,
    SupportedMetric,
};
#[cfg(feature = "flatbuffers")]
use crate::{alloc::CompoundError, flatbuffers as fb};
use crate::{
    alloc::{
        Allocator, AllocatorCore, AllocatorError, GlobalAllocator, Poly, ScopedAllocator, TryClone,
    },
    bits::{self, Representation, Unsigned},
    distances::{self, UnequalLengths},
    error::InlineError,
    meta,
    num::PowerOfTwo,
    poly, AsFunctor, CompressIntoWith,
};

// A convenience definition to shorten the extensive where-clauses present in this file.
type Rf32 = distances::Result<f32>;

///////////////
// Quantizer //
///////////////

/// A description of the buffer size (in bytes) and alignment required for a compressed query.
#[derive(Debug, Clone)]
pub struct QueryBufferDescription {
    size: usize,
    align: PowerOfTwo,
}

impl QueryBufferDescription {
    /// Construct a new [`QueryBufferDescription`]
    pub fn new(size: usize, align: PowerOfTwo) -> Self {
        Self { size, align }
    }

    /// Return the number of bytes needed in a buffer for a compressed query.
    pub fn bytes(&self) -> usize {
        self.size
    }

    /// Return the necessary alignment of the base pointer for a query buffer.
    pub fn align(&self) -> PowerOfTwo {
        self.align
    }
}

/// A dyn-compatible trait providing a common interface for a bit-width specific
/// [`SphericalQuantizer`].
///
/// This allows us to have a single [`dyn Quantizer`] type without generics while still
/// supporting the range of bit-widths and query strategies we wish to support.
///
/// A level of indirection for each distance computation, unfortunately, is required to
/// support this. But we try to structure the code so there is only a single level of
/// indirection.
///
/// # Allocator
///
/// The quantizer is parameterized by the allocator provided used to acquire any necessary
/// memory for returned data structures.  The contract is as follows:
///
/// 1. Any allocation made as part of a returned data structure from a function will be
///    performed through the allocator given to that function.
///
/// 2. If dynamic memory allocation for scratch space is required, a separate `scratch`
///    allocator will be required and all scratch space allocations will go through that
///    allocator.
pub trait Quantizer<A = GlobalAllocator>: Send + Sync
where
    A: Allocator + std::panic::UnwindSafe + Send + Sync + 'static,
{
    /// The effective number of bits in the encoding.
    fn nbits(&self) -> usize;

    /// The number of bytes occupied by each compressed vector.
    fn bytes(&self) -> usize;

    /// The effective dimensionality of each compressed vector.
    fn dim(&self) -> usize;

    /// The dimensionality of the full-precision input vectors.
    fn full_dim(&self) -> usize;

    /// Return a distance computer capable on operating on validly initialized [`Opaque`]
    /// slices of length [`Self::bytes`].
    ///
    /// These slices should be initialized by [`Self::compress`].
    ///
    /// The query layout associated with this computer will always be
    /// [`QueryLayout::SameAsData`].
    fn distance_computer(&self, allocator: A) -> Result<DistanceComputer<A>, AllocatorError>;

    /// Return a scoped distance computer capable on operating on validly initialized
    /// [`Opaque`] slices of length [`Self::bytes`].
    ///
    /// These slices should be initialized by [`Self::compress`].
    fn distance_computer_ref(&self) -> &dyn DynDistanceComputer;

    /// A stand alone distance computer specialized for the specified query layout.
    ///
    /// Only layouts for which [`Self::is_supported`] returns `true` are supported.
    ///
    /// # Note
    ///
    /// The returned object will **only** be compatible with queries compressed using
    /// [`Self::compress_query`] using the same layout. If possible, the API
    /// [`Self::fused_query_computer`] should be used to avoid this ambiguity.
    fn query_computer(
        &self,
        layout: QueryLayout,
        allocator: A,
    ) -> Result<DistanceComputer<A>, DistanceComputerError>;

    /// Return the number of bytes and alignment of a buffer used to contain a compressed
    /// query with the provided layout.
    ///
    /// Only layouts for which [`Self::is_supported`] returns `true` are supported.
    fn query_buffer_description(
        &self,
        layout: QueryLayout,
    ) -> Result<QueryBufferDescription, UnsupportedQueryLayout>;

    /// Compress the query using the specified layout into `buffer`.
    ///
    /// This requires that buffer have the exact size and alignment as that returned from
    /// `query_buffer_description`.
    ///
    /// Only layouts for which [`Self::is_supported`] returns `true` are supported.
    fn compress_query(
        &self,
        x: &[f32],
        layout: QueryLayout,
        allow_rescale: bool,
        buffer: OpaqueMut<'_>,
        scratch: ScopedAllocator<'_>,
    ) -> Result<(), QueryCompressionError>;

    /// Return a query for the argument `x` capable on operating on validly initialized
    /// [`Opaque`] slices of length [`Self::bytes`].
    ///
    /// These slices should be initialized by [`Self::compress`].
    ///
    /// Note: Only layouts for which [`Self::is_supported`] returns `true` are supported.
    fn fused_query_computer(
        &self,
        x: &[f32],
        layout: QueryLayout,
        allow_rescale: bool,
        allocator: A,
        scratch: ScopedAllocator<'_>,
    ) -> Result<QueryComputer<A>, QueryComputerError>;

    /// Return whether or not this plan supports the given [`QueryLayout`].
    fn is_supported(&self, layout: QueryLayout) -> bool;

    /// Compress the vector `x` into the opaque slice.
    ///
    /// # Note
    ///
    /// This requires the length of the slice to be exactly [`Self::bytes`]. There is no
    /// alignment restriction on the base pointer.
    fn compress(
        &self,
        x: &[f32],
        into: OpaqueMut<'_>,
        scratch: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError>;

    /// Return the metric this plan was created with.
    fn metric(&self) -> SupportedMetric;

    /// Clone the backing object.
    fn try_clone_into(&self, allocator: A) -> Result<Poly<dyn Quantizer<A>, A>, AllocatorError>;

    crate::utils::features! {
        #![feature = "flatbuffers"]
        /// Serialize `self` into a flatbuffer, returning the flatbuffer. The function
        /// [`try_deserialize`] should undo this operation.
        fn serialize(&self, allocator: A) -> Result<Poly<[u8], A>, AllocatorError>;
    }
}

#[derive(Debug, Error)]
#[error("Layout {layout} is not supported for {desc}")]
pub struct UnsupportedQueryLayout {
    layout: QueryLayout,
    desc: &'static str,
}

impl UnsupportedQueryLayout {
    fn new(layout: QueryLayout, desc: &'static str) -> Self {
        Self { layout, desc }
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DistanceComputerError {
    #[error(transparent)]
    UnsupportedQueryLayout(#[from] UnsupportedQueryLayout),
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QueryCompressionError {
    #[error(transparent)]
    UnsupportedQueryLayout(#[from] UnsupportedQueryLayout),
    #[error(transparent)]
    CompressionError(#[from] CompressionError),
    #[error(transparent)]
    NotCanonical(#[from] NotCanonical),
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum QueryComputerError {
    #[error(transparent)]
    UnsupportedQueryLayout(#[from] UnsupportedQueryLayout),
    #[error(transparent)]
    CompressionError(#[from] CompressionError),
    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

/// Errors that can occur during data compression
#[derive(Debug, Error)]
#[error("Error occured during query compression")]
pub enum CompressionError {
    /// The input buffer did not have the expected layout. This is an input error.
    NotCanonical(#[source] InlineError<16>),

    /// Forward any error that occurs during the compression process.
    ///
    /// See [`quantizer::CompressionError`] for the complete list.
    CompressionError(#[source] quantizer::CompressionError),
}

impl CompressionError {
    fn not_canonical<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::NotCanonical(InlineError::new(error))
    }
}

#[derive(Debug, Error)]
#[error("An opaque argument did not have the required alignment or length")]
pub struct NotCanonical {
    source: Box<dyn std::error::Error + Send + Sync>,
}

impl NotCanonical {
    fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            source: Box::new(err),
        }
    }
}

////////////
// Opaque //
////////////

/// A type-erased slice wrapper used to hide the implementation of spherically quantized
/// vectors. This allows multiple bit-width implementations to share the same type.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Opaque<'a>(&'a [u8]);

impl<'a> Opaque<'a> {
    /// Construct a new `Opaque` referencing `slice`.
    pub fn new(slice: &'a [u8]) -> Self {
        Self(slice)
    }

    /// Consume `self`, returning the wrapped slice.
    pub fn into_inner(self) -> &'a [u8] {
        self.0
    }
}

impl std::ops::Deref for Opaque<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.0
    }
}
impl<'short> Reborrow<'short> for Opaque<'_> {
    type Target = Opaque<'short>;
    fn reborrow(&'short self) -> Self::Target {
        *self
    }
}

/// A type-erased slice wrapper used to hide the implementation of spherically quantized
/// vectors. This allows multiple bit-width implementations to share the same type.
#[derive(Debug)]
#[repr(transparent)]
pub struct OpaqueMut<'a>(&'a mut [u8]);

impl<'a> OpaqueMut<'a> {
    /// Construct a new `OpaqueMut` referencing `slice`.
    pub fn new(slice: &'a mut [u8]) -> Self {
        Self(slice)
    }

    /// Inspect the referenced slice.
    pub fn inspect(&mut self) -> &mut [u8] {
        self.0
    }
}

impl std::ops::Deref for OpaqueMut<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.0
    }
}

impl std::ops::DerefMut for OpaqueMut<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0
    }
}

//////////////////
// Query Layout //
//////////////////

/// The layout to use for the query in [`DistanceComputer`] and [`QueryComputer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryLayout {
    /// Use the same compression strategy as the data vectors.
    ///
    /// This may result in slow compression if high bit-widths are used.
    SameAsData,

    /// Use 4-bits for the query vector using a bitwise transpose layout.
    FourBitTransposed,

    /// Use scalar quantization for the query using the same number of bits per dimension
    /// as the dataset.
    ScalarQuantized,

    /// Use `f32` to encode the query.
    FullPrecision,
}

impl QueryLayout {
    #[cfg(test)]
    fn all() -> [Self; 4] {
        [
            Self::SameAsData,
            Self::FourBitTransposed,
            Self::ScalarQuantized,
            Self::FullPrecision,
        ]
    }
}

impl std::fmt::Display for QueryLayout {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, fmt)
    }
}

//////////////////////
// Layout Reporting //
//////////////////////

/// Because dynamic dispatch is used heavily in the implementation, it can be easy to lose
/// track of the actual layout used for the [`DistanceComputer`] and [`QueryComputer`].
///
/// This trait provides a mechanism by which we ensure the correct runtime layout is always
/// reported without requiring manual tracking.
trait ReportQueryLayout {
    fn report_query_layout(&self) -> QueryLayout;
}

impl<T, M, L, R> ReportQueryLayout for Reify<T, M, L, R>
where
    T: ReportQueryLayout,
{
    fn report_query_layout(&self) -> QueryLayout {
        self.inner.report_query_layout()
    }
}

impl<D, Q> ReportQueryLayout for Curried<D, Q>
where
    Q: ReportQueryLayout,
{
    fn report_query_layout(&self) -> QueryLayout {
        self.query.report_query_layout()
    }
}

impl<const NBITS: usize, A> ReportQueryLayout for Data<NBITS, A>
where
    Unsigned: Representation<NBITS>,
    A: AllocatorCore,
{
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::SameAsData
    }
}

impl<const NBITS: usize, A> ReportQueryLayout for Query<NBITS, bits::Dense, A>
where
    Unsigned: Representation<NBITS>,
    A: AllocatorCore,
{
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::ScalarQuantized
    }
}

impl<A> ReportQueryLayout for Query<4, bits::BitTranspose, A>
where
    A: AllocatorCore,
{
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::FourBitTransposed
    }
}

impl<A> ReportQueryLayout for FullQuery<A>
where
    A: AllocatorCore,
{
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::FullPrecision
    }
}

//-----------------------//
// Reification Utilities //
//-----------------------//

/// An adaptor trait defining how to go from an `Opaque` slice to a fully reified type.
///
/// THis is the building block for building distance computers with the reificiation code
/// inlined into the callsite.
trait FromOpaque: 'static + Send + Sync {
    type Target<'a>;
    type Error: std::error::Error + Send + Sync + 'static;

    fn from_opaque<'a>(query: Opaque<'a>, dim: usize) -> Result<Self::Target<'a>, Self::Error>;
}

/// Reify as full-precision.
#[derive(Debug, Default)]
pub(super) struct AsFull;

/// Reify as data.
#[derive(Debug, Default)]
pub(super) struct AsData<const NBITS: usize>;

/// Reify as scalar quantized query.
#[derive(Debug)]
pub(super) struct AsQuery<const NBITS: usize, Perm = bits::Dense> {
    _marker: PhantomData<Perm>,
}

// This impelmentation works around the `derive` impl requiring `Perm: Default`.
impl<const NBITS: usize, Perm> Default for AsQuery<NBITS, Perm> {
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl FromOpaque for AsFull {
    type Target<'a> = FullQueryRef<'a>;
    type Error = meta::slice::NotCanonical;

    fn from_opaque<'a>(query: Opaque<'a>, dim: usize) -> Result<Self::Target<'a>, Self::Error> {
        Self::Target::from_canonical(query.into_inner(), dim)
    }
}

impl ReportQueryLayout for AsFull {
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::FullPrecision
    }
}

impl<const NBITS: usize> FromOpaque for AsData<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Target<'a> = DataRef<'a, NBITS>;
    type Error = meta::NotCanonical;

    fn from_opaque<'a>(query: Opaque<'a>, dim: usize) -> Result<Self::Target<'a>, Self::Error> {
        Self::Target::from_canonical_back(query.into_inner(), dim)
    }
}

impl<const NBITS: usize> ReportQueryLayout for AsData<NBITS> {
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::SameAsData
    }
}

impl<const NBITS: usize, Perm> FromOpaque for AsQuery<NBITS, Perm>
where
    Unsigned: Representation<NBITS>,
    Perm: bits::PermutationStrategy<NBITS> + Send + Sync + 'static,
{
    type Target<'a> = QueryRef<'a, NBITS, Perm>;
    type Error = meta::NotCanonical;

    fn from_opaque<'a>(query: Opaque<'a>, dim: usize) -> Result<Self::Target<'a>, Self::Error> {
        Self::Target::from_canonical_back(query.into_inner(), dim)
    }
}

impl<const NBITS: usize> ReportQueryLayout for AsQuery<NBITS, bits::Dense> {
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::ScalarQuantized
    }
}

impl<const NBITS: usize> ReportQueryLayout for AsQuery<NBITS, bits::BitTranspose> {
    fn report_query_layout(&self) -> QueryLayout {
        QueryLayout::FourBitTransposed
    }
}

//-------//
// Reify //
//-------//

/// Helper struct to convert an [`Opaque`] to a fully-typed [`DataRef`].
pub(super) struct Reify<T, M, L, R> {
    inner: T,
    dim: usize,
    arch: M,
    _markers: PhantomData<(L, R)>,
}

impl<T, M, L, R> Reify<T, M, L, R> {
    pub(super) fn new(inner: T, dim: usize, arch: M) -> Self {
        Self {
            inner,
            dim,
            arch,
            _markers: PhantomData,
        }
    }
}

impl<M, T, R> DynQueryComputer for Reify<T, M, (), R>
where
    M: Architecture,
    R: FromOpaque,
    T: ReportQueryLayout + Send + Sync,
    for<'a> &'a T: Target1<M, Rf32, R::Target<'a>>,
{
    fn evaluate(&self, x: Opaque<'_>) -> Result<f32, QueryDistanceError> {
        self.arch.run2(
            |this: &Self, x| {
                let x = R::from_opaque(x, this.dim)
                    .map_err(|err| QueryDistanceError::XReify(InlineError::new(err)))?;
                this.arch
                    .run1(&this.inner, x)
                    .map_err(QueryDistanceError::UnequalLengths)
            },
            self,
            x,
        )
    }

    fn layout(&self) -> QueryLayout {
        self.inner.report_query_layout()
    }
}

impl<T, M, Q, R> DynDistanceComputer for Reify<T, M, Q, R>
where
    M: Architecture,
    Q: FromOpaque + Default + ReportQueryLayout,
    R: FromOpaque,
    T: for<'a> Target2<M, Rf32, Q::Target<'a>, R::Target<'a>> + Copy + Send + Sync,
{
    fn evaluate(&self, query: Opaque<'_>, x: Opaque<'_>) -> Result<f32, DistanceError> {
        self.arch.run3(
            |this: &Self, query, x| {
                let query = Q::from_opaque(query, this.dim)
                    .map_err(|err| DistanceError::QueryReify(InlineError::<24>::new(err)))?;

                let x = R::from_opaque(x, this.dim)
                    .map_err(|err| DistanceError::XReify(InlineError::<16>::new(err)))?;

                this.arch
                    .run2_inline(this.inner, query, x)
                    .map_err(DistanceError::UnequalLengths)
            },
            self,
            query,
            x,
        )
    }

    fn layout(&self) -> QueryLayout {
        Q::default().report_query_layout()
    }
}

///////////////////////
// Query Computation //
///////////////////////

/// Errors that can occur while perfoming distance cacluations on opaque vectors.
#[derive(Debug, Error)]
pub enum QueryDistanceError {
    /// The right-hand data argument appears to be malformed.
    #[error("trouble trying to reify the argument")]
    XReify(#[source] InlineError<16>),

    /// Distance computation failed because the logical lengths of the two vectors differ.
    #[error("encountered while trying to compute distances")]
    UnequalLengths(#[source] UnequalLengths),
}

pub trait DynQueryComputer: Send + Sync {
    fn evaluate(&self, x: Opaque<'_>) -> Result<f32, QueryDistanceError>;
    fn layout(&self) -> QueryLayout;
}

/// An opaque [`PreprocessedDistanceFunction`] for the [`Quantizer`] trait object.
///
/// # Note
///
/// This is only valid to call on [`Opaque`] slices compressed by the same [`Quantizer`] that
/// created the computer.
///
/// Otherwise, distance computations may return garbage values or panic.
pub struct QueryComputer<A = GlobalAllocator>
where
    A: AllocatorCore,
{
    inner: Poly<dyn DynQueryComputer, A>,
}

impl<A> QueryComputer<A>
where
    A: AllocatorCore,
{
    fn new<T>(inner: T, allocator: A) -> Result<Self, AllocatorError>
    where
        T: DynQueryComputer + 'static,
    {
        let inner = Poly::new(inner, allocator)?;
        Ok(Self {
            inner: poly!(DynQueryComputer, inner),
        })
    }

    /// Report the layout used by the query computer.
    pub fn layout(&self) -> QueryLayout {
        self.inner.layout()
    }

    /// This is a temporary function until custom allocator support fully comes on line.
    pub fn into_inner(self) -> Poly<dyn DynQueryComputer, A> {
        self.inner
    }
}

impl<A> std::fmt::Debug for QueryComputer<A>
where
    A: AllocatorCore,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dynamic fused query computer with layout \"{}\"",
            self.layout()
        )
    }
}

impl<A> PreprocessedDistanceFunction<Opaque<'_>, Result<f32, QueryDistanceError>>
    for QueryComputer<A>
where
    A: AllocatorCore,
{
    fn evaluate_similarity(&self, x: Opaque<'_>) -> Result<f32, QueryDistanceError> {
        self.inner.evaluate(x)
    }
}

/// To handle multiple query bit-widths, we use type erasure on the actual distance
/// function implementation.
///
/// This struct represents the partial application of the `inner` distance function with
/// `query` in a generic way so we only have one level of dynamic dispatch when computing
/// distances.
pub(super) struct Curried<D, Q> {
    inner: D,
    query: Q,
}

impl<D, Q> Curried<D, Q> {
    pub(super) fn new(inner: D, query: Q) -> Self {
        Self { inner, query }
    }
}

impl<A, D, Q, T, R> Target1<A, R, T> for &Curried<D, Q>
where
    A: Architecture,
    Q: for<'a> Reborrow<'a>,
    D: for<'a> Target2<A, R, <Q as Reborrow<'a>>::Target, T> + Copy,
{
    fn run(self, arch: A, x: T) -> R {
        self.inner.run(arch, self.query.reborrow(), x)
    }
}

///////////////////////
// Distance Computer //
///////////////////////

/// Errors that can occur while perfoming distance cacluations on opaque vectors.
#[derive(Debug, Error)]
pub enum DistanceError {
    /// The left-hand data argument appears to be malformed.
    #[error("trouble trying to reify the left-hand argument")]
    QueryReify(InlineError<24>),

    /// The right-hand data argument appears to be malformed.
    #[error("trouble trying to reify the right-hand argument")]
    XReify(InlineError<16>),

    /// Distance computation failed because the logical lengths of the two vectors differ.
    ///
    /// If vector reificiation occurs successfully, then this should not be returned.
    #[error("encountered while trying to compute distances")]
    UnequalLengths(UnequalLengths),
}

pub trait DynDistanceComputer: Send + Sync {
    fn evaluate(&self, query: Opaque<'_>, x: Opaque<'_>) -> Result<f32, DistanceError>;
    fn layout(&self) -> QueryLayout;
}

/// An opaque [`DistanceFunction`] for the [`Quantizer`] trait object.
///
/// # Note
///
/// Left-hand arguments must be [`Opaque`] slices compressed using
/// [`Quantizer::compress_query`] using [`Self::layout`].
///
/// Right-hand arguments must be [`Opaque`] slices compressed using [`Quantizer::compress`].
///
/// Otherwise, distance computations may return garbage values or panic.
pub struct DistanceComputer<A = GlobalAllocator>
where
    A: AllocatorCore,
{
    inner: Poly<dyn DynDistanceComputer, A>,
}

impl<A> DistanceComputer<A>
where
    A: AllocatorCore,
{
    pub(super) fn new<T>(inner: T, allocator: A) -> Result<Self, AllocatorError>
    where
        T: DynDistanceComputer + 'static,
    {
        let inner = Poly::new(inner, allocator)?;
        Ok(Self {
            inner: poly!(DynDistanceComputer, inner),
        })
    }

    /// Report the layout used by the query computer.
    pub fn layout(&self) -> QueryLayout {
        self.inner.layout()
    }

    pub fn into_inner(self) -> Poly<dyn DynDistanceComputer, A> {
        self.inner
    }
}

impl<A> std::fmt::Debug for DistanceComputer<A>
where
    A: AllocatorCore,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "dynamic distance computer with layout \"{}\"",
            self.layout()
        )
    }
}

impl<A> DistanceFunction<Opaque<'_>, Opaque<'_>, Result<f32, DistanceError>> for DistanceComputer<A>
where
    A: AllocatorCore,
{
    fn evaluate_similarity(&self, query: Opaque<'_>, x: Opaque<'_>) -> Result<f32, DistanceError> {
        self.inner.evaluate(query, x)
    }
}

//////////
// Impl //
//////////

/// The base number of bytes to allocate when attempting to serialize a quantizer.
#[cfg(all(not(test), feature = "flatbuffers"))]
const DEFAULT_SERIALIZED_BYTES: usize = 1024;

// When testing, use a small value so we trigger the reallocation logic.
#[cfg(all(test, feature = "flatbuffers"))]
const DEFAULT_SERIALIZED_BYTES: usize = 1;

/// Implementation for [`Quantizer`] specializing on the number of bits used for data
/// compression.
pub struct Impl<const NBITS: usize, A = GlobalAllocator>
where
    A: Allocator,
{
    quantizer: SphericalQuantizer<A>,
    distance: Poly<dyn DynDistanceComputer, A>,
}

/// Pre-dispatch distance functions between compressed data vectors from `quantizer`
/// specialized for the current run-time mciro architecture.
pub trait Constructible<A = GlobalAllocator>
where
    A: Allocator,
{
    fn dispatch_distance(
        quantizer: &SphericalQuantizer<A>,
    ) -> Result<Poly<dyn DynDistanceComputer, A>, AllocatorError>;
}

impl<const NBITS: usize, A: Allocator> Constructible<A> for Impl<NBITS, A>
where
    A: Allocator,
    AsData<NBITS>: FromOpaque,
    SphericalQuantizer<A>: Dispatchable<AsData<NBITS>, NBITS>,
{
    fn dispatch_distance(
        quantizer: &SphericalQuantizer<A>,
    ) -> Result<Poly<dyn DynDistanceComputer, A>, AllocatorError> {
        diskann_wide::arch::dispatch2_no_features(
            ComputerDispatcher::<AsData<NBITS>, NBITS>::new(),
            quantizer,
            quantizer.allocator().clone(),
        )
        .map(|obj| obj.inner)
    }
}

impl<const NBITS: usize, A> TryClone for Impl<NBITS, A>
where
    A: Allocator,
    AsData<NBITS>: FromOpaque,
    SphericalQuantizer<A>: Dispatchable<AsData<NBITS>, NBITS>,
{
    fn try_clone(&self) -> Result<Self, AllocatorError> {
        Self::new(self.quantizer.try_clone()?)
    }
}

impl<const NBITS: usize, A: Allocator> Impl<NBITS, A> {
    /// Construct a new plan around `quantizer` providing distance computers for `metric`.
    pub fn new(quantizer: SphericalQuantizer<A>) -> Result<Self, AllocatorError>
    where
        Self: Constructible<A>,
    {
        let distance = Self::dispatch_distance(&quantizer)?;
        Ok(Self {
            quantizer,
            distance,
        })
    }

    /// Return the underlying [`SphericalQuantizer`].
    pub fn quantizer(&self) -> &SphericalQuantizer<A> {
        &self.quantizer
    }

    /// Return `true` if this plan supports `layout` for query computers.
    ///
    /// Otherwise, return `false`.
    pub fn supports(layout: QueryLayout) -> bool {
        if const { NBITS == 1 } {
            [
                QueryLayout::SameAsData,
                QueryLayout::FourBitTransposed,
                QueryLayout::FullPrecision,
            ]
            .contains(&layout)
        } else {
            [
                QueryLayout::SameAsData,
                QueryLayout::ScalarQuantized,
                QueryLayout::FullPrecision,
            ]
            .contains(&layout)
        }
    }

    /// Return a [`DistanceComputer`] that is specialized for the most specific runtime
    /// architecture.
    fn query_computer<Q, B>(&self, allocator: B) -> Result<DistanceComputer<B>, AllocatorError>
    where
        Q: FromOpaque,
        B: AllocatorCore,
        SphericalQuantizer<A>: Dispatchable<Q, NBITS>,
    {
        diskann_wide::arch::dispatch2_no_features(
            ComputerDispatcher::<Q, NBITS>::new(),
            &self.quantizer,
            allocator,
        )
    }

    fn compress_query<'a, T>(
        &self,
        query: &'a [f32],
        storage: T,
        scratch: ScopedAllocator<'a>,
    ) -> Result<(), QueryCompressionError>
    where
        SphericalQuantizer<A>: CompressIntoWith<
            &'a [f32],
            T,
            ScopedAllocator<'a>,
            Error = quantizer::CompressionError,
        >,
    {
        self.quantizer
            .compress_into_with(query, storage, scratch)
            .map_err(|err| CompressionError::CompressionError(err).into())
    }

    /// Return a [`QueryComputer`] that is specialized for the most specific runtime
    /// architecture.
    fn fused_query_computer<Q, T, B>(
        &self,
        query: &[f32],
        mut storage: T,
        allocator: B,
        scratch: ScopedAllocator<'_>,
    ) -> Result<QueryComputer<B>, QueryComputerError>
    where
        Q: FromOpaque,
        T: for<'a> ReborrowMut<'a>
            + for<'a> Reborrow<'a, Target = Q::Target<'a>>
            + ReportQueryLayout
            + Send
            + Sync
            + 'static,
        B: AllocatorCore,
        SphericalQuantizer<A>: for<'a> CompressIntoWith<
            &'a [f32],
            <T as ReborrowMut<'a>>::Target,
            ScopedAllocator<'a>,
            Error = quantizer::CompressionError,
        >,
        SphericalQuantizer<A>: Dispatchable<Q, NBITS>,
    {
        if let Err(err) = self
            .quantizer
            .compress_into_with(query, storage.reborrow_mut(), scratch)
        {
            return Err(CompressionError::CompressionError(err).into());
        }

        diskann_wide::arch::dispatch3_no_features(
            ComputerDispatcher::<Q, NBITS>::new(),
            &self.quantizer,
            storage,
            allocator,
        )
        .map_err(|e| e.into())
    }

    #[cfg(feature = "flatbuffers")]
    fn serialize<B>(&self, allocator: B) -> Result<Poly<[u8], B>, AllocatorError>
    where
        B: Allocator + std::panic::UnwindSafe,
        A: std::panic::RefUnwindSafe,
    {
        let mut buf = FlatBufferBuilder::new_in(Poly::broadcast(
            0u8,
            DEFAULT_SERIALIZED_BYTES,
            allocator.clone(),
        )?);

        let quantizer = &self.quantizer;

        let (root, mut buf) = match std::panic::catch_unwind(move || {
            let offset = quantizer.pack(&mut buf);

            let root = fb::spherical::Quantizer::create(
                &mut buf,
                &fb::spherical::QuantizerArgs {
                    quantizer: Some(offset),
                    nbits: NBITS as u32,
                },
            );
            (root, buf)
        }) {
            Ok(ret) => ret,
            Err(err) => match err.downcast_ref::<String>() {
                Some(msg) => {
                    if msg.contains("AllocatorError") {
                        return Err(AllocatorError);
                    } else {
                        std::panic::resume_unwind(err);
                    }
                }
                None => std::panic::resume_unwind(err),
            },
        };

        // Finish serializing and then copy out the finished data into a newly allocated buffer.
        fb::spherical::finish_quantizer_buffer(&mut buf, root);
        Poly::from_iter(buf.finished_data().iter().copied(), allocator)
    }
}

//----------------------//
// Distance Dispatching //
//----------------------//

/// This trait and [`ComputerDispatcher`] are the glue for pre-dispatching
/// micro-architecture compatibility of distance computers.
///
/// This trait takes
///
/// * `M`: The target micro-architecture.
/// * `Q`: The target query type
///
/// And generates a specialized `DistanceComputer` and `QueryComputer`.
///
/// The [`ComputerDispatcher`] struct implements the [`diskann_wide::arch::Target2`] and
/// [`diskann_wide::arch::Target3`] traits to do the architecture-dispatching.
trait BuildComputer<M, Q, const N: usize>
where
    M: Architecture,
    Q: FromOpaque,
{
    /// Build a [`DistanceComputer`] targeting the micro-architecture `M`.
    ///
    /// The resulting object should implement distance calculations using just a single
    /// level of indirection.
    fn build_computer<A>(
        &self,
        arch: M,
        allocator: A,
    ) -> Result<DistanceComputer<A>, AllocatorError>
    where
        A: AllocatorCore;

    /// Build a [`DistanceComputer`] with `query` targeting the micro-architecture `M`.
    ///
    /// The resulting object should implement distance calculations using just a single
    /// level of indirection.
    fn build_fused_computer<R, A>(
        &self,
        arch: M,
        query: R,
        allocator: A,
    ) -> Result<QueryComputer<A>, AllocatorError>
    where
        R: ReportQueryLayout + for<'a> Reborrow<'a, Target = Q::Target<'a>> + Send + Sync + 'static,
        A: AllocatorCore;
}

fn identity<T>(x: T) -> T {
    x
}

macro_rules! dispatch_map {
    ($N:literal, $Q:ty, $arch:ty) => {
        dispatch_map!($N, $Q, $arch, identity);
    };
    ($N:literal, $Q:ty, $arch:ty, $op:ident) => {
        impl<A> BuildComputer<$arch, $Q, $N> for SphericalQuantizer<A>
        where
            A: Allocator,
        {
            fn build_computer<B>(
                &self,
                input_arch: $arch,
                allocator: B,
            ) -> Result<DistanceComputer<B>, AllocatorError>
            where
                B: AllocatorCore,
            {
                type D = AsData<$N>;

                // Perform any architecture down-casting.
                let arch = ($op)(input_arch);
                let dim = self.output_dim();
                match self.metric() {
                    SupportedMetric::SquaredL2 => {
                        let reify = Reify::<CompensatedSquaredL2, _, $Q, D>::new(
                            self.as_functor(),
                            dim,
                            arch,
                        );
                        DistanceComputer::new(reify, allocator)
                    }
                    SupportedMetric::InnerProduct => {
                        let reify =
                            Reify::<CompensatedIP, _, $Q, D>::new(self.as_functor(), dim, arch);
                        DistanceComputer::new(reify, allocator)
                    }
                    SupportedMetric::Cosine => {
                        let reify =
                            Reify::<CompensatedCosine, _, $Q, D>::new(self.as_functor(), dim, arch);
                        DistanceComputer::new(reify, allocator)
                    }
                }
            }

            fn build_fused_computer<R, B>(
                &self,
                input_arch: $arch,
                query: R,
                allocator: B,
            ) -> Result<QueryComputer<B>, AllocatorError>
            where
                R: ReportQueryLayout
                    + for<'a> Reborrow<'a, Target = <$Q as FromOpaque>::Target<'a>>
                    + Send
                    + Sync
                    + 'static,
                B: AllocatorCore,
            {
                type D = AsData<$N>;
                let arch = ($op)(input_arch);
                let dim = self.output_dim();
                match self.metric() {
                    SupportedMetric::SquaredL2 => {
                        let computer: CompensatedSquaredL2 = self.as_functor();
                        let curried = Curried::new(computer, query);
                        let reify = Reify::<_, _, (), D>::new(curried, dim, arch);
                        Ok(QueryComputer::new(reify, allocator)?)
                    }
                    SupportedMetric::InnerProduct => {
                        let computer: CompensatedIP = self.as_functor();
                        let curried = Curried::new(computer, query);
                        let reify = Reify::<_, _, (), D>::new(curried, dim, arch);
                        Ok(QueryComputer::new(reify, allocator)?)
                    }
                    SupportedMetric::Cosine => {
                        let computer: CompensatedCosine = self.as_functor();
                        let curried = Curried::new(computer, query);
                        let reify = Reify::<_, _, (), D>::new(curried, dim, arch);
                        Ok(QueryComputer::new(reify, allocator)?)
                    }
                }
            }
        }
    };
}

dispatch_map!(1, AsFull, Scalar);
dispatch_map!(2, AsFull, Scalar);
dispatch_map!(4, AsFull, Scalar);
dispatch_map!(8, AsFull, Scalar);

dispatch_map!(1, AsData<1>, Scalar);
dispatch_map!(2, AsData<2>, Scalar);
dispatch_map!(4, AsData<4>, Scalar);
dispatch_map!(8, AsData<8>, Scalar);

// Special Cases
dispatch_map!(1, AsQuery<4, bits::BitTranspose>, Scalar);
dispatch_map!(2, AsQuery<2>, Scalar);
dispatch_map!(4, AsQuery<4>, Scalar);
dispatch_map!(8, AsQuery<8>, Scalar);

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86_64")] {
        fn downcast_to_v3(arch: V4) -> V3 {
            arch.into()
        }

        // V3
        dispatch_map!(1, AsFull, V3);
        dispatch_map!(2, AsFull, V3);
        dispatch_map!(4, AsFull, V3);
        dispatch_map!(8, AsFull, V3);

        dispatch_map!(1, AsData<1>, V3);
        dispatch_map!(2, AsData<2>, V3);
        dispatch_map!(4, AsData<4>, V3);
        dispatch_map!(8, AsData<8>, V3);

        dispatch_map!(1, AsQuery<4, bits::BitTranspose>, V3);
        dispatch_map!(2, AsQuery<2>, V3);
        dispatch_map!(4, AsQuery<4>, V3);
        dispatch_map!(8, AsQuery<8>, V3);

        // V4
        dispatch_map!(1, AsFull, V4, downcast_to_v3);
        dispatch_map!(2, AsFull, V4, downcast_to_v3);
        dispatch_map!(4, AsFull, V4, downcast_to_v3);
        dispatch_map!(8, AsFull, V4, downcast_to_v3);

        dispatch_map!(1, AsData<1>, V4, downcast_to_v3);
        dispatch_map!(2, AsData<2>, V4); // specialized
        dispatch_map!(4, AsData<4>, V4, downcast_to_v3);
        dispatch_map!(8, AsData<8>, V4, downcast_to_v3);

        dispatch_map!(1, AsQuery<4, bits::BitTranspose>, V4, downcast_to_v3);
        dispatch_map!(2, AsQuery<2>, V4); // specialized
        dispatch_map!(4, AsQuery<4>, V4, downcast_to_v3);
        dispatch_map!(8, AsQuery<8>, V4, downcast_to_v3);
    }
}

/// This struct and the [`BuildComputer`] trait are the glue for pre-dispatching
/// micro-architecture compatibility of distance computers.
///
/// This trait takes
///
/// * `Q`: The target query type.
/// * `N`: The nubmer of data bits to target.
///
/// This struct implements [`diskann_wide::arch::Target2`] and
/// [`diskann_wide::arch::Target3`] traits to do the architecture-dispatching, relying on
/// `Impl<N, A> as BuildQueryComputer` for the implementation.
#[derive(Debug, Clone, Copy)]
struct ComputerDispatcher<Q, const N: usize> {
    _query_type: std::marker::PhantomData<Q>,
}

impl<Q, const N: usize> ComputerDispatcher<Q, N> {
    fn new() -> Self {
        Self {
            _query_type: std::marker::PhantomData,
        }
    }
}

impl<M, const N: usize, A, B, Q>
    diskann_wide::arch::Target2<
        M,
        Result<DistanceComputer<B>, AllocatorError>,
        &SphericalQuantizer<A>,
        B,
    > for ComputerDispatcher<Q, N>
where
    M: Architecture,
    A: Allocator,
    B: AllocatorCore,
    Q: FromOpaque,
    SphericalQuantizer<A>: BuildComputer<M, Q, N>,
{
    fn run(
        self,
        arch: M,
        quantizer: &SphericalQuantizer<A>,
        allocator: B,
    ) -> Result<DistanceComputer<B>, AllocatorError> {
        quantizer.build_computer(arch, allocator)
    }
}

impl<M, const N: usize, A, R, B, Q>
    diskann_wide::arch::Target3<
        M,
        Result<QueryComputer<B>, AllocatorError>,
        &SphericalQuantizer<A>,
        R,
        B,
    > for ComputerDispatcher<Q, N>
where
    M: Architecture,
    A: Allocator,
    B: AllocatorCore,
    Q: FromOpaque,
    R: ReportQueryLayout + for<'a> Reborrow<'a, Target = Q::Target<'a>> + Send + Sync + 'static,
    SphericalQuantizer<A>: BuildComputer<M, Q, N>,
{
    fn run(
        self,
        arch: M,
        quantizer: &SphericalQuantizer<A>,
        query: R,
        allocator: B,
    ) -> Result<QueryComputer<B>, AllocatorError> {
        quantizer.build_fused_computer(arch, query, allocator)
    }
}

#[cfg(target_arch = "x86_64")]
trait Dispatchable<Q, const N: usize>:
    BuildComputer<Scalar, Q, N> + BuildComputer<V3, Q, N> + BuildComputer<V4, Q, N>
where
    Q: FromOpaque,
{
}

#[cfg(target_arch = "x86_64")]
impl<Q, const N: usize, T> Dispatchable<Q, N> for T
where
    Q: FromOpaque,
    T: BuildComputer<Scalar, Q, N> + BuildComputer<V3, Q, N> + BuildComputer<V4, Q, N>,
{
}

#[cfg(not(target_arch = "x86_64"))]
trait Dispatchable<Q, const N: usize>: BuildComputer<Scalar, Q, N>
where
    Q: FromOpaque,
{
}

#[cfg(not(target_arch = "x86_64"))]
impl<Q, const N: usize, T> Dispatchable<Q, N> for T
where
    Q: FromOpaque,
    T: BuildComputer<Scalar, Q, N>,
{
}

//---------------------------//
// Quantizer Implementations //
//---------------------------//

impl<A, B> Quantizer<B> for Impl<1, A>
where
    A: Allocator + std::panic::RefUnwindSafe + Send + Sync + 'static,
    B: Allocator + std::panic::UnwindSafe + Send + Sync + 'static,
{
    fn nbits(&self) -> usize {
        1
    }

    fn dim(&self) -> usize {
        self.quantizer.output_dim()
    }

    fn full_dim(&self) -> usize {
        self.quantizer.input_dim()
    }

    fn bytes(&self) -> usize {
        DataRef::<1>::canonical_bytes(self.quantizer.output_dim())
    }

    fn distance_computer(&self, allocator: B) -> Result<DistanceComputer<B>, AllocatorError> {
        self.query_computer::<AsData<1>, _>(allocator)
    }

    fn distance_computer_ref(&self) -> &dyn DynDistanceComputer {
        &*self.distance
    }

    fn query_computer(
        &self,
        layout: QueryLayout,
        allocator: B,
    ) -> Result<DistanceComputer<B>, DistanceComputerError> {
        match layout {
            QueryLayout::SameAsData => Ok(self.query_computer::<AsData<1>, _>(allocator)?),
            QueryLayout::FourBitTransposed => {
                Ok(self.query_computer::<AsQuery<4, bits::BitTranspose>, _>(allocator)?)
            }
            QueryLayout::ScalarQuantized => {
                Err(UnsupportedQueryLayout::new(layout, "1-bit compression").into())
            }
            QueryLayout::FullPrecision => Ok(self.query_computer::<AsFull, _>(allocator)?),
        }
    }

    fn query_buffer_description(
        &self,
        layout: QueryLayout,
    ) -> Result<QueryBufferDescription, UnsupportedQueryLayout> {
        let dim = <Self as Quantizer<B>>::dim(self);
        match layout {
            QueryLayout::SameAsData => Ok(QueryBufferDescription::new(
                DataRef::<1>::canonical_bytes(dim),
                PowerOfTwo::alignment_of::<u8>(),
            )),
            QueryLayout::FourBitTransposed => Ok(QueryBufferDescription::new(
                QueryRef::<4, bits::BitTranspose>::canonical_bytes(dim),
                PowerOfTwo::alignment_of::<u8>(),
            )),
            QueryLayout::ScalarQuantized => {
                Err(UnsupportedQueryLayout::new(layout, "1-bit compression"))
            }
            QueryLayout::FullPrecision => Ok(QueryBufferDescription::new(
                FullQueryRef::canonical_bytes(dim),
                FullQueryRef::canonical_align(),
            )),
        }
    }

    fn compress_query(
        &self,
        x: &[f32],
        layout: QueryLayout,
        allow_rescale: bool,
        mut buffer: OpaqueMut<'_>,
        scratch: ScopedAllocator<'_>,
    ) -> Result<(), QueryCompressionError> {
        let dim = <Self as Quantizer<B>>::dim(self);
        let mut finish = |v: &[f32]| -> Result<(), QueryCompressionError> {
            match layout {
                QueryLayout::SameAsData => self.compress_query(
                    v,
                    DataMut::<1>::from_canonical_back_mut(&mut buffer, dim)
                        .map_err(NotCanonical::new)?,
                    scratch,
                ),
                QueryLayout::FourBitTransposed => self.compress_query(
                    v,
                    QueryMut::<4, bits::BitTranspose>::from_canonical_back_mut(&mut buffer, dim)
                        .map_err(NotCanonical::new)?,
                    scratch,
                ),
                QueryLayout::ScalarQuantized => {
                    Err(UnsupportedQueryLayout::new(layout, "1-bit compression").into())
                }
                QueryLayout::FullPrecision => self.compress_query(
                    v,
                    FullQueryMut::from_canonical_mut(&mut buffer, dim)
                        .map_err(NotCanonical::new)?,
                    scratch,
                ),
            }
        };

        if allow_rescale && self.quantizer.metric() == SupportedMetric::InnerProduct {
            let mut copy = x.to_owned();
            self.quantizer.rescale(&mut copy);
            finish(&copy)
        } else {
            finish(x)
        }
    }

    fn fused_query_computer(
        &self,
        x: &[f32],
        layout: QueryLayout,
        allow_rescale: bool,
        allocator: B,
        scratch: ScopedAllocator<'_>,
    ) -> Result<QueryComputer<B>, QueryComputerError> {
        let dim = <Self as Quantizer<B>>::dim(self);
        let finish = |v: &[f32], allocator: B| -> Result<QueryComputer<B>, QueryComputerError> {
            match layout {
                    QueryLayout::SameAsData => self.fused_query_computer::<AsData<1>, Data<1, _>, _>(
                        v,
                        Data::new_in(dim, allocator.clone())?,
                        allocator,
                        scratch,
                    ),
                    QueryLayout::FourBitTransposed => self
                        .fused_query_computer::<AsQuery<4, bits::BitTranspose>, Query<4, bits::BitTranspose, _>, _>(
                            v,
                            Query::new_in(dim, allocator.clone())?,
                            allocator,
                            scratch,
                        ),
                    QueryLayout::ScalarQuantized => {
                        Err(UnsupportedQueryLayout::new(layout, "1-bit compression").into())
                    }
                    QueryLayout::FullPrecision => self.fused_query_computer::<AsFull, FullQuery<_>, _>(
                        v,
                        FullQuery::empty(dim, allocator.clone())?,
                        allocator,
                        scratch,
                    ),
                }
        };

        if allow_rescale && self.quantizer.metric() == SupportedMetric::InnerProduct {
            let mut copy = x.to_owned();
            self.quantizer.rescale(&mut copy);
            finish(&copy, allocator)
        } else {
            finish(x, allocator)
        }
    }

    fn is_supported(&self, layout: QueryLayout) -> bool {
        Self::supports(layout)
    }

    fn compress(
        &self,
        x: &[f32],
        mut into: OpaqueMut<'_>,
        scratch: ScopedAllocator<'_>,
    ) -> Result<(), CompressionError> {
        let dim = <Self as Quantizer<B>>::dim(self);
        let into = DataMut::<1>::from_canonical_back_mut(into.inspect(), dim)
            .map_err(CompressionError::not_canonical)?;
        self.quantizer
            .compress_into_with(x, into, scratch)
            .map_err(CompressionError::CompressionError)
    }

    fn metric(&self) -> SupportedMetric {
        self.quantizer.metric()
    }

    fn try_clone_into(&self, allocator: B) -> Result<Poly<dyn Quantizer<B>, B>, AllocatorError> {
        let clone = (*self).try_clone()?;
        poly!({ Quantizer<B> }, clone, allocator)
    }

    #[cfg(feature = "flatbuffers")]
    fn serialize(&self, allocator: B) -> Result<Poly<[u8], B>, AllocatorError> {
        Impl::<1, A>::serialize(self, allocator)
    }
}

macro_rules! plan {
    ($N:literal) => {
        impl<A, B> Quantizer<B> for Impl<$N, A>
        where
            A: Allocator + std::panic::RefUnwindSafe + Send + Sync + 'static,
            B: Allocator + std::panic::UnwindSafe + Send + Sync + 'static,
        {
            fn nbits(&self) -> usize {
                $N
            }

            fn dim(&self) -> usize {
                self.quantizer.output_dim()
            }

            fn full_dim(&self) -> usize {
                self.quantizer.input_dim()
            }

            fn bytes(&self) -> usize {
                DataRef::<$N>::canonical_bytes(<Self as Quantizer<B>>::dim(self))
            }

            fn distance_computer(
                &self,
                allocator: B
            ) -> Result<DistanceComputer<B>, AllocatorError> {
                self.query_computer::<AsData<$N>, _>(allocator)
            }

            fn distance_computer_ref(&self) -> &dyn DynDistanceComputer {
                &*self.distance
            }

            fn query_computer(
                &self,
                layout: QueryLayout,
                allocator: B,
            ) -> Result<DistanceComputer<B>, DistanceComputerError> {
                match layout {
                    QueryLayout::SameAsData => Ok(self.query_computer::<AsData<$N>, _>(allocator)?)
                    ,
                    QueryLayout::FourBitTransposed => Err(UnsupportedQueryLayout::new(
                        layout,
                        concat!($N, "-bit compression"),
                    ).into()),
                    QueryLayout::ScalarQuantized => {
                        Ok(self.query_computer::<AsQuery<$N, bits::Dense>, _>(allocator)?)
                    },
                    QueryLayout::FullPrecision => Ok(self.query_computer::<AsFull, _>(allocator)?),

                }
            }

            fn query_buffer_description(
                &self,
                layout: QueryLayout
            ) -> Result<QueryBufferDescription, UnsupportedQueryLayout>
            {
                let dim = <Self as Quantizer<B>>::dim(self);
                match layout {
                    QueryLayout::SameAsData => Ok(QueryBufferDescription::new(
                        DataRef::<$N>::canonical_bytes(dim),
                        PowerOfTwo::alignment_of::<u8>(),
                    )),
                    QueryLayout::FourBitTransposed => Err(UnsupportedQueryLayout {
                        layout,
                        desc: concat!($N, "-bit compression"),
                    }),
                    QueryLayout::ScalarQuantized => Ok(QueryBufferDescription::new(
                        QueryRef::<$N, bits::Dense>::canonical_bytes(dim),
                        PowerOfTwo::alignment_of::<u8>(),
                    )),
                    QueryLayout::FullPrecision => Ok(QueryBufferDescription::new(
                        FullQueryRef::canonical_bytes(dim),
                        FullQueryRef::canonical_align(),
                    )),
                }
            }

            fn compress_query(
                &self,
                x: &[f32],
                layout: QueryLayout,
                allow_rescale: bool,
                mut buffer: OpaqueMut<'_>,
                scratch: ScopedAllocator<'_>,
            ) -> Result<(), QueryCompressionError> {
                let dim = <Self as Quantizer<B>>::dim(self);
                let mut finish = |v: &[f32]| -> Result<(), QueryCompressionError> {
                    match layout {
                        QueryLayout::SameAsData => self.compress_query(
                            v,
                            DataMut::<$N>::from_canonical_back_mut(
                                &mut buffer,
                                dim,
                            ).map_err(NotCanonical::new)?,
                            scratch,
                        ),
                        QueryLayout::FourBitTransposed => {
                            Err(UnsupportedQueryLayout::new(
                                layout,
                                concat!($N, "-bit compression"),
                            ).into())
                        },
                        QueryLayout::ScalarQuantized => self.compress_query(
                            v,
                            QueryMut::<$N, bits::Dense>::from_canonical_back_mut(
                                &mut buffer,
                                dim,
                            ).map_err(NotCanonical::new)?,
                            scratch,
                        ),
                        QueryLayout::FullPrecision => self.compress_query(
                            v,
                            FullQueryMut::from_canonical_mut(
                                &mut buffer,
                                dim,
                            ).map_err(NotCanonical::new)?,
                            scratch,
                        ),
                    }
                };

                if allow_rescale && self.quantizer.metric() == SupportedMetric::InnerProduct {
                    let mut copy = x.to_owned();
                    self.quantizer.rescale(&mut copy);
                    finish(&copy)
                } else {
                    finish(x)
                }
            }

            fn fused_query_computer(
                &self,
                x: &[f32],
                layout: QueryLayout,
                allow_rescale: bool,
                allocator: B,
                scratch: ScopedAllocator<'_>,
            ) -> Result<QueryComputer<B>, QueryComputerError>
            {
                let dim = <Self as Quantizer<B>>::dim(self);
                let finish = |v: &[f32]| -> Result<QueryComputer<B>, QueryComputerError> {
                    match layout {
                        QueryLayout::SameAsData => {
                            self.fused_query_computer::<AsData<$N>, Data<$N, _>, B>(
                                v,
                                Data::new_in(dim, allocator.clone())?,
                                allocator,
                                scratch,
                            )
                        },
                        QueryLayout::FourBitTransposed => {
                            Err(UnsupportedQueryLayout::new(
                                layout,
                                concat!($N, "-bit compression"),
                            ).into())
                        },
                        QueryLayout::ScalarQuantized => {
                            self.fused_query_computer::<AsQuery<$N, bits::Dense>, Query<$N, bits::Dense, _>, B>(
                                v,
                                Query::new_in(dim, allocator.clone())?,
                                allocator,
                                scratch,
                            )
                        },
                        QueryLayout::FullPrecision => {
                            self.fused_query_computer::<AsFull, FullQuery<_>, B>(
                                v,
                                FullQuery::empty(dim, allocator.clone())?,
                                allocator,
                                scratch,
                            )
                        },
                    }
                };

                let metric = <Self as Quantizer<B>>::metric(self);
                if allow_rescale && metric == SupportedMetric::InnerProduct {
                    let mut copy = x.to_owned();
                    self.quantizer.rescale(&mut copy);
                    finish(&copy)
                } else {
                    finish(x)
                }
            }

            fn is_supported(&self, layout: QueryLayout) -> bool {
                Self::supports(layout)
            }

            fn compress(
                &self,
                x: &[f32],
                mut into: OpaqueMut<'_>,
                scratch: ScopedAllocator<'_>,
            ) -> Result<(), CompressionError> {
                let dim = <Self as Quantizer<B>>::dim(self);
                let into = DataMut::<$N>::from_canonical_back_mut(into.inspect(), dim)
                    .map_err(CompressionError::not_canonical)?;

                self.quantizer.compress_into_with(x, into, scratch)
                    .map_err(CompressionError::CompressionError)
            }

            fn metric(&self) -> SupportedMetric {
                self.quantizer.metric()
            }

            fn try_clone_into(&self, allocator: B) -> Result<Poly<dyn Quantizer<B>, B>, AllocatorError> {
                let clone = (&*self).try_clone()?;
                poly!({ Quantizer<B> }, clone, allocator)
            }

            #[cfg(feature = "flatbuffers")]
            fn serialize(&self, allocator: B) -> Result<Poly<[u8], B>, AllocatorError> {
                Impl::<$N, A>::serialize(self, allocator)
            }
        }
    };
    ($N:literal, $($Ns:literal),*) => {
        plan!($N);
        $(plan!($Ns);)*
    }
}

plan!(2, 4, 8);

////////////////
// Flatbuffer //
////////////////

#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum DeserializationError {
    #[error("unhandled file identifier in flatbuffer")]
    InvalidIdentifier,

    #[error("unsupported number of bits ({0})")]
    UnsupportedBitWidth(u32),

    #[error(transparent)]
    InvalidQuantizer(#[from] super::quantizer::DeserializationError),

    #[error(transparent)]
    InvalidFlatBuffer(#[from] flatbuffers::InvalidFlatbuffer),

    #[error(transparent)]
    AllocatorError(#[from] AllocatorError),
}

/// Attempt to deserialize a `spherical::Quantizer` flatbuffer into one of the concrete
/// implementations of `Quantizer`.
///
/// This function guarantees that the returned `Poly` is the first object allocated through
/// `alloc`.
#[cfg(feature = "flatbuffers")]
#[cfg_attr(docsrs, doc(cfg(feature = "flatbuffers")))]
pub fn try_deserialize<O, A>(
    data: &[u8],
    alloc: A,
) -> Result<Poly<dyn Quantizer<O>, A>, DeserializationError>
where
    O: Allocator + std::panic::UnwindSafe + Send + Sync + 'static,
    A: Allocator + std::panic::RefUnwindSafe + Send + Sync + 'static,
{
    // An inner impl is used to ensure that the returned `Poly` is allocated before any of
    // the allocations needed by the members.
    //
    // This ensures that if a bump allocator is used, the root object appears first.
    fn unpack_bits<'a, const NBITS: usize, O, A>(
        proto: fb::spherical::SphericalQuantizer<'_>,
        alloc: A,
    ) -> Result<Poly<dyn Quantizer<O> + 'a, A>, DeserializationError>
    where
        O: Allocator + Send + Sync + std::panic::UnwindSafe + 'static,
        A: Allocator + Send + Sync + 'a,
        Impl<NBITS, A>: Quantizer<O> + Constructible<A>,
    {
        let imp = match Poly::new_with(
            #[inline(never)]
            |alloc| -> Result<_, super::quantizer::DeserializationError> {
                let quantizer = SphericalQuantizer::try_unpack(alloc, proto)?;
                Ok(Impl::new(quantizer)?)
            },
            alloc,
        ) {
            Ok(imp) => imp,
            Err(CompoundError::Allocator(err)) => {
                return Err(err.into());
            }
            Err(CompoundError::Constructor(err)) => {
                return Err(err.into());
            }
        };
        Ok(poly!({ Quantizer<O> }, imp))
    }

    // Check that this is one of the known identifiers.
    if !fb::spherical::quantizer_buffer_has_identifier(data) {
        return Err(DeserializationError::InvalidIdentifier);
    }

    // Match as much as we can without allocating.
    //
    // Then, we branch on the number of bits.
    let root = fb::spherical::root_as_quantizer(data)?;
    let nbits = root.nbits();
    let proto = root.quantizer();

    match nbits {
        1 => unpack_bits::<1, _, _>(proto, alloc),
        2 => unpack_bits::<2, _, _>(proto, alloc),
        4 => unpack_bits::<4, _, _>(proto, alloc),
        8 => unpack_bits::<8, _, _>(proto, alloc),
        n => Err(DeserializationError::UnsupportedBitWidth(n)),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::views::{Matrix, MatrixView};
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        algorithms::{transforms::TargetDim, TransformKind},
        alloc::{AlignedAllocator, GlobalAllocator, Poly},
        num::PowerOfTwo,
        spherical::PreScale,
    };

    ////////////////////
    // Test Quantizer //
    ////////////////////

    fn test_plan_1_bit(plan: &dyn Quantizer) {
        assert_eq!(
            plan.nbits(),
            1,
            "this test only applies to 1-bit quantization"
        );

        // Check Layouts.
        for layout in QueryLayout::all() {
            match layout {
                QueryLayout::SameAsData
                | QueryLayout::FourBitTransposed
                | QueryLayout::FullPrecision => assert!(
                    plan.is_supported(layout),
                    "expected {} to be supported",
                    layout
                ),
                QueryLayout::ScalarQuantized => assert!(
                    !plan.is_supported(layout),
                    "expected {} to not be supported",
                    layout
                ),
            }
        }
    }

    fn test_plan_n_bit(plan: &dyn Quantizer, nbits: usize) {
        assert_ne!(nbits, 1, "there is another test for 1-bit quantizers");
        assert_eq!(
            plan.nbits(),
            nbits,
            "this test only applies to 1-bit quantization"
        );

        // Check Layouts.
        for layout in QueryLayout::all() {
            match layout {
                QueryLayout::SameAsData
                | QueryLayout::ScalarQuantized
                | QueryLayout::FullPrecision => assert!(
                    plan.is_supported(layout),
                    "expected {} to be supported",
                    layout
                ),
                QueryLayout::FourBitTransposed => assert!(
                    !plan.is_supported(layout),
                    "expected {} to not be supported",
                    layout
                ),
            }
        }
    }

    #[inline(never)]
    fn test_plan(plan: &dyn Quantizer, nbits: usize, dataset: MatrixView<f32>) {
        // Perform the bit-specific test.
        if nbits == 1 {
            test_plan_1_bit(plan);
        } else {
            test_plan_n_bit(plan, nbits);
        }

        // Run bit-width agnostic tests.
        assert_eq!(plan.full_dim(), dataset.ncols());

        // Use the correct alignment for the base pointers.
        let alloc = AlignedAllocator::new(PowerOfTwo::new(4).unwrap());
        let mut a = Poly::broadcast(u8::default(), plan.bytes(), alloc).unwrap();
        let mut b = Poly::broadcast(u8::default(), plan.bytes(), alloc).unwrap();
        let scoped_global = ScopedAllocator::global();

        plan.compress(dataset.row(0), OpaqueMut::new(&mut a), scoped_global)
            .unwrap();
        plan.compress(dataset.row(1), OpaqueMut::new(&mut b), scoped_global)
            .unwrap();

        let f = plan.distance_computer(GlobalAllocator).unwrap();
        let _: f32 = f
            .evaluate_similarity(Opaque::new(&a), Opaque::new(&b))
            .unwrap();

        let test_errors = |f: &dyn DynDistanceComputer| {
            // `a` too short
            let err = f
                .evaluate(Opaque::new(&a[..a.len() - 1]), Opaque::new(&b))
                .unwrap_err();
            assert!(matches!(err, DistanceError::QueryReify(_)));

            // `a` too long
            let err = f
                .evaluate(Opaque::new(&vec![0u8; a.len() + 1]), Opaque::new(&b))
                .unwrap_err();
            assert!(matches!(err, DistanceError::QueryReify(_)));

            // `b` too short
            let err = f
                .evaluate(Opaque::new(&a), Opaque::new(&b[..b.len() - 1]))
                .unwrap_err();
            assert!(matches!(err, DistanceError::XReify(_)));

            // `a` too long
            let err = f
                .evaluate(Opaque::new(&a), Opaque::new(&vec![0u8; b.len() + 1]))
                .unwrap_err();
            assert!(matches!(err, DistanceError::XReify(_)));
        };

        test_errors(&*f.inner);

        let f = plan.distance_computer_ref();
        let _: f32 = f.evaluate(Opaque::new(&a), Opaque::new(&b)).unwrap();
        test_errors(f);

        // Test all supported flavors of `QueryComputer`.
        for layout in QueryLayout::all() {
            if !plan.is_supported(layout) {
                let check_message = |msg: &str| {
                    assert!(
                        msg.contains(&(layout.to_string())),
                        "error message ({}) should contain the layout \"{}\"",
                        msg,
                        layout
                    );
                    assert!(
                        msg.contains(&format!("{}", nbits)),
                        "error message ({}) should contain the number of bits \"{}\"",
                        msg,
                        nbits
                    );
                };

                // Error for query computer
                {
                    let err = plan
                        .fused_query_computer(
                            dataset.row(1),
                            layout,
                            false,
                            GlobalAllocator,
                            scoped_global,
                        )
                        .unwrap_err();

                    let msg = err.to_string();
                    check_message(&msg);
                }

                // Query buffer
                {
                    let err = plan.query_buffer_description(layout).unwrap_err();
                    let msg = err.to_string();
                    check_message(&msg);
                }

                // Compresss Query Into
                {
                    let buffer = &mut [];
                    let err = plan
                        .compress_query(
                            dataset.row(1),
                            layout,
                            true,
                            OpaqueMut::new(buffer),
                            scoped_global,
                        )
                        .unwrap_err();
                    let msg = err.to_string();
                    check_message(&msg);
                }

                // Standalone Query Computer
                {
                    let err = plan.query_computer(layout, GlobalAllocator).unwrap_err();
                    let msg = err.to_string();
                    check_message(&msg);
                }

                continue;
            }

            let g = plan
                .fused_query_computer(
                    dataset.row(1),
                    layout,
                    false,
                    GlobalAllocator,
                    scoped_global,
                )
                .unwrap();
            assert_eq!(
                g.layout(),
                layout,
                "the query computer should faithfully preserve the requested layout"
            );

            let direct: f32 = g.evaluate_similarity(Opaque(&a)).unwrap();

            // Check that the fused computer correctly returns errors for invalid inputs.
            {
                let err = g
                    .evaluate_similarity(Opaque::new(&a[..a.len() - 1]))
                    .unwrap_err();
                assert!(matches!(err, QueryDistanceError::XReify(_)));

                let err = g
                    .evaluate_similarity(Opaque::new(&vec![0u8; a.len() + 1]))
                    .unwrap_err();
                assert!(matches!(err, QueryDistanceError::XReify(_)));
            }

            let sizes = plan.query_buffer_description(layout).unwrap();
            let mut buf =
                Poly::broadcast(0u8, sizes.bytes(), AlignedAllocator::new(sizes.align())).unwrap();

            plan.compress_query(
                dataset.row(1),
                layout,
                false,
                OpaqueMut::new(&mut buf),
                scoped_global,
            )
            .unwrap();

            let standalone = plan.query_computer(layout, GlobalAllocator).unwrap();

            assert_eq!(
                standalone.layout(),
                layout,
                "the standalone computer did not preserve the requested layout",
            );

            let indirect: f32 = standalone
                .evaluate_similarity(Opaque(&buf), Opaque(&a))
                .unwrap();

            assert_eq!(
                direct, indirect,
                "the two different query computation APIs did not return the same result"
            );

            // Errors
            let too_small = &dataset.row(0)[..dataset.ncols() - 1];
            assert!(plan
                .fused_query_computer(too_small, layout, false, GlobalAllocator, scoped_global)
                .is_err());
        }

        // Errors
        {
            let mut too_small = vec![u8::default(); plan.bytes() - 1];
            assert!(plan
                .compress(dataset.row(0), OpaqueMut(&mut too_small), scoped_global)
                .is_err());

            let mut too_big = vec![u8::default(); plan.bytes() + 1];
            assert!(plan
                .compress(dataset.row(0), OpaqueMut(&mut too_big), scoped_global)
                .is_err());

            let mut just_right = vec![u8::default(); plan.bytes()];
            assert!(plan
                .compress(
                    &dataset.row(0)[..dataset.ncols() - 1],
                    OpaqueMut(&mut just_right),
                    scoped_global
                )
                .is_err());
        }
    }

    fn make_impl<const NBITS: usize>(metric: SupportedMetric) -> (Impl<NBITS>, Matrix<f32>)
    where
        Impl<NBITS>: Constructible,
    {
        let data = test_dataset();
        let mut rng = StdRng::seed_from_u64(0x7d535118722ff197);

        let quantizer = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::PaddingHadamard {
                target_dim: TargetDim::Natural,
            },
            metric,
            PreScale::None,
            &mut rng,
            GlobalAllocator,
        )
        .unwrap();

        (Impl::<NBITS>::new(quantizer).unwrap(), data)
    }

    #[test]
    fn test_plan_1bit_l2() {
        let (plan, data) = make_impl::<1>(SupportedMetric::SquaredL2);
        test_plan(&plan, 1, data.as_view());
    }

    #[test]
    fn test_plan_1bit_ip() {
        let (plan, data) = make_impl::<1>(SupportedMetric::InnerProduct);
        test_plan(&plan, 1, data.as_view());
    }

    #[test]
    fn test_plan_1bit_cosine() {
        let (plan, data) = make_impl::<1>(SupportedMetric::Cosine);
        test_plan(&plan, 1, data.as_view());
    }

    #[test]
    fn test_plan_2bit_l2() {
        let (plan, data) = make_impl::<2>(SupportedMetric::SquaredL2);
        test_plan(&plan, 2, data.as_view());
    }

    #[test]
    fn test_plan_2bit_ip() {
        let (plan, data) = make_impl::<2>(SupportedMetric::InnerProduct);
        test_plan(&plan, 2, data.as_view());
    }

    #[test]
    fn test_plan_2bit_cosine() {
        let (plan, data) = make_impl::<2>(SupportedMetric::Cosine);
        test_plan(&plan, 2, data.as_view());
    }

    #[test]
    fn test_plan_4bit_l2() {
        let (plan, data) = make_impl::<4>(SupportedMetric::SquaredL2);
        test_plan(&plan, 4, data.as_view());
    }

    #[test]
    fn test_plan_4bit_ip() {
        let (plan, data) = make_impl::<4>(SupportedMetric::InnerProduct);
        test_plan(&plan, 4, data.as_view());
    }

    #[test]
    fn test_plan_4bit_cosine() {
        let (plan, data) = make_impl::<4>(SupportedMetric::Cosine);
        test_plan(&plan, 4, data.as_view());
    }

    #[test]
    fn test_plan_8bit_l2() {
        let (plan, data) = make_impl::<8>(SupportedMetric::SquaredL2);
        test_plan(&plan, 8, data.as_view());
    }

    #[test]
    fn test_plan_8bit_ip() {
        let (plan, data) = make_impl::<8>(SupportedMetric::InnerProduct);
        test_plan(&plan, 8, data.as_view());
    }

    #[test]
    fn test_plan_8bit_cosine() {
        let (plan, data) = make_impl::<8>(SupportedMetric::Cosine);
        test_plan(&plan, 8, data.as_view());
    }

    fn test_dataset() -> Matrix<f32> {
        let data = vec![
            0.28657,
            -0.0318168,
            0.0666847,
            0.0329265,
            -0.00829283,
            0.168735,
            -0.000846311,
            -0.360779, // row 0
            -0.0968938,
            0.161921,
            -0.0979579,
            0.102228,
            -0.259928,
            -0.139634,
            0.165384,
            -0.293443, // row 1
            0.130205,
            0.265737,
            0.401816,
            -0.407552,
            0.13012,
            -0.0475244,
            0.511723,
            -0.4372, // row 2
            -0.0979126,
            0.135861,
            -0.0154144,
            -0.14047,
            -0.0250029,
            -0.190279,
            0.407283,
            -0.389184, // row 3
            -0.264153,
            0.0696822,
            -0.145585,
            0.370284,
            0.186825,
            -0.140736,
            0.274703,
            -0.334563, // row 4
            0.247613,
            0.513165,
            -0.0845867,
            0.0532264,
            -0.00480601,
            -0.122408,
            0.47227,
            -0.268301, // row 5
            0.103198,
            0.30756,
            -0.316293,
            -0.0686877,
            -0.330729,
            -0.461997,
            0.550857,
            -0.240851, // row 6
            0.128258,
            0.786291,
            -0.0268103,
            0.111763,
            -0.308962,
            -0.17407,
            0.437154,
            -0.159879, // row 7
            0.00374063,
            0.490301,
            0.0327826,
            -0.0340962,
            -0.118605,
            0.163879,
            0.2737,
            -0.299942, // row 8
            -0.284077,
            0.249377,
            -0.0307734,
            -0.0661631,
            0.233854,
            0.427987,
            0.614132,
            -0.288649, // row 9
            -0.109492,
            0.203939,
            -0.73956,
            -0.130748,
            0.22072,
            0.0647836,
            0.328726,
            -0.374602, // row 10
            -0.223114,
            0.0243489,
            0.109195,
            -0.416914,
            0.0201052,
            -0.0190542,
            0.947078,
            -0.333229, // row 11
            -0.165869,
            -0.00296729,
            -0.414378,
            0.231321,
            0.205365,
            0.161761,
            0.148608,
            -0.395063, // row 12
            -0.0498255,
            0.193279,
            -0.110946,
            -0.181174,
            -0.274578,
            -0.227511,
            0.190208,
            -0.256174, // row 13
            -0.188106,
            -0.0292958,
            0.0930939,
            0.0558456,
            0.257437,
            0.685481,
            0.307922,
            -0.320006, // row 14
            0.250035,
            0.275942,
            -0.0856306,
            -0.352027,
            -0.103509,
            -0.00890859,
            0.276121,
            -0.324718, // row 15
        ];

        Matrix::try_from(data.into(), 16, 8).unwrap()
    }

    #[cfg(feature = "flatbuffers")]
    mod serialization {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        use super::*;
        use crate::alloc::{BumpAllocator, GlobalAllocator};

        #[inline(never)]
        fn test_plan_serialization(
            quantizer: &dyn Quantizer,
            nbits: usize,
            dataset: MatrixView<f32>,
        ) {
            // Run bit-width agnostic tests.
            assert_eq!(quantizer.full_dim(), dataset.ncols());
            let scoped_global = ScopedAllocator::global();

            let serialized = quantizer.serialize(GlobalAllocator).unwrap();
            let deserialized =
                try_deserialize::<GlobalAllocator, _>(&serialized, GlobalAllocator).unwrap();

            assert_eq!(deserialized.nbits(), nbits);
            assert_eq!(deserialized.bytes(), quantizer.bytes());
            assert_eq!(deserialized.dim(), quantizer.dim());
            assert_eq!(deserialized.full_dim(), quantizer.full_dim());
            assert_eq!(deserialized.metric(), quantizer.metric());

            for layout in QueryLayout::all() {
                assert_eq!(
                    deserialized.is_supported(layout),
                    quantizer.is_supported(layout)
                );
            }

            // Use the correct alignment for the base pointers.
            let alloc = AlignedAllocator::new(PowerOfTwo::new(4).unwrap());
            {
                let mut a = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();
                let mut b = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();

                for row in dataset.row_iter() {
                    quantizer
                        .compress(row, OpaqueMut::new(&mut a), scoped_global)
                        .unwrap();
                    deserialized
                        .compress(row, OpaqueMut::new(&mut b), scoped_global)
                        .unwrap();

                    // Compressed representation should be identical.
                    assert_eq!(a, b);
                }
            }

            // Distance Computer
            {
                let mut a0 = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();
                let mut a1 = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();
                let mut b0 = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();
                let mut b1 = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();

                let q_computer = quantizer.distance_computer(GlobalAllocator).unwrap();
                let q_computer_ref = quantizer.distance_computer_ref();
                let d_computer = deserialized.distance_computer(GlobalAllocator).unwrap();
                let d_computer_ref = deserialized.distance_computer_ref();

                for r0 in dataset.row_iter() {
                    quantizer
                        .compress(r0, OpaqueMut::new(&mut a0), scoped_global)
                        .unwrap();
                    deserialized
                        .compress(r0, OpaqueMut::new(&mut b0), scoped_global)
                        .unwrap();
                    for r1 in dataset.row_iter() {
                        quantizer
                            .compress(r1, OpaqueMut::new(&mut a1), scoped_global)
                            .unwrap();
                        deserialized
                            .compress(r1, OpaqueMut::new(&mut b1), scoped_global)
                            .unwrap();

                        let a0 = Opaque::new(&a0);
                        let a1 = Opaque::new(&a1);

                        let q_computer_dist = q_computer.evaluate_similarity(a0, a1).unwrap();
                        let d_computer_dist = d_computer.evaluate_similarity(a0, a1).unwrap();

                        assert_eq!(q_computer_dist, d_computer_dist);

                        let q_computer_ref_dist = q_computer_ref.evaluate(a0, a1).unwrap();

                        assert_eq!(q_computer_dist, q_computer_ref_dist);

                        let d_computer_ref_dist = d_computer_ref.evaluate(a0, a1).unwrap();
                        assert_eq!(d_computer_dist, d_computer_ref_dist);
                    }
                }
            }

            // Query Computer
            {
                let mut a = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();
                let mut b = Poly::broadcast(u8::default(), quantizer.bytes(), alloc).unwrap();

                for layout in QueryLayout::all() {
                    if !quantizer.is_supported(layout) {
                        continue;
                    }

                    for r in dataset.row_iter() {
                        let q_computer = quantizer
                            .fused_query_computer(r, layout, false, GlobalAllocator, scoped_global)
                            .unwrap();
                        let d_computer = deserialized
                            .fused_query_computer(r, layout, false, GlobalAllocator, scoped_global)
                            .unwrap();

                        for u in dataset.row_iter() {
                            quantizer
                                .compress(u, OpaqueMut::new(&mut a), scoped_global)
                                .unwrap();
                            deserialized
                                .compress(u, OpaqueMut::new(&mut b), scoped_global)
                                .unwrap();

                            assert_eq!(
                                q_computer.evaluate_similarity(Opaque::new(&a)).unwrap(),
                                d_computer.evaluate_similarity(Opaque::new(&b)).unwrap(),
                            );
                        }
                    }
                }
            }
        }

        // An allocator that succeeds on its first allocation but fails on its second.
        #[derive(Debug, Clone)]
        struct FlakyAllocator {
            have_allocated: Arc<AtomicBool>,
        }

        impl FlakyAllocator {
            fn new(have_allocated: Arc<AtomicBool>) -> Self {
                Self { have_allocated }
            }
        }

        // SAFETY: This is a wrapper around GlobalAllocator that only succeeds once.
        unsafe impl AllocatorCore for FlakyAllocator {
            fn allocate(
                &self,
                layout: std::alloc::Layout,
            ) -> Result<std::ptr::NonNull<[u8]>, AllocatorError> {
                if self.have_allocated.swap(true, Ordering::Relaxed) {
                    Err(AllocatorError)
                } else {
                    GlobalAllocator.allocate(layout)
                }
            }

            unsafe fn deallocate(&self, ptr: std::ptr::NonNull<[u8]>, layout: std::alloc::Layout) {
                // SAFETY: Inherited from caller.
                unsafe { GlobalAllocator.deallocate(ptr, layout) }
            }
        }

        fn test_plan_panic_boundary<const NBITS: usize>(v: &Impl<NBITS>)
        where
            Impl<NBITS>: Quantizer,
        {
            // Ensure that we do not panic if reallocation returns an error.
            let have_allocated = Arc::new(AtomicBool::new(false));
            let _: AllocatorError = v
                .serialize(FlakyAllocator::new(have_allocated.clone()))
                .unwrap_err();
            assert!(have_allocated.load(Ordering::Relaxed));
        }

        #[test]
        fn test_plan_1bit_l2() {
            let (plan, data) = make_impl::<1>(SupportedMetric::SquaredL2);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 1, data.as_view());
        }

        #[test]
        fn test_plan_1bit_ip() {
            let (plan, data) = make_impl::<1>(SupportedMetric::InnerProduct);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 1, data.as_view());
        }

        #[test]
        fn test_plan_2bit_l2() {
            let (plan, data) = make_impl::<2>(SupportedMetric::SquaredL2);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 2, data.as_view());
        }

        #[test]
        fn test_plan_2bit_ip() {
            let (plan, data) = make_impl::<2>(SupportedMetric::InnerProduct);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 2, data.as_view());
        }

        #[test]
        fn test_plan_4bit_l2() {
            let (plan, data) = make_impl::<4>(SupportedMetric::SquaredL2);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 4, data.as_view());
        }

        #[test]
        fn test_plan_4bit_ip() {
            let (plan, data) = make_impl::<4>(SupportedMetric::InnerProduct);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 4, data.as_view());
        }

        #[test]
        fn test_plan_8bit_l2() {
            let (plan, data) = make_impl::<8>(SupportedMetric::SquaredL2);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 8, data.as_view());
        }

        #[test]
        fn test_plan_8bit_ip() {
            let (plan, data) = make_impl::<8>(SupportedMetric::InnerProduct);
            test_plan_panic_boundary(&plan);
            test_plan_serialization(&plan, 8, data.as_view());
        }

        #[test]
        fn test_allocation_order() {
            let (plan, _) = make_impl::<1>(SupportedMetric::SquaredL2);
            let buf = plan.serialize(GlobalAllocator).unwrap();

            let allocator = BumpAllocator::new(8192, PowerOfTwo::new(64).unwrap()).unwrap();
            let deserialized =
                try_deserialize::<GlobalAllocator, _>(&buf, allocator.clone()).unwrap();
            assert_eq!(
                Poly::as_ptr(&deserialized).cast::<u8>(),
                allocator.as_ptr(),
                "expected the returned box to be allocated first",
            );
        }
    }
}
