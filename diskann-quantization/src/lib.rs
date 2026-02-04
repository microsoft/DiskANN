/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![cfg_attr(docsrs, feature(doc_cfg))]

//! A collection of utilities and algorithms for training quantizers, compressing data,
//! and performing distance computations on compressed data.
//!
//! # Quantizers
//!
//! The currently implemented quantizers are listed below. Refer to the documentation
//! within each module for more information.
//!
//! Note that the capabilities of each quantizer are varied and some aspects are still in
//! progress. If you have any questions or need something implemented, please reach out!
//!
//! * [Scalar](crate::scalar): Compressing dimensions to 8-bits and lower.
//! * [Binary](crate::binary): Light weight 1-bit compression.
//! * [Product](crate::product): Compressing multiple dimensions into a single code.
//!   **NOTE**: Support for product quantization is currently limited as the implementation
//!   is spread between this crate and the original implementation in DiskANN.
//! * [Spherical](crate::spherical): A compression technique similar to scalar quantization,
//!   put with the characteristic that vectors are normalized, transformed, and then
//!   projected onto a unit sphere.
//!
//!   This class of algorithms is based off the [RabitQ](https://arxiv.org/abs/2405.12497)
//!   paper and associated work.
//! * [MinMax](crate::minmax): A compression similar to scalar quantization, but where the
//!   the coefficients to quantize are computed on a per-vector basis; allowing this to be
//!   used in a streaming setting (i.e. without any need for training).
//!
//!
//! # [BitSlice](crate::bits::BitSliceBase)
//!
//! A primary abstraction introduced is a BitSlice. Think of it as a Rust slice, but for
//! packing together bits. It currently supports unsigned integers of widths 1 to 8.
//! The borrowed version of the slice consists of a pointer (with a lifetime) and a length
//! and is therefore just 16-bytes in size and amenable to the niche optimization.
//! It also comes in a boxed variant for convenience.
//!
//! Some trickery is involved in the constructor to ensure that the pointer can only ever be
//! obtained in a safe way.
//!
//! ## Distances over BitSlices
//!
//! Distances over bit-slices are a key-component for building efficient distances for
//! [binary](crate::binary::BinaryQuantizer) and [scalar](crate::scalar::ScalarQuantizer)
//! quantized vectors.
//!
//! Performance characteristics of the implemented functions are listed here:
//!
//! * [`crate::distances::SquaredL2`]/[`crate::distances::InnerProduct`]: Fast SIMD
//!   implementations available for 4 and 8-bit vectors. Fallback implementations are used
//!   for 2, 3, 5, 6, and 7-bit vectors. Scalar pop-count used for 1-bit distances.
//!
//! * [`crate::distances::Hamming`]: Scalar pop-count used.
//!
//! # Adding New Quantizers
//!
//! Adding new quantizers is a fairly straight-forward.
//!
//! 1. Implement training using a similar style to existing quantizers by:
//!
//!   A. Define a quantizer that will hold the final any central quantization state.
//!
//!   B. Implement a training module with training parameters and an associated `train`
//!     method (available either as a trait or inherhent method, whichever makes more
//!     sense).
//!
//! 2. Determine the representation for compressed vectors. This can be done using
//!    combinations of [`crate::bits::BitSlice`]s, utilities in the `ownership` module and
//!    more.
//!
//!    Ensure that this representation implements [`diskann_utils::Reborrow`] and
//!    [`diskann_utils::ReborrowMut`] so that working with generalized references types is sane.
//!
//! 3. Implement distance functors (e.g. [`crate::scalar::CompensatedIP`]) to enable distance
//!    computations on compressed vectors and between compress and full-precision vectors.
//!
//!    Additionally, implement [`crate::AsFunctor`] for the central quantizer for each
//!    distance functor.
//!
//! # Design Considerations
//!
//! In this section, we will cover some of the design philosophy regarding quantizers,
//! particularly how this interacts with vector search and data providers.
//!
//! ## No Inherent Methods for Distances in Quantizers
//!
//! Inherent methods for distances on quantizer structs such as
//!
//! * `fn distance_l2(&self, full_precision: &[f32], quantized: &[u8]) -> f32`
//! * `fn distance_qq_l2(&self, quantized_0: &[u8], quantized_1: &[u8]) -> f32`
//!
//! do not work well for a number of reasons:
//!
//! 1. Inherent methods are unavailable to call via trait restrictions on generic arguments.
//!
//! 2. Fixed argument/return types limit flexibility. One example is marking the difference
//!    between [`diskann_vector::MathematicalValue`] and [`diskann_vector::SimilarityScore`].
//!
//!    The solution to accepting multiple argument/return types is exactly the approach
//!    taken in this crate, i.e. using auxiliary [`diskann_vector::DistanceFunction`]
//!    implementations on light-weight types (e.g. [`crate::scalar::CompensatedIP`]).
//!
//! 3. This approach does not work when hefty query preprocessing can make distance
//!    computations faster, such as with table-based PQ implementations.
//!
//!    Instead, creating a [`diskann_vector::PreprocessedDistanceFunction`] can be used.
//!
//! 4. Does not really extend well to performing batches of distance computations when that
//!    can be computed more efficiently.
//!
//! ## No Common Trait for Quantizer-based Distances
//!
//! Creating a trait for quantizer-based distances would solve point number 1 above about
//! usability in generic contexts, but the other points still remain. I argue that a
//! functor-based interface is generally more flexible and useful.
//!
//! Additionally, not all quantizers implement the same distance functions. For example,
//! binary quantization only uses [`crate::distances::Hamming`].
//!
//! ## No Common Trait for Training
//!
//! The requirements for training between [scalar quantization](crate::scalar) and
//! [produce quantization](crate::product) are sufficiently different that using a common
//! training interface feels overly restrictive.
//!
//! ## Thoughts on Canonical Layouts for Compressed Vector Representations
//!
//! As much as possible, we try to avoid using raw `&[u8]` as the representation for
//! compressed vectors in favor of richer types such as [`crate::bits::BitSlice`] and
//! [`crate::scalar::CompensatedVectorRef`] (though product-quantization still uses this
//! for historical reasons). This is because compressed representations may need to contain
//! more than just a raw encoding (e.g. the compensation coefficient for scalar quantization)
//! and using a richer type makes this less error prone.
//!
//! But this opens a can of worms regarding storage within the data provider. Lets take, for
//! example, [`crate::scalar::CompensatedVector`], which consists of a slice for scalar
//! quantization codes and a compensation coefficient. The provider can choose to store
//! these two components in separate data structures or fuse them together into a single
//! block of memory (note that [`crate::scalar::MutCompensatedVectorRef`] is defined so that
//! both the codes and compensation coefficient can be updated in-place. The provider may
//! **also** wish to perform cache-line padding.
//!
//! Because of the myriad of ways data can be stored and retrieved, the utilities in this
//! crate instead focus on providing tools to express the compressed state that allow for
//! this design space exploration on the data provider level.
//!
//! Eventually, however, we may wish to impose canonical layouts for these representations
//! within the `quantization` crate if that turns out to be necessary.

mod ownership;
mod utils;

// misc
pub mod alloc;
pub mod bits;
pub mod cancel;
pub mod distances;
pub mod error;
pub mod meta;
pub mod num;
pub mod random;
mod traits;
pub mod views;

pub use traits::{AsFunctor, CompressInto, CompressIntoWith};

// serialization
#[cfg(feature = "flatbuffers")]
#[allow(mismatched_lifetime_syntaxes)] // The generated code isn't clippy-clean.
pub(crate) mod flatbuffers;

// common algorithms
pub mod algorithms;

// quantization
pub mod binary;
pub mod minmax;
pub mod multi_vector;
pub mod product;
pub mod scalar;
pub mod spherical;

/// Selector for the parallelization strategy used by some algorithms.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Parallelism {
    /// Use single-threaded execution.
    #[default]
    Sequential,

    /// Use Rayon based parallelism in the dynamically scoped Rayon thread pool.
    #[cfg(feature = "rayon")]
    #[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
    Rayon,
}

// The code-generation module needs to be public to prevent symbols from getting culled.
#[doc(hidden)]
#[cfg(feature = "codegen")]
pub mod __codegen;

// Miri can't handle `trybuild`.
#[cfg(all(test, not(miri)))]
mod tests {
    #[test]
    fn compile_tests() {
        let t = trybuild::TestCases::new();
        // Begin with a `pass` test to force full compilation of all the test binaries.
        //
        // This ensures that post-monomorphization errors are tested.
        t.pass("tests/compile-fail/bootstrap/bootstrap.rs");
        t.compile_fail("tests/compile-fail/*.rs");
        t.compile_fail("tests/compile-fail/error/*.rs");
        t.compile_fail("tests/compile-fail/multi/*.rs");
    }
}

#[cfg(test)]
mod test_util;
