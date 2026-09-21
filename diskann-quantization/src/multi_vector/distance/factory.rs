/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Factory + concrete `MaxSimKernel<T>` impls for the multi-vector distance
//! API. BYOTE entry point — see [`build_max_sim`].

use std::num::NonZeroUsize;

use diskann_vector::PureDistanceFunction;
use diskann_vector::distance::InnerProduct;
use diskann_wide::Architecture;
use diskann_wide::arch::Scalar;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::fallback::FallbackKernel;
use super::isa::{MaxSimIsa, NotSupported};
use super::kernel::{Erase, MaxSimKernel};
use super::max_sim::MaxSimError;
use crate::matrix_kernels as mk;
use crate::multi_vector::distance::QueryMatRef;
use crate::multi_vector::{BlockTransposed, Mat, MatRef, Standard};

// ─────────────────────────────────────────────────────────────────────────
//  Prepared<A, Q, NR> — concrete kernel for the arch-dispatched paths.
// ─────────────────────────────────────────────────────────────────────────

// ZST for selecting the packing for the documents.
//
// TODO: Some of these decisions should eventually ve moved lower into `matrix_kernels`.
#[derive(Debug, Clone, Copy)]
struct Pack<const NR: usize>;

#[derive(Debug)]
struct Prepared<A, Q, const NR: usize> {
    arch: A,
    prepared: Q,
    _packing: Pack<NR>,
}

impl<A, const GROUP: usize, const NR: usize> MaxSimKernel<f32>
    for Prepared<A, BlockTransposed<f32, GROUP>, NR>
where
    A: Architecture,
    for<'a> mk::maxsim::packed_f32_x_unpacked_f32::Driver<'a, A, GROUP, NR>: mk::Drive,
{
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn compute_max_sim(
        &self,
        doc: MatRef<'_, Standard<f32>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        if scores.len() != self.nrows() {
            return Err(MaxSimError::InvalidBufferLength(scores.len(), self.nrows()));
        }

        if doc.vector_dim() != self.prepared.ncols() {
            return Err(MaxSimError::UnequalDim(
                doc.vector_dim(),
                self.prepared.ncols(),
            ));
        }

        let Some(k) = NonZeroUsize::new(self.prepared.ncols()).map(mk::DimK::new) else {
            scores.fill(if doc.num_vectors() == 0 {
                f32::MAX
            } else {
                0.0
            });
            return Ok(());
        };

        let Some(a) = mk::blocks::packed::View::from_block_transposed(self.prepared.as_view())
        else {
            return Ok(());
        };

        let Some(b) = mk::blocks::unpacked::View::from_matrix_view(doc.as_matrix_view()) else {
            scores.fill(f32::MAX);
            return Ok(());
        };

        // SAFETY: The dimension check establishes that `a.k() == b.k() == k`.
        // The length check establishes that `scores` occupies exactly the
        // packed blocks in `a`.
        let mut driver = unsafe {
            mk::maxsim::packed_f32_x_unpacked_f32::Driver::new(
                self.arch,
                a,
                b,
                scores,
                k,
                mk::Cache::detect(),
            )
        };

        mk::Drive::drive(&mut driver);

        scores.iter_mut().for_each(|s| *s = -*s);

        Ok(())
    }
}

impl<A, const GROUP: usize, const NR: usize> MaxSimKernel<half::f16>
    for Prepared<A, BlockTransposed<f32, GROUP>, NR>
where
    A: Architecture,
    for<'a> mk::maxsim::packed_f32_x_unpacked_f16::Driver<'a, A, GROUP, NR>: mk::Drive,
{
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn compute_max_sim(
        &self,
        doc: MatRef<'_, Standard<half::f16>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        if scores.len() != self.nrows() {
            return Err(MaxSimError::InvalidBufferLength(scores.len(), self.nrows()));
        }

        if doc.vector_dim() != self.prepared.ncols() {
            return Err(MaxSimError::UnequalDim(
                doc.vector_dim(),
                self.prepared.ncols(),
            ));
        }

        let Some(k) = NonZeroUsize::new(self.prepared.ncols()).map(mk::DimK::new) else {
            scores.fill(if doc.num_vectors() == 0 {
                f32::MAX
            } else {
                0.0
            });
            return Ok(());
        };

        let Some(a) = mk::blocks::packed::View::from_block_transposed(self.prepared.as_view())
        else {
            return Ok(());
        };

        let Some(b) = mk::blocks::unpacked::View::from_matrix_view(doc.as_matrix_view()) else {
            scores.fill(f32::MAX);
            return Ok(());
        };

        // SAFETY: The dimension check establishes that `a.k() == b.k() == k`.
        // The length check establishes that `scores` occupies exactly the
        // packed blocks in `a`.
        let mut driver = unsafe {
            mk::maxsim::packed_f32_x_unpacked_f16::Driver::new(
                self.arch,
                a,
                b,
                scores,
                k,
                mk::Cache::detect(),
            )
        };

        mk::Drive::drive(&mut driver);

        scores.iter_mut().for_each(|s| *s = -*s);

        Ok(())
    }
}

impl<A, const GROUP: usize, const NR: usize, const PACK: usize> MaxSimKernel<i8>
    for Prepared<A, BlockTransposed<i8, GROUP, PACK>, NR>
where
    A: Architecture,
    for<'a> mk::maxsim::packed_i8_x_unpacked_i8::Driver<'a, A, GROUP, NR, PACK>: mk::Drive,
{
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn compute_max_sim(
        &self,
        doc: MatRef<'_, Standard<i8>>,
        scores: &mut [i32],
    ) -> Result<(), MaxSimError> {
        if scores.len() != self.nrows() {
            return Err(MaxSimError::InvalidBufferLength(scores.len(), self.nrows()));
        }

        if doc.vector_dim() != self.prepared.ncols() {
            return Err(MaxSimError::UnequalDim(
                doc.vector_dim(),
                self.prepared.ncols(),
            ));
        }

        let Some(k) = NonZeroUsize::new(self.prepared.ncols()).map(mk::DimK::new) else {
            scores.fill(if doc.num_vectors() == 0 { i32::MAX } else { 0 });
            return Ok(());
        };

        let Some(a) = mk::blocks::packed::View::from_block_transposed(self.prepared.as_view())
        else {
            return Ok(());
        };

        let Some(b) = mk::blocks::unpacked::View::from_matrix_view(doc.as_matrix_view()) else {
            scores.fill(i32::MAX);
            return Ok(());
        };

        // SAFETY: The dimension check establishes that `a.k() == b.k() == k`.
        // The length check establishes that `scores` occupies exactly the
        // packed blocks in `a`.
        let mut driver = unsafe {
            mk::maxsim::packed_i8_x_unpacked_i8::Driver::new(
                self.arch,
                a,
                b,
                scores,
                k,
                mk::Cache::detect(),
            )
        };

        mk::Drive::drive(&mut driver);

        scores.iter_mut().for_each(|s| *s = -*s);

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  ReferenceKernel<T> — double loop over the single-vector inner product.
// ─────────────────────────────────────────────────────────────────────────

/// Reference MaxSim implementation, selected by [`MaxSimElement::build`].
///
/// May assume `scores` and `doc` were validated by
/// [`MaxSimKernel::compute_max_sim`].
type ReferenceFn<T> =
    fn(QueryMatRef<'_, Standard<T>>, MatRef<'_, Standard<T>>, &mut [<T as MaxSimElement>::Score]);

/// Unoptimized kernel backing [`MaxSimIsa::Reference`].
struct ReferenceKernel<T: MaxSimElement> {
    query: Mat<Standard<T>>,
    run: ReferenceFn<T>,
}

impl<T: MaxSimElement> std::fmt::Debug for ReferenceKernel<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceKernel")
            .field("nrows", &self.query.num_vectors())
            .finish()
    }
}

impl<T: MaxSimElement> ReferenceKernel<T> {
    fn new(query: MatRef<'_, Standard<T>>, run: ReferenceFn<T>) -> Self {
        Self {
            query: query.to_owned(),
            run,
        }
    }
}

impl<T: MaxSimElement> MaxSimKernel<T> for ReferenceKernel<T> {
    fn nrows(&self) -> usize {
        self.query.num_vectors()
    }

    fn compute_max_sim(
        &self,
        doc: MatRef<'_, Standard<T>>,
        scores: &mut [T::Score],
    ) -> Result<(), MaxSimError> {
        if scores.len() != self.nrows() {
            return Err(MaxSimError::InvalidBufferLength(scores.len(), self.nrows()));
        }
        if doc.vector_dim() != self.query.vector_dim() {
            return Err(MaxSimError::UnequalDim(
                doc.vector_dim(),
                self.query.vector_dim(),
            ));
        }
        (self.run)(self.query.as_view().into(), doc, scores);
        Ok(())
    }
}

/// [`ReferenceKernel`] implementation for element types scored in `f32`.
fn reference_scores<T: Copy>(
    query: QueryMatRef<'_, Standard<T>>,
    doc: MatRef<'_, Standard<T>>,
    scores: &mut [f32],
) where
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    FallbackKernel::max_sim_kernel(query, doc, |i, score| scores[i] = score);
}

/// [`ReferenceKernel`] implementation for `i8`, which scores in exact `i32`.
fn reference_scores_i8(
    query: QueryMatRef<'_, Standard<i8>>,
    doc: MatRef<'_, Standard<i8>>,
    scores: &mut [i32],
) {
    FallbackKernel::max_sim_kernel_i8(query, doc, |i, score| scores[i] = score);
}

// ─────────────────────────────────────────────────────────────────────────
//  BuildAndErase<E> — Target1 impls used by `dispatch1_no_features` (Auto).
// ─────────────────────────────────────────────────────────────────────────

struct BuildAndErase<E>(E);

// ───── f32 Target1 impls ─────

impl<E: Erase<f32>> diskann_wide::arch::Target1<Scalar, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Scalar, query: MatRef<'_, Standard<f32>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<2>,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<f32>> diskann_wide::arch::Target1<V3, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V3, query: MatRef<'_, Standard<f32>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 16>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<f32>> diskann_wide::arch::Target1<V4, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V4, query: MatRef<'_, Standard<f32>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 32>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

#[cfg(target_arch = "aarch64")]
impl<E: Erase<f32>> diskann_wide::arch::Target1<Neon, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Neon, query: MatRef<'_, Standard<f32>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

// ───── f16 Target1 impls ─────

impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<Scalar, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Scalar, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(
            query
                .as_matrix_view()
                .map(|v| diskann_wide::cast_f16_to_f32(*v))
                .as_view(),
        );
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<2>,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<V3, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V3, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 16>::from_matrix_view(
            query
                .as_matrix_view()
                .map(|v| diskann_wide::cast_f16_to_f32(*v))
                .as_view(),
        );
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<V4, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V4, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 32>::from_matrix_view(
            query
                .as_matrix_view()
                .map(|v| diskann_wide::cast_f16_to_f32(*v))
                .as_view(),
        );
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

#[cfg(target_arch = "aarch64")]
impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<Neon, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Neon, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        // Neon dispatches to Scalar (no Neon-specific kernel).
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(
            query
                .as_matrix_view()
                .map(|v| diskann_wide::cast_f16_to_f32(*v))
                .as_view(),
        );
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

// ───── i8 Target1 impls ─────

impl<E: Erase<i8>> diskann_wide::arch::Target1<Scalar, E::Output, MatRef<'_, Standard<i8>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Scalar, query: MatRef<'_, Standard<i8>>) -> E::Output {
        let prepared = BlockTransposed::<i8, 8, 2>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<2>,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<i8>> diskann_wide::arch::Target1<V3, E::Output, MatRef<'_, Standard<i8>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V3, query: MatRef<'_, Standard<i8>>) -> E::Output {
        let prepared = BlockTransposed::<i8, 16, 2>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared {
            arch,
            prepared,
            _packing: Pack::<6>,
        })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<i8>> diskann_wide::arch::Target1<V4, E::Output, MatRef<'_, Standard<i8>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V4, query: MatRef<'_, Standard<i8>>) -> E::Output {
        // V4 retargets to V3 until the VNNI kernel lands.
        diskann_wide::arch::Target1::<V3, _, _>::run(self, V3::from(arch), query)
    }
}

#[cfg(target_arch = "aarch64")]
impl<E: Erase<i8>> diskann_wide::arch::Target1<Neon, E::Output, MatRef<'_, Standard<i8>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Neon, query: MatRef<'_, Standard<i8>>) -> E::Output {
        // Neon retargets to Scalar until the dotprod kernel lands.
        diskann_wide::arch::Target1::<Scalar, _, _>::run(self, Scalar::from(arch), query)
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  MaxSimElement — sealed trait gating accepted element types.
// ─────────────────────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Scalar element types accepted by [`build_max_sim`].
///
/// Sealed: external crates cannot add impls. Quantized representations
/// (PQ, SQ, packed sub-byte) are intentionally excluded — they need
/// codebook/scale state that [`MatRef<'_, Standard<Self>>`] can't carry.
pub trait MaxSimElement: sealed::Sealed + Sized + Copy + Send + Sync + 'static {
    /// Score produced per query row: `f32` for floating-point elements, `i32`
    /// for `i8` where the inner product is exact.
    type Score: Copy + Default + PartialEq + std::fmt::Debug;

    /// Score written for every query row when the document set is empty.
    const NO_MATCH: Self::Score;

    /// Build the concrete kernel for this element type and hand it to
    /// `erase.erase(...)`.
    ///
    /// # Errors
    ///
    /// Returns [`NotSupported`] when the requested ISA cannot run on this
    /// build (e.g. AVX-512 unavailable; aarch64 on x86_64).
    fn build<E: Erase<Self>>(
        isa: MaxSimIsa,
        query: MatRef<'_, Standard<Self>>,
        erase: E,
    ) -> Result<E::Output, NotSupported>;
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for half::f16 {}
impl sealed::Sealed for i8 {}

impl MaxSimElement for f32 {
    type Score = f32;
    const NO_MATCH: f32 = f32::MAX;

    fn build<E: Erase<f32>>(
        isa: MaxSimIsa,
        query: MatRef<'_, Standard<f32>>,
        erase: E,
    ) -> Result<E::Output, NotSupported> {
        match isa {
            MaxSimIsa::Auto => Ok(diskann_wide::arch::dispatch1_no_features(
                BuildAndErase(erase),
                query,
            )),
            MaxSimIsa::Scalar => Ok(Scalar::new().run1(BuildAndErase(erase), query)),
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V3 => {
                let arch = V3::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "AVX2/FMA unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V4 => {
                let arch = V4::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "AVX-512 unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(not(target_arch = "x86_64"))]
            MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => Err(NotSupported {
                isa,
                reason: "x86_64 target only",
            }),
            #[cfg(target_arch = "aarch64")]
            MaxSimIsa::Neon => {
                let arch = Neon::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "Neon unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(not(target_arch = "aarch64"))]
            MaxSimIsa::Neon => Err(NotSupported {
                isa,
                reason: "aarch64 target only",
            }),
            MaxSimIsa::Reference => {
                Ok(erase.erase(ReferenceKernel::new(query, reference_scores::<f32>)))
            }
        }
    }
}

impl MaxSimElement for half::f16 {
    type Score = f32;
    const NO_MATCH: f32 = f32::MAX;

    fn build<E: Erase<half::f16>>(
        isa: MaxSimIsa,
        query: MatRef<'_, Standard<half::f16>>,
        erase: E,
    ) -> Result<E::Output, NotSupported> {
        match isa {
            MaxSimIsa::Auto => Ok(diskann_wide::arch::dispatch1_no_features(
                BuildAndErase(erase),
                query,
            )),
            MaxSimIsa::Scalar => Ok(Scalar::new().run1(BuildAndErase(erase), query)),
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V3 => {
                let arch = V3::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "AVX2/FMA unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V4 => {
                let arch = V4::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "AVX-512 unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(not(target_arch = "x86_64"))]
            MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => Err(NotSupported {
                isa,
                reason: "x86_64 target only",
            }),
            #[cfg(target_arch = "aarch64")]
            MaxSimIsa::Neon => {
                let arch = Neon::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "Neon unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(not(target_arch = "aarch64"))]
            MaxSimIsa::Neon => Err(NotSupported {
                isa,
                reason: "aarch64 target only",
            }),
            MaxSimIsa::Reference => {
                Ok(erase.erase(ReferenceKernel::new(query, reference_scores::<half::f16>)))
            }
        }
    }
}

impl MaxSimElement for i8 {
    type Score = i32;
    const NO_MATCH: i32 = i32::MAX;

    fn build<E: Erase<i8>>(
        isa: MaxSimIsa,
        query: MatRef<'_, Standard<i8>>,
        erase: E,
    ) -> Result<E::Output, NotSupported> {
        match isa {
            MaxSimIsa::Auto => Ok(diskann_wide::arch::dispatch1_no_features(
                BuildAndErase(erase),
                query,
            )),
            MaxSimIsa::Scalar => Ok(Scalar::new().run1(BuildAndErase(erase), query)),
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V3 => {
                let arch = V3::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "AVX2/FMA unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V4 => {
                let arch = V4::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "AVX-512 unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(not(target_arch = "x86_64"))]
            MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => Err(NotSupported {
                isa,
                reason: "x86_64 target only",
            }),
            #[cfg(target_arch = "aarch64")]
            MaxSimIsa::Neon => {
                let arch = Neon::new_checked().ok_or(NotSupported {
                    isa,
                    reason: "Neon unavailable on this CPU",
                })?;
                Ok(arch.run1(BuildAndErase(erase), query))
            }
            #[cfg(not(target_arch = "aarch64"))]
            MaxSimIsa::Neon => Err(NotSupported {
                isa,
                reason: "aarch64 target only",
            }),
            MaxSimIsa::Reference => {
                Ok(erase.erase(ReferenceKernel::new(query, reference_scores_i8)))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Factory entry point.
// ─────────────────────────────────────────────────────────────────────────

/// Build a multi-vector MaxSim kernel for any [`MaxSimElement`] type.
///
/// Thin wrapper over [`MaxSimElement::build`] so callers don't have to name
/// the trait at the call site.
///
/// # Errors
///
/// Returns [`NotSupported`] when the requested ISA cannot run on this build.
pub fn build_max_sim<T: MaxSimElement, E: Erase<T>>(
    isa: MaxSimIsa,
    query: MatRef<'_, Standard<T>>,
    erase: E,
) -> Result<E::Output, NotSupported> {
    T::build(isa, query, erase)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::{BoxErase, Chamfer, MaxSim, QueryMatRef};
    use diskann_vector::DistanceFunctionMut;

    /// Local helper trait — picks a sane test value of `T` from an `f32`
    /// so both `f32` and `half::f16` parameterizations share the same data
    /// generator.
    trait FromF32 {
        fn from_f32(v: f32) -> Self;
    }

    impl FromF32 for f32 {
        fn from_f32(v: f32) -> Self {
            v
        }
    }

    impl FromF32 for half::f16 {
        fn from_f32(v: f32) -> Self {
            diskann_wide::cast_f32_to_f16(v)
        }
    }

    impl FromF32 for i8 {
        fn from_f32(v: f32) -> Self {
            v as i8
        }
    }

    /// Projects a kernel score onto the `f32` distance the fallback path
    /// produces, so both parameterizations share the same assertions.
    trait ScoreAsF32: MaxSimElement {
        fn score_as_f32(score: Self::Score) -> f32;
    }

    impl ScoreAsF32 for f32 {
        fn score_as_f32(score: f32) -> f32 {
            score
        }
    }

    impl ScoreAsF32 for half::f16 {
        fn score_as_f32(score: f32) -> f32 {
            score
        }
    }

    impl ScoreAsF32 for i8 {
        fn score_as_f32(score: i32) -> f32 {
            if score == Self::NO_MATCH {
                f32::MAX
            } else {
                score as f32
            }
        }
    }

    fn scores_buffer<T: MaxSimElement>(len: usize) -> Vec<T::Score> {
        vec![T::Score::default(); len]
    }

    fn make_mat<T: Copy>(data: &[T], nrows: usize, ncols: usize) -> MatRef<'_, Standard<T>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    fn make_test_data<T: FromF32>(len: usize, ceil: usize, shift: usize) -> Vec<T> {
        (0..len)
            .map(|v| T::from_f32(((v + shift) % ceil) as f32))
            .collect()
    }

    /// Shapes for the `chamfer_matches_fallback` / `max_sim_matches_fallback`
    /// agreement checks: `(num_queries, num_docs, dim)`.
    ///
    /// Targets the factory wiring (query setup, score writeback) above the
    /// kernel layer; exhaustive panel/remainder coverage is pinned in
    /// `kernels::tiled_reduce::tests`.
    const TEST_CASES: &[(usize, usize, usize)] = &[
        (0, 1, 1),   // Empty query
        (1, 0, 1),   // Empty docs
        (1, 1, 0),   // Empty dim
        (1, 0, 0),   // Empty docs and dim
        (1, 1, 4),   // Degenerate
        (5, 3, 5),   // Prime k; nq > 1 and nd > 1 exercise per-row writeback
        (17, 4, 64), // A-panel remainder crossing both Scalar and V3 panel widths
        (16, 6, 32), // B-remainder ≠ 1 (V3 b_remainder = 2)
    ];

    fn check_chamfer_matches<T>(tol: f32, label: &str)
    where
        T: ScoreAsF32 + FromF32,
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<T>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<T>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let expected = Chamfer::evaluate(QueryMatRef::from(query), doc);

            let kernel = build_max_sim::<T, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
            let mut scores = scores_buffer::<T>(nq);
            kernel.compute_max_sim(doc, &mut scores).unwrap();
            let actual: f32 = scores.iter().map(|&s| T::score_as_f32(s)).sum();

            assert!(
                (actual - expected).abs() < tol,
                "{label}Chamfer mismatch for ({nq},{nd},{dim}): actual={actual}, expected={expected}",
            );
        }
    }

    fn check_max_sim_matches<T>(tol: f32, label: &str)
    where
        T: ScoreAsF32 + FromF32,
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<T>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<T>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let mut expected_scores = vec![0.0f32; nq];
            let _ = MaxSim::new(&mut expected_scores).evaluate(QueryMatRef::from(query), doc);

            let kernel = build_max_sim::<T, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
            let mut actual_scores = scores_buffer::<T>(nq);
            kernel.compute_max_sim(doc, &mut actual_scores).unwrap();

            for i in 0..nq {
                let actual = T::score_as_f32(actual_scores[i]);
                assert!(
                    (actual - expected_scores[i]).abs() < tol,
                    "{label}MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={actual}, expected={}",
                    expected_scores[i],
                );
            }
        }
    }

    /// The `i8` reference path is an independent integer implementation
    /// ([`FallbackKernel::max_sim_kernel_i8`]), so it needs its own guard; the
    /// `f32`/`f16` reference paths share `max_sim_kernel` with the oracle and
    /// would only be testing themselves.
    ///
    /// Every other ISA is reached via [`MaxSimIsa::Auto`] somewhere in the CI
    /// matrix. Widen into a full sweep once V4 and Neon gain native `i8`
    /// kernels instead of retargeting to V3 and Scalar.
    #[test]
    fn i8_reference_matches_oracle() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<i8>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<i8>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let mut expected = vec![0.0f32; nq];
            let _ = MaxSim::new(&mut expected).evaluate(QueryMatRef::from(query), doc);

            let kernel = build_max_sim::<i8, _>(MaxSimIsa::Reference, query, BoxErase).unwrap();
            let mut scores = scores_buffer::<i8>(nq);
            kernel.compute_max_sim(doc, &mut scores).unwrap();

            for i in 0..nq {
                let actual = <i8 as ScoreAsF32>::score_as_f32(scores[i]);
                assert!(
                    (actual - expected[i]).abs() < 1e-10,
                    "i8 reference MaxSim[{i}] mismatch for ({nq},{nd},{dim}): \
                     actual={actual}, expected={}",
                    expected[i],
                );
            }
        }
    }

    #[test]
    fn dimensions_f32() {
        let data = vec![1.0f32; 5 * 8];
        let query = make_mat(&data, 5, 8);
        let kernel = build_max_sim::<f32, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
        assert_eq!(kernel.nrows(), 5);
    }

    #[test]
    fn dimensions_f16() {
        let data = vec![diskann_wide::cast_f32_to_f16(1.0); 5 * 8];
        let query = make_mat(data.as_slice(), 5, 8);
        let kernel = build_max_sim::<half::f16, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
        assert_eq!(kernel.nrows(), 5);
    }

    #[test]
    fn dimensions_i8() {
        let data = vec![1i8; 5 * 8];
        let query = make_mat(&data, 5, 8);
        let kernel = build_max_sim::<i8, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
        assert_eq!(kernel.nrows(), 5);
    }

    fn check_size_mismatch<T>(label: &str)
    where
        T: MaxSimElement + FromF32,
    {
        let query_data = make_test_data::<T>(3 * 4, 4, 0);
        let doc_data = make_test_data::<T>(2 * 4, 4, 1);
        let query = make_mat(&query_data, 3, 4);
        let doc = make_mat(&doc_data, 2, 4);

        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            let kernel = build_max_sim::<T, _>(isa, query, BoxErase).unwrap();

            let mut too_short = scores_buffer::<T>(2);
            match kernel.compute_max_sim(doc, &mut too_short) {
                Err(MaxSimError::InvalidBufferLength(2, 3)) => {}
                other => {
                    panic!("{label}({isa:?}) expected InvalidBufferLength(2, 3), got {other:?}",)
                }
            }

            let mut too_long = scores_buffer::<T>(4);
            match kernel.compute_max_sim(doc, &mut too_long) {
                Err(MaxSimError::InvalidBufferLength(4, 3)) => {}
                other => {
                    panic!("{label}({isa:?}) expected InvalidBufferLength(4, 3), got {other:?}",)
                }
            }
        }
    }

    fn check_zero_docs_fills_sentinel<T>(label: &str)
    where
        T: MaxSimElement + FromF32,
    {
        let query_data = make_test_data::<T>(3 * 4, 4, 0);
        let doc_data: Vec<T> = Vec::new();
        let query = make_mat(&query_data, 3, 4);
        let doc = make_mat(doc_data.as_slice(), 0, 4);

        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            let kernel = build_max_sim::<T, _>(isa, query, BoxErase).unwrap();
            let mut scores = scores_buffer::<T>(3);
            kernel.compute_max_sim(doc, &mut scores).unwrap();
            for (i, &s) in scores.iter().enumerate() {
                assert_eq!(
                    s,
                    T::NO_MATCH,
                    "{label}({isa:?}) zero-doc slot {i} should be the NO_MATCH sentinel",
                );
            }
        }
    }

    fn check_zero_query<T>(label: &str)
    where
        T: MaxSimElement + FromF32,
    {
        let query_data: Vec<T> = Vec::new();
        let doc_data = make_test_data::<T>(2 * 4, 4, 0);
        let query = make_mat(query_data.as_slice(), 0, 4);
        let doc = make_mat(&doc_data, 2, 4);

        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            let kernel = build_max_sim::<T, _>(isa, query, BoxErase).unwrap();
            assert_eq!(
                kernel.nrows(),
                0,
                "{label}({isa:?}) empty query should yield nrows=0",
            );
            let mut scores = scores_buffer::<T>(0);
            kernel
                .compute_max_sim(doc, &mut scores)
                .unwrap_or_else(|e| panic!("{label}({isa:?}) expected Ok, got {e:?}"));
        }
    }

    macro_rules! test_matches_fallback {
        ($mod_name:ident, $ty:ty, $tol:expr, $label:literal) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn chamfer_matches_fallback() {
                    check_chamfer_matches::<$ty>($tol, $label);
                }

                #[test]
                fn max_sim_matches_fallback() {
                    check_max_sim_matches::<$ty>($tol, $label);
                }

                #[test]
                fn errors_on_size_mismatch() {
                    check_size_mismatch::<$ty>($label);
                }

                #[test]
                fn zero_docs_fills_sentinel() {
                    check_zero_docs_fills_sentinel::<$ty>($label);
                }

                #[test]
                fn zero_query_returns_ok() {
                    check_zero_query::<$ty>($label);
                }
            }
        };
    }

    test_matches_fallback!(f32, f32, 1e-10, "f32 ");
    test_matches_fallback!(f16, half::f16, 1e-10, "f16 ");
    test_matches_fallback!(i8, i8, 1e-10, "i8 ");
}
