// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Factory + concrete `MaxSimKernel<T>` implementations for the multi-vector
//! distance API. See [`build_max_sim_f32`] / [`build_max_sim_f16`] for the
//! BYOTE entry points.

use diskann_utils::Reborrow;
use diskann_vector::distance::InnerProduct;
use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
use diskann_wide::Architecture;
use diskann_wide::arch::Scalar;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::isa::{MaxSimIsa, NotSupported};
use super::kernel::{Erase, MaxSimKernel};
use super::kernels::f16::F16Entry;
use super::kernels::f32::F32Kernel;
use super::max_sim::MaxSim;
use crate::multi_vector::distance::QueryMatRef;
use crate::multi_vector::{BlockTransposed, BlockTransposedRef, Mat, MatRef, Standard};

// ─────────────────────────────────────────────────────────────────────────
//  Prepared<A, Q> — concrete kernel for the arch-dispatched paths.
// ─────────────────────────────────────────────────────────────────────────

/// Concrete kernel: owns an arch token and a block-transposed prepared query.
/// One generic `MaxSimKernel<T>` impl covers every arch (Scalar/V3/V4/Neon)
/// for every supported element type (f32, f16) via the `Kernel<A>` / `Target3`
/// dispatch in the `kernels` module.
#[derive(Debug)]
struct Prepared<A, Q> {
    arch: A,
    prepared: Q,
}

impl<A, const GROUP: usize> MaxSimKernel<f32> for Prepared<A, BlockTransposed<f32, GROUP>>
where
    A: Architecture,
    F32Kernel<GROUP>: for<'a> diskann_wide::arch::Target3<
            A,
            (),
            BlockTransposedRef<'a, f32, GROUP>,
            MatRef<'a, Standard<f32>>,
            &'a mut [f32],
        >,
{
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn compute_max_sim(&self, doc: MatRef<'_, Standard<f32>>, scores: &mut [f32]) {
        let mut scratch = vec![f32::MIN; self.prepared.padded_nrows()];
        self.arch.run3(
            F32Kernel::<GROUP>,
            self.prepared.reborrow(),
            doc,
            &mut scratch,
        );
        for (dst, &src) in scores.iter_mut().zip(&scratch[..self.prepared.nrows()]) {
            *dst = -src;
        }
    }
}

impl<A, const GROUP: usize> MaxSimKernel<half::f16>
    for Prepared<A, BlockTransposed<half::f16, GROUP>>
where
    A: Architecture,
    F16Entry<GROUP>: for<'a> diskann_wide::arch::Target3<
            A,
            (),
            BlockTransposedRef<'a, half::f16, GROUP>,
            MatRef<'a, Standard<half::f16>>,
            &'a mut [f32],
        >,
{
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn compute_max_sim(&self, doc: MatRef<'_, Standard<half::f16>>, scores: &mut [f32]) {
        let mut scratch = vec![f32::MIN; self.prepared.padded_nrows()];
        self.arch.run3(
            F16Entry::<GROUP>,
            self.prepared.reborrow(),
            doc,
            &mut scratch,
        );
        for (dst, &src) in scores.iter_mut().zip(&scratch[..self.prepared.nrows()]) {
            *dst = -src;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  ReferenceKernel<T> — non-SIMD fallback that wraps MaxSim::evaluate.
// ─────────────────────────────────────────────────────────────────────────

/// `MaxSimIsa::Reference` path. Owns the query as a `Mat<Standard<T>>` and
/// delegates to the existing `MaxSim` fallback per `compute_max_sim` call.
struct ReferenceKernel<T: Copy> {
    query: Mat<Standard<T>>,
}

impl<T: Copy + std::fmt::Debug> std::fmt::Debug for ReferenceKernel<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceKernel")
            .field("nrows", &self.query.num_vectors())
            .finish()
    }
}

impl<T: Copy> ReferenceKernel<T> {
    fn new(query: MatRef<'_, Standard<T>>) -> Self {
        let repr = *query.repr();
        let src = query.as_slice();
        let mut idx = 0usize;
        let owned = Mat::<Standard<T>>::from_fn(repr, || {
            let v = src[idx];
            idx += 1;
            v
        });
        Self { query: owned }
    }
}

impl<T> MaxSimKernel<T> for ReferenceKernel<T>
where
    T: Copy + Send + Sync + std::fmt::Debug + 'static,
    InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
{
    fn nrows(&self) -> usize {
        self.query.num_vectors()
    }

    fn compute_max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]) {
        if scores.is_empty() {
            return;
        }
        let query: QueryMatRef<'_, Standard<T>> = self.query.as_view().into();
        let Ok(mut max_sim) = MaxSim::new(scores) else {
            return;
        };
        let _ = max_sim.evaluate(query, doc);
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  BuildAndErase<E> — Target1 impls used by `dispatch1_no_features` (Auto).
// ─────────────────────────────────────────────────────────────────────────

/// Internal Target1 carrier used only by the `MaxSimIsa::Auto` arm of
/// `build_max_sim_*`. `dispatch1_no_features` picks the highest available
/// arch on the host CPU and calls the matching `Target1::run` below.
struct BuildAndErase<E>(E);

// ───── f32 Target1 impls ─────

impl<E: Erase<f32>> diskann_wide::arch::Target1<Scalar, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Scalar, query: MatRef<'_, Standard<f32>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<f32>> diskann_wide::arch::Target1<V3, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V3, query: MatRef<'_, Standard<f32>>) -> E::Output {
        let prepared = BlockTransposed::<f32, 16>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<f32>> diskann_wide::arch::Target1<V4, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V4, query: MatRef<'_, Standard<f32>>) -> E::Output {
        // V4 has no dedicated kernel yet; retarget to V3.
        let arch = arch.retarget();
        let prepared = BlockTransposed::<f32, 16>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

#[cfg(target_arch = "aarch64")]
impl<E: Erase<f32>> diskann_wide::arch::Target1<Neon, E::Output, MatRef<'_, Standard<f32>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Neon, query: MatRef<'_, Standard<f32>>) -> E::Output {
        // Neon has no dedicated kernel yet; retarget to Scalar.
        let arch = arch.retarget();
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

// ───── f16 Target1 impls ─────

impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<Scalar, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Scalar, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let prepared = BlockTransposed::<half::f16, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<V3, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V3, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let prepared = BlockTransposed::<half::f16, 16>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

#[cfg(target_arch = "x86_64")]
impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<V4, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: V4, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let arch = arch.retarget();
        let prepared = BlockTransposed::<half::f16, 16>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

#[cfg(target_arch = "aarch64")]
impl<E: Erase<half::f16>>
    diskann_wide::arch::Target1<Neon, E::Output, MatRef<'_, Standard<half::f16>>>
    for BuildAndErase<E>
{
    fn run(self, arch: Neon, query: MatRef<'_, Standard<half::f16>>) -> E::Output {
        let arch = arch.retarget();
        let prepared = BlockTransposed::<half::f16, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Factory functions.
// ─────────────────────────────────────────────────────────────────────────

/// Build a multi-vector MaxSim kernel for `f32` queries.
///
/// Dispatches on `isa`, constructs the corresponding concrete kernel, and
/// hands it to `erase.erase(...)`. Returns [`NotSupported`] when the requested
/// ISA cannot run on this build (e.g. AVX-512 unavailable; aarch64 on x86_64).
pub fn build_max_sim_f32<E: Erase<f32>>(
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
        MaxSimIsa::Reference => Ok(erase.erase(ReferenceKernel::<f32>::new(query))),
    }
}

/// Build a multi-vector MaxSim kernel for `half::f16` queries. Same contract
/// as [`build_max_sim_f32`].
pub fn build_max_sim_f16<E: Erase<half::f16>>(
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
        MaxSimIsa::Reference => Ok(erase.erase(ReferenceKernel::<half::f16>::new(query))),
    }
}
