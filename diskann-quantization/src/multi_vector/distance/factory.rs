// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Factory for constructing [`MaxSimKernel<T>`](super::kernel::MaxSimKernel)
//! implementations. See [`build_max_sim`] for the entry point.

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
//  Prepared<A, Q> — concrete kernel for the SIMD-dispatched paths.
// ─────────────────────────────────────────────────────────────────────────

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
//  ReferenceKernel<T> — non-SIMD fallback.
// ─────────────────────────────────────────────────────────────────────────

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
//  BuildAndErase<E> — Target1 carrier for dispatch1_no_features (Auto).
// ─────────────────────────────────────────────────────────────────────────

/// Maps each architecture to its optimal GROUP and hands the resulting
/// `Prepared` kernel to the caller's `Erase` visitor.
struct BuildAndErase<E>(E);

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
        // V4 dispatches to V3 (no V4-specific kernel).
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
        // Neon dispatches to Scalar (no Neon-specific kernel).
        let arch = arch.retarget();
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

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
        // V4 dispatches to V3 (no V4-specific kernel).
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
        // Neon dispatches to Scalar (no Neon-specific kernel).
        let arch = arch.retarget();
        let prepared = BlockTransposed::<half::f16, 8>::from_matrix_view(query.as_matrix_view());
        self.0.erase(Prepared { arch, prepared })
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  MaxSimElement — sealed trait gating accepted element types.
// ─────────────────────────────────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// Element types accepted by [`build_max_sim`]. Sealed — the library ships
/// impls for `f32` and `half::f16`.
pub trait MaxSimElement: sealed::Sealed + Sized + Copy + Send + Sync + 'static {
    /// Build a kernel for this element type. Returns [`NotSupported`] if the
    /// requested ISA is unavailable on this host.
    fn build<E: Erase<Self>>(
        isa: MaxSimIsa,
        query: MatRef<'_, Standard<Self>>,
        erase: E,
    ) -> Result<E::Output, NotSupported>;
}

impl sealed::Sealed for f32 {}
impl sealed::Sealed for half::f16 {}

impl MaxSimElement for f32 {
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
            MaxSimIsa::Reference => Ok(erase.erase(ReferenceKernel::<f32>::new(query))),
        }
    }
}

impl MaxSimElement for half::f16 {
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
            MaxSimIsa::Reference => Ok(erase.erase(ReferenceKernel::<half::f16>::new(query))),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Factory entry point.
// ─────────────────────────────────────────────────────────────────────────

/// Build a MaxSim kernel for the given ISA and query matrix, passing the
/// concrete kernel through `erase` for caller-controlled type erasure.
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

    fn make_mat<T: Copy>(data: &[T], nrows: usize, ncols: usize) -> MatRef<'_, Standard<T>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    fn make_test_data<T: FromF32>(len: usize, ceil: usize, shift: usize) -> Vec<T> {
        (0..len)
            .map(|v| T::from_f32(((v + shift) % ceil) as f32))
            .collect()
    }

    /// (num_queries, num_docs, dim) — exercises factory wiring, not kernel
    /// internals (those are pinned in `kernels::tiled_reduce::tests`).
    const TEST_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 4),   // Degenerate
        (5, 3, 5),   // Prime k; nq > 1 and nd > 1 exercise per-row writeback
        (17, 4, 64), // A-panel remainder crossing both Scalar and V3 panel widths
        (16, 6, 32), // B-remainder ≠ 1 (V3 b_remainder = 2)
    ];

    fn check_chamfer_matches<T>(tol: f32, label: &str)
    where
        T: MaxSimElement + FromF32,
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<T>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<T>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let expected = Chamfer::evaluate(QueryMatRef::from(query), doc);

            let kernel = build_max_sim::<T, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
            let mut scores = vec![0.0f32; nq];
            kernel.compute_max_sim(doc, &mut scores);
            let actual: f32 = scores.iter().sum();

            assert!(
                (actual - expected).abs() < tol,
                "{label}Chamfer mismatch for ({nq},{nd},{dim}): actual={actual}, expected={expected}",
            );
        }
    }

    fn check_max_sim_matches<T>(tol: f32, label: &str)
    where
        T: MaxSimElement + FromF32,
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
    {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data::<T>(nq * dim, dim, dim / 2);
            let doc_data = make_test_data::<T>(nd * dim, dim, dim);

            let query = make_mat(&query_data, nq, dim);
            let doc = make_mat(&doc_data, nd, dim);

            let mut expected_scores = vec![0.0f32; nq];
            let _ = MaxSim::new(&mut expected_scores)
                .unwrap()
                .evaluate(QueryMatRef::from(query), doc);

            let kernel = build_max_sim::<T, _>(MaxSimIsa::Auto, query, BoxErase).unwrap();
            let mut actual_scores = vec![0.0f32; nq];
            kernel.compute_max_sim(doc, &mut actual_scores);

            for i in 0..nq {
                assert!(
                    (actual_scores[i] - expected_scores[i]).abs() < tol,
                    "{label}MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={}, expected={}",
                    actual_scores[i],
                    expected_scores[i],
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
            }
        };
    }

    test_matches_fallback!(f32, f32, 1e-10, "f32 ");
    test_matches_fallback!(f16, half::f16, 1e-10, "f16 ");
}
