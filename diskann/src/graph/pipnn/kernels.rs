/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Partition and leaf kernels behind one object per build.
//!
//! The kernels compile once for each SIMD architecture and metric. [`Kernels`]
//! hides both, so the partition and leaf steps compile once and call the
//! kernels through a trait object. One call covers a stripe or a leaf, so the
//! dynamic dispatch costs nothing measurable next to the kernel work.
//! [`select`] runs the architecture dispatch of `diskann-wide` once per build.

use std::marker::PhantomData;

use crate::ANNResult;
use diskann_utils::views::rowmajor;
use diskann_vector::distance::Metric;
use diskann_wide::arch::{self, Target};

use super::{
    Cosine, CosineNormalized, InnerProduct, L2,
    leaf_kernel::{LeafKernelWorkspace, select_leaf_neighbors},
    leaf_metric::LeafMetric,
    partition_kernel::{PartitionKernelWorkspace, assign_leaders},
    partition_metric::{PartitionLeaders, PartitionMetric},
    simd::Simd,
    topk::Candidate,
};

/// The partition and leaf kernels of one architecture and metric.
pub(super) trait Kernels: Sync {
    /// Create the leader set of one split from a matrix with one leader in
    /// each row. The matrix must not be empty.
    fn create_leaders<'a>(&self, values: rowmajor::Ref<'a, f32>) -> PartitionLeaders<'a>;

    /// Assign each point of one stripe to its nearest leaders.
    ///
    /// See [`assign_leaders`] for the contract of `points`, `output` and
    /// `workspace`.
    fn assign_leaders(
        &self,
        points: rowmajor::Ref<'_, f32>,
        leaders: &PartitionLeaders<'_>,
        output: rowmajor::Mut<'_, u32>,
        workspace: &mut PartitionKernelWorkspace,
    ) -> ANNResult<()>;

    /// Find the k nearest other points of each point in one leaf.
    ///
    /// See [`select_leaf_neighbors`] for the contract of `points`, `output` and
    /// `workspace`.
    fn select_leaf_neighbors(
        &self,
        points: rowmajor::Ref<'_, f32>,
        output: rowmajor::Mut<'_, Candidate>,
        workspace: &mut LeafKernelWorkspace,
    ) -> ANNResult<()>;
}

/// Select the kernels of `metric` for the architecture that `diskann-wide`
/// detects on this CPU.
pub(super) fn select(metric: Metric) -> Box<dyn Kernels> {
    arch::dispatch_no_features(Select { metric })
}

/// Boxes the kernels of one architecture.
struct Select {
    metric: Metric,
}

impl<A> Target<A, Box<dyn Kernels>> for Select
where
    A: Simd,
{
    fn run(self, arch: A) -> Box<dyn Kernels> {
        fn boxed<A, M>(arch: A) -> Box<dyn Kernels>
        where
            A: Simd,
            M: LeafMetric + PartitionMetric,
        {
            Box::new(Kernel::<A, M> {
                arch,
                metric: PhantomData,
            })
        }
        match self.metric {
            Metric::L2 => boxed::<A, L2>(arch),
            Metric::Cosine => boxed::<A, Cosine>(arch),
            Metric::CosineNormalized => boxed::<A, CosineNormalized>(arch),
            Metric::InnerProduct => boxed::<A, InnerProduct>(arch),
        }
    }
}

/// The kernels of architecture `A` and metric `M`.
struct Kernel<A, M> {
    arch: A,
    metric: PhantomData<M>,
}

impl<A, M> Kernels for Kernel<A, M>
where
    A: Simd,
    M: LeafMetric + PartitionMetric,
{
    fn create_leaders<'a>(&self, values: rowmajor::Ref<'a, f32>) -> PartitionLeaders<'a> {
        M::create_leaders(values)
    }

    fn assign_leaders(
        &self,
        points: rowmajor::Ref<'_, f32>,
        leaders: &PartitionLeaders<'_>,
        output: rowmajor::Mut<'_, u32>,
        workspace: &mut PartitionKernelWorkspace,
    ) -> ANNResult<()> {
        assign_leaders::<A, M>(self.arch, points, leaders, output, workspace)
    }

    fn select_leaf_neighbors(
        &self,
        points: rowmajor::Ref<'_, f32>,
        output: rowmajor::Mut<'_, Candidate>,
        workspace: &mut LeafKernelWorkspace,
    ) -> ANNResult<()> {
        select_leaf_neighbors::<A, M>(self.arch, points, output, workspace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_utils::views::rowmajor::{Matrix, MatrixMut};

    #[test]
    fn selected_kernels_rank_with_their_metric() {
        // Point 0 is the query. Each metric ranks the other points differently:
        // L2 picks point 3 (nearest), inner product picks point 2 (largest dot),
        // and cosine picks point 1 (smallest angle). Normalized cosine ranks by
        // the dot product, like inner product.
        let x = [[1.0_f32, 0.0], [1.2, 0.3], [5.0, 2.0], [0.95, 0.3]];
        let dimensions = 2;
        let points = rowmajor::Ref::try_from_data(x.as_flattened(), x.len(), dimensions).unwrap();
        let leaders = rowmajor::Ref::try_from_data(x[1..].as_flattened(), 3, dimensions).unwrap();
        let k = 1;

        for (metric, nearest) in [
            (Metric::L2, 3),
            (Metric::InnerProduct, 2),
            (Metric::Cosine, 1),
            (Metric::CosineNormalized, 2),
        ] {
            let kernels = select(metric);

            let mut neighbors = rowmajor::Owned::from_element(x.len(), k, Candidate::default());
            kernels
                .select_leaf_neighbors(
                    points,
                    neighbors.as_view_mut(),
                    &mut LeafKernelWorkspace::default(),
                )
                .unwrap();
            assert_eq!(
                neighbors.row(0)[0].local_idx as usize,
                nearest,
                "{metric:?} leaf"
            );

            let leader_set = kernels.create_leaders(leaders);
            assert_eq!(leader_set.leader_count(), 3, "{metric:?} leaders");
            let mut assignments = rowmajor::Owned::from_element(1, k, 0u32);
            kernels
                .assign_leaders(
                    points.subview(0..1).unwrap(),
                    &leader_set,
                    assignments.as_view_mut(),
                    &mut PartitionKernelWorkspace::default(),
                )
                .unwrap();
            // Leader i is point i + 1.
            assert_eq!(
                assignments.row(0)[0] as usize,
                nearest - 1,
                "{metric:?} partition"
            );
        }
    }
}
