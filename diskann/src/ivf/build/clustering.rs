/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fit replacement centroids over provider vector views.

use std::num::NonZeroUsize;

use diskann_quantization::algorithms::kmeans::lloyds::lloyds;
use diskann_utils::{
    Reborrow,
    views::rowmajor::{self, Matrix, MatrixMut},
};

use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    ivf::region::{AsMatRef, RegionView},
};

/// Fit replacement centroids for selected sources without assigning regional memberships.
pub trait Clusterer<I, L, R, T = f32>
where
    R: RegionView<I, L>,
{
    /// Fill `dst` with finite centroids of the canonical dimension.
    ///
    /// Implementations own initialization, convergence, empty-cluster handling,
    /// and reusable fitting scratch. `dst` has `num_centers(sources.len())` rows
    /// and `dim()` columns. The total number of centers must be at least two.
    fn fit(&mut self, sources: &[L], region: &R, dst: rowmajor::Mut<'_, T>) -> ANNResult<()>;

    /// Total output row count for this source count, or `None` on overflow.
    fn num_centers(&self, sources: usize) -> Option<usize>;

    fn dim(&self) -> NonZeroUsize;
}

/// First-two-row seeds followed by Lloyd iterations, which reset empty centers to zero.
pub struct TwoMeans {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeans {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }

    const fn inner_centers() -> usize {
        2
    }

    fn fit_one(
        &self,
        points: rowmajor::Ref<'_, f32>,
        mut dst: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()> {
        if points.nrows() < Self::inner_centers() || points.ncols() != self.dim.get() {
            return Err(ANNError::message(
                "IVF two-means needs at least two points with the configured dimension",
            ));
        }

        dst.as_mut_slice().copy_from_slice(
            points
                .subview(0..Self::inner_centers())
                .ok_or_else(|| ANNError::message("IVF two-means seed rows are missing"))?
                .as_slice(),
        );

        let _ = lloyds(points, dst, self.iterations.get());

        Ok(())
    }
}

impl<I, L: Copy, R> Clusterer<I, L, R> for TwoMeans
where
    R: RegionView<I, L>,
    for<'a> R::VectorsRef<'a>: AsMatRef<f32>,
{
    fn fit(&mut self, sources: &[L], region: &R, mut dst: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
        if dst.ncols() != self.dim.get()
            || sources.len().checked_mul(Self::inner_centers()) != Some(dst.nrows())
        {
            return Err(ANNError::message(
                "IVF two-means scratch needs two rows per source and the configured dimension",
            ));
        }

        let mut s = 0;
        for &p in sources {
            let vecs = region.get_vectors(p).ok_or_else(|| {
                ANNError::message("IVF split source vectors are missing from the region")
            })?;

            let vecs = vecs.reborrow();

            let points = vecs.as_view().into_ann_result()?;

            let end = s + Self::inner_centers();

            self.fit_one(
                points,
                dst.subview_mut(s..end).ok_or_else(|| {
                    ANNError::message("IVF centroid scratch range is out of bounds")
                })?,
            )?;

            s = end;
        }

        Ok(())
    }

    fn num_centers(&self, sources: usize) -> Option<usize> {
        sources.checked_mul(Self::inner_centers())
    }

    fn dim(&self) -> NonZeroUsize {
        self.dim
    }
}
