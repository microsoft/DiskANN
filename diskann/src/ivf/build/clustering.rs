/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fit replacement centroids over provider vector views.

use std::num::NonZeroUsize;

use diskann_quantization::algorithms::kmeans::lloyds::lloyds;
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};

use crate::{ANNError, ANNResult, error::IntoANNResult, ivf::region::AsMatRef};

/// Fit a parent's replacement centroids without assigning regional memberships.
pub trait Clusterer<V, T = f32> {
    /// Fill `dst` with finite centroids of the canonical dimension.
    ///
    /// Implementations own initialization, convergence, empty-cluster handling,
    /// and reusable fitting scratch. `dst` has `num_centers()` rows and `dim()`
    /// columns. The number of centers must be at least two.
    fn fit(&mut self, points: V, dst: rowmajor::Mut<'_, T>) -> ANNResult<()>;

    fn num_centers(&self) -> NonZeroUsize;

    fn dim(&self) -> NonZeroUsize;
}

/// Random input seeds followed by Lloyd iterations, which reset empty centers to zero.
pub struct TwoMeans {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeans {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }
}

impl<B: AsMatRef<f32>> Clusterer<B> for TwoMeans {
    fn fit(&mut self, points: B, mut dst: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
        let dim = self.dim.get();
        let num_centers = <Self as Clusterer<B>>::num_centers(self).get();

        let vectors = points.as_view().into_ann_result()?;

        if vectors.nrows() < num_centers {
            return Err(ANNError::message("an IVF split needs at least two points"));
        }
        if vectors.ncols() != dim || dst.ncols() != dim || dst.nrows() != num_centers {
            return Err(ANNError::message(
                "two-means needs canonical input rows and a two-row destination of the canonical dimension",
            ));
        }

        dst.as_mut_slice().copy_from_slice(
            vectors
                .subview(0..num_centers)
                .ok_or(ANNError::message("oh no!"))?
                .as_slice(),
        );

        let _ = lloyds(vectors, dst, self.iterations.get());

        Ok(())
    }

    fn num_centers(&self) -> NonZeroUsize {
        const { crate::utils::TWO }
    }

    fn dim(&self) -> NonZeroUsize {
        self.dim
    }
}
