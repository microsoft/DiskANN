/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use anyhow::{ensure, Result};
use diskann_vector::{
    distance::{Cosine, CosineNormalized, InnerProduct, SquaredL2},
    PureDistanceFunction,
};

use crate::utils::SimilarityMeasure;

fn norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Check the normalized-input contract, allowing for FP16 storage rounding.
pub(super) fn validate_normalized(rows: &[Vec<f32>], label: &str) -> Result<()> {
    for (row, vector) in rows.iter().enumerate() {
        ensure!(
            (norm(vector) - 1.0).abs() <= 1.0e-3,
            "{label} row {row} is not unit length; cosine_normalized requires normalized inputs"
        );
    }
    Ok(())
}

/// All scores follow the accumulator's smaller-is-nearer convention.
pub(super) fn evaluate(metric: SimilarityMeasure, query: &[f32], point: &[f32]) -> f32 {
    match metric {
        SimilarityMeasure::SquaredL2 => SquaredL2::evaluate(query, point),
        SimilarityMeasure::InnerProduct => InnerProduct::evaluate(query, point),
        SimilarityMeasure::Cosine => Cosine::evaluate(query, point),
        SimilarityMeasure::CosineNormalized => CosineNormalized::evaluate(query, point),
    }
}

/// Prepare a search-only centroid. Means of normalized points are not generally unit vectors.
pub(super) fn prepare_centroid(metric: SimilarityMeasure, centroid: &[f32]) -> Vec<f32> {
    let mut prepared = centroid.to_vec();
    if matches!(
        metric,
        SimilarityMeasure::Cosine | SimilarityMeasure::CosineNormalized
    ) {
        // f64 avoids overflow and underflow when normalizing finite f32 coordinates.
        let norm = norm(centroid);
        if norm > 0.0 {
            for value in &mut prepared {
                *value = (f64::from(*value) / norm) as f32;
            }
        }
        // A zero mean has no direction; leave it zero rather than creating NaNs.
    }
    prepared
}
