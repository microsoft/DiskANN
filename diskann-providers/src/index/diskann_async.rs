/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use crate::model;

// Type aliases and index construction functions that depend on the in-memory
// providers have been moved to the `diskann-inmem` crate (`diskann_inmem::diskann_async`).

pub fn train_pq<Pool>(
    data: diskann_utils::views::MatrixView<f32>,
    num_pq_chunks: usize,
    rng: &mut dyn rand::RngCore,
    pool: Pool,
) -> ANNResult<model::pq::FixedChunkPQTable>
where
    Pool: crate::utils::AsThreadPool,
{
    let dim = data.ncols();
    let pivot_args = model::GeneratePivotArguments::new(
        data.nrows(),
        data.ncols(),
        model::pq::NUM_PQ_CENTROIDS,
        num_pq_chunks,
        5,
        false,
    )?;
    let mut centroid = vec![0.0; dim];
    let mut offsets = vec![0; num_pq_chunks + 1];
    let mut full_pivot_data = vec![0.0; model::pq::NUM_PQ_CENTROIDS * dim];

    model::pq::generate_pq_pivots_from_membuf(
        &pivot_args,
        data.as_slice(),
        &mut centroid,
        &mut offsets,
        &mut full_pivot_data,
        rng,
        &mut (false),
        pool,
    )?;

    model::pq::FixedChunkPQTable::new(
        dim,
        full_pivot_data.into(),
        centroid.into(),
        offsets.into(),
        None,
    )
}

// The test module has been moved to diskann-inmem since it tests the in-memory
// providers which now live in that crate. See diskann_inmem::diskann_async for
// the canonical helper functions (simplified_builder, new_index, new_quant_index, etc.)
