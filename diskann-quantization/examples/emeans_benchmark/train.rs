/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    Parallelism,
    algorithms::{
        emeans::{EmeansConfig, emeans},
        kmeans::lloyds,
    },
    cancel::DontCancel,
    product::{
        BasicTable,
        train::{LightPQTrainingParameters, TrainQuantizer},
    },
    random::{RngBuilder, StdRngBuilder},
    views::ChunkOffsets,
};
use diskann_utils::{
    strided::Strided,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use rand::{SeedableRng, rngs::StdRng, seq::index};
use serde::{Deserialize, Serialize};
use std::{error::Error, num::NonZeroUsize};

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Clustering,
    Pq,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    Lloyd,
    Emeans,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub algorithm: Algorithm,
    pub epochs: usize,
}

pub struct Trained {
    pub table: BasicTable,
    pub batch_size: Option<usize>,
    pub initial_step: Option<f64>,
    pub revived_clusters: usize,
}

// Both paths include their actual initialization, scratch preparation and table packing.
pub fn fit(
    data: rowmajor::Ref<'_, f32>,
    mode: Mode,
    clusters: usize,
    partitions: usize,
    run: Run,
    seed: u64,
) -> Result<Trained, Box<dyn Error>> {
    let schema = ChunkOffsets::partition(
        NonZeroUsize::new(data.ncols()).ok_or("zero dimension")?,
        NonZeroUsize::new(partitions).ok_or("zero partitions")?,
    )?;
    let builder = StdRngBuilder::new(seed);
    if mode == Mode::Pq && matches!(run.algorithm, Algorithm::Lloyd) {
        let table = LightPQTrainingParameters::new(clusters, run.epochs).train(
            data,
            schema.as_view(),
            Parallelism::Sequential,
            &builder,
            &DontCancel,
        )?;
        return Ok(Trained {
            table,
            batch_size: None,
            initial_step: None,
            revived_clusters: 0,
        });
    }
    let mut packed = rowmajor::Owned::from_element(clusters, data.ncols(), 0.0);
    let (mut batch_size, mut initial_step, mut revived_clusters) = (None, None, 0);
    for part in 0..partitions {
        let range = schema.as_view().at(part);
        let view = Strided::try_from_data(
            &data.as_slice()[range.start..],
            data.nrows(),
            range.len(),
            data.ncols(),
        )?;
        let mut centers = rowmajor::Owned::from_element(clusters, range.len(), 0.0);
        let mut rng = if mode == Mode::Clustering {
            StdRng::seed_from_u64(seed)
        } else {
            builder.build_rng(part)
        };
        match run.algorithm {
            Algorithm::Emeans => {
                let result = emeans(
                    view,
                    centers.as_view_mut(),
                    &EmeansConfig {
                        epochs: run.epochs,
                        ..Default::default()
                    },
                    &mut rng,
                    &DontCancel,
                )?;
                batch_size = Some(result.batch_size);
                initial_step = Some(result.initial_step_size);
                revived_clusters += result.revived_clusters;
            }
            Algorithm::Lloyd => {
                // Clustering mode has one partition. This samples exactly the same initial
                // rows as emeans(), using the same RNG seed and sampling implementation.
                for (center, row) in
                    centers
                        .rows_mut()
                        .zip(index::sample(&mut rng, data.nrows(), clusters))
                {
                    center.copy_from_slice(data.row(row));
                }
                lloyds::lloyds(data, centers.as_view_mut(), run.epochs);
            }
        }
        for (dst, src) in packed.rows_mut().zip(centers.rows()) {
            dst[range.clone()].copy_from_slice(src);
        }
    }
    Ok(Trained {
        table: BasicTable::new(packed, schema)?,
        batch_size,
        initial_step,
        revived_clusters,
    })
}
