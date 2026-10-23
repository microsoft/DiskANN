/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ivf::workingset::PointRef;
use diskann_utils::views::rowmajor::{self, Matrix};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

use crate::inputs::graph_ivf_two_means::Layout;

pub(in crate::graph_ivf) enum Data {
    Contiguous { values: Vec<f32>, dim: usize },
    Scattered { rows: Vec<Vec<f32>> },
}

impl Data {
    pub(in crate::graph_ivf) fn new(
        layout: Layout,
        size: usize,
        dim: usize,
        seed: u64,
        corpus: Option<&rowmajor::Owned<f32>>,
    ) -> anyhow::Result<Self> {
        let len = size
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("workload dimensions overflow"))?;
        let mut rng = StdRng::seed_from_u64(seed ^ ((size as u64) << 32) ^ dim as u64);
        let values: Vec<f32> = if let Some(data) = corpus {
            let selected = rand::seq::index::sample(&mut rng, data.nrows(), size);
            selected
                .into_iter()
                .flat_map(|i| data.row(i).iter().copied())
                .collect()
        } else {
            (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
        };
        Ok(Self::from_values(layout, size, dim, seed, values))
    }

    pub(in crate::graph_ivf) fn from_values(
        layout: Layout,
        size: usize,
        dim: usize,
        seed: u64,
        values: Vec<f32>,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ ((size as u64) << 32) ^ dim as u64);
        match layout {
            Layout::Contiguous => Self::Contiguous { values, dim },
            Layout::Scattered => {
                let mut order: Vec<_> = (0..size).collect();
                order.shuffle(&mut rng);
                let mut rows = vec![Vec::new(); size];
                // Allocate in a shuffled order, then restore logical point order via references.
                // Randomizing allocation does not change values, point order, seed, or f64 sums.
                for &logical in &order {
                    rows[logical] = values[logical * dim..(logical + 1) * dim].to_vec();
                }
                Self::Scattered { rows }
            }
        }
    }

    pub(in crate::graph_ivf) fn points(&self) -> Vec<PointRef<'_, usize>> {
        match self {
            Self::Contiguous { values, dim } => values
                .chunks_exact(*dim)
                .enumerate()
                .map(|(id, vector)| PointRef { id, vector })
                .collect(),
            Self::Scattered { rows } => rows
                .iter()
                .enumerate()
                .map(|(id, vector)| PointRef { id, vector })
                .collect(),
        }
    }
}
