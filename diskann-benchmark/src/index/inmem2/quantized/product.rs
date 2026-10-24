/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::{self as dbr};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{self as dv};

use crate::{index::inmem2, utils::nz};

pub(super) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2::quantized) struct Product {
        pub(super) chunks: NonZeroUsize,
        pub(super) centers: NonZeroUsize,
        pub(super) seed: u64,
        pub(super) num_threads: NonZeroUsize,
    }

    impl Product {
        pub(super) fn example() -> Self {
            nz!(CHUNKS = 16);
            nz!(CENTERS = 16);
            nz!(NUM_THREADS = 8);

            Self {
                chunks: CHUNKS,
                centers: CENTERS,
                seed: 0xc0ffee,
                num_threads: NUM_THREADS,
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct Product {
    chunks: NonZeroUsize,
    centers: NonZeroUsize,
    seed: u64,
    metric: di::repr::quantized::product::Metric,
    num_threads: NonZeroUsize,
}

impl inmem2::quantized::Quantizer for Product {
    const KIND: &'static str = "product";
    const TAG: &'static str = "inmem2-pq-static";

    type Raw = dto::Product;

    fn from_raw(raw: &dto::Product, metric: dv::distance::Metric) -> anyhow::Result<Self> {
        let dto::Product {
            chunks,
            centers,
            seed,
            num_threads,
        } = raw;

        nz!(MAX_CENTERS = 256);

        // We encode to bytes - if there are more than 256 centers, we'll end up losing
        // information.
        if *centers > MAX_CENTERS {
            anyhow::bail!("more than 256 centers is unsupported - got {}", centers);
        }

        Ok(Self {
            chunks: *chunks,
            centers: *centers,
            seed: *seed,
            metric: metric.into(),
            num_threads: *num_threads,
        })
    }

    fn example() -> dto::Product {
        dto::Product::example()
    }

    fn train(&self, data: rowmajor::Ref<'_, f32>) -> anyhow::Result<di::repr::quantized::Backend> {
        use diskann_quantization::{
            cancel::DontCancel,
            product::{self, train::TrainQuantizer},
            random,
            views::ChunkOffsets,
            Parallelism,
        };

        let Self {
            chunks,
            centers,
            seed,
            metric,
            num_threads,
        } = self;

        let trainer = product::train::LightPQTrainingParameters::new(centers.get(), 5);
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.get())
            .build()?;

        let Some(dim) = NonZeroUsize::new(data.ncols()) else {
            anyhow::bail!("zero dimensional data is not supported");
        };

        let table = threadpool.install(|| -> anyhow::Result<_> {
            let table = trainer.train(
                data,
                ChunkOffsets::partition(dim, *chunks)?.as_view(),
                Parallelism::Rayon,
                &random::StdRngBuilder::new(*seed),
                &DontCancel,
            )?;

            Ok(table)
        })?;

        let quantizer =
            di::repr::quantized::product::Product::new(table, *metric, Some(*num_threads))?;

        Ok(di::repr::quantized::Backend::new(quantizer))
    }
}

impl std::fmt::Display for Product {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            chunks,
            centers,
            seed,
            metric,
            num_threads,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("chunks", chunks);
        kv.push("centers", centers);
        kv.push("seed", seed);
        kv.push_eager("metric", format_args!("{:?}", metric));
        kv.push("num_threads", num_threads);
        write!(f, "{}", kv)
    }
}

///////////
// tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_parses() {
        assert!(<Product as inmem2::quantized::Quantizer>::from_raw(
            &dto::Product::example(),
            dv::distance::Metric::L2
        )
        .is_ok());
    }
}
