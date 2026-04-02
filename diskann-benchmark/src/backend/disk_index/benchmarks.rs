/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::Serialize;
use std::io::Write;

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{
    dispatcher::{DispatchRule, FailureScore, MatchScore},
    output::Output,
    utils::datatype::{DataType, Type},
    Benchmark, Checkpoint,
};
use diskann_storage::FileStorageProvider;
use half::f16;

use crate::{
    backend::disk_index::{
        build::{build_disk_index, DiskBuildStats},
        search::{search_disk_index, DiskSearchStats},
    },
    inputs::disk::{DiskIndexLoad, DiskIndexOperation, DiskIndexSource},
};

/// Disk Index
struct DiskIndex<'a, T> {
    input: &'a DiskIndexOperation,
    _vector_type: std::marker::PhantomData<T>,
}

#[derive(Debug, Serialize)]
pub(super) struct DiskIndexStats {
    pub(super) build: Option<DiskBuildStats>,
    pub(super) search: DiskSearchStats,
}

impl<'a, T> DiskIndex<'a, T>
where
    T: VectorRepr,
{
    fn new(input: &'a DiskIndexOperation) -> Self {
        Self {
            input,
            _vector_type: std::marker::PhantomData,
        }
    }

    fn run(
        &self,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> Result<DiskIndexStats, anyhow::Error> {
        writeln!(output, "{}", self.input.source)?;
        let (build_stats, index_load) = match &self.input.source {
            DiskIndexSource::Load(load) => Ok((None, (*load).clone())),
            DiskIndexSource::Build(build) => build_disk_index::<T, _>(&FileStorageProvider, build)
                .map(|stats| {
                    (
                        Some(stats),
                        DiskIndexLoad {
                            data_type: build.data_type,
                            load_path: build.save_path.clone(),
                        },
                    )
                }),
        }?;
        if let Some(build_stats) = &build_stats {
            writeln!(output, "{}", build_stats)?;
        }

        writeln!(output, "{}", self.input.search_phase)?;
        let search_stats =
            search_disk_index::<T, _>(&index_load, &self.input.search_phase, &FileStorageProvider)?;
        writeln!(output, "{}", search_stats)?;

        Ok(DiskIndexStats {
            build: build_stats,
            search: search_stats,
        })
    }
}

impl<T> Benchmark for DiskIndex<'static, T>
where
    T: VectorRepr + 'static,
    Type<T>: DispatchRule<DataType>,
{
    type Input = DiskIndexOperation;
    type Output = DiskIndexStats;

    fn try_match(input: &DiskIndexOperation) -> Result<MatchScore, FailureScore> {
        match &input.source {
            DiskIndexSource::Load(load) => Type::<T>::try_match(&load.data_type),
            DiskIndexSource::Build(build) => Type::<T>::try_match(&build.data_type),
        }
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&DiskIndexOperation>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => match &arg.source {
                DiskIndexSource::Load(load) => Type::<T>::description(f, Some(&load.data_type)),
                DiskIndexSource::Build(build) => Type::<T>::description(f, Some(&build.data_type)),
            },
            None => Type::<T>::description(f, None::<&DataType>),
        }
    }

    fn run(
        input: &DiskIndexOperation,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<DiskIndexStats> {
        DiskIndex::<T>::new(input).run(checkpoint, output)
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(super) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    benchmarks.register::<DiskIndex<'static, f32>>("disk-index-f32");
    benchmarks.register::<DiskIndex<'static, f16>>("disk-index-f16");
    benchmarks.register::<DiskIndex<'static, u8>>("disk-index-u8");
    benchmarks.register::<DiskIndex<'static, i8>>("disk-index-i8");
}
