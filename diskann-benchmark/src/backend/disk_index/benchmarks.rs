/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::Serialize;
use std::io::Write;

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::{
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    output::Output,
    utils::datatype::{DataType, Type},
    Any, Checkpoint,
};
use diskann_providers::storage::FileStorageProvider;
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

impl<T> dispatcher::Map for DiskIndex<'static, T>
where
    T: 'static,
{
    type Type<'a> = DiskIndex<'a, T>;
}

/// Dispatch to Disk Index operations.
impl<'a, T> DispatchRule<&'a DiskIndexOperation> for DiskIndex<'a, T>
where
    Type<T>: DispatchRule<DataType>,
    T: VectorRepr,
{
    type Error = std::convert::Infallible;

    // Matching simply requires that we match the inner type.
    fn try_match(from: &&'a DiskIndexOperation) -> Result<MatchScore, FailureScore> {
        match &from.source {
            DiskIndexSource::Load(load) => Type::<T>::try_match(&load.data_type),
            DiskIndexSource::Build(build) => Type::<T>::try_match(&build.data_type),
        }
    }

    fn convert(from: &'a DiskIndexOperation) -> Result<Self, Self::Error> {
        Ok(Self::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a DiskIndexOperation>,
    ) -> std::fmt::Result {
        // At this level, we only care about the data type, so return that description.
        match from {
            Some(arg) => match &arg.source {
                DiskIndexSource::Load(load) => Type::<T>::description(f, Some(&load.data_type)),
                DiskIndexSource::Build(build) => Type::<T>::description(f, Some(&build.data_type)),
            },
            None => Type::<T>::description(f, None::<&DataType>),
        }
    }
}

/// Central Dispatch
impl<'a, T> DispatchRule<&'a Any> for DiskIndex<'a, T>
where
    Type<T>: DispatchRule<DataType>,
    T: VectorRepr,
{
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<DiskIndexOperation, Self>()
    }

    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<DiskIndexOperation, Self>()
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<DiskIndexOperation, Self>(f, from, DiskIndexOperation::tag())
    }
}

////////////////////////////
// Benchmark Registration //
////////////////////////////

macro_rules! register_disk_index {
    ($registry:ident, $name:literal, $t:ty) => {
        $registry.register::<DiskIndex<'static, $t>>($name, |object, checkpoint, output| {
            let res = object.run(checkpoint, output)?;
            Ok(serde_json::to_value(res)?)
        });
    };
}

pub(super) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    register_disk_index!(benchmarks, "disk-index-f32", f32);
    register_disk_index!(benchmarks, "disk-index-f16", f16);
    register_disk_index!(benchmarks, "disk-index-u8", u8);
    register_disk_index!(benchmarks, "disk-index-i8", i8);
}
