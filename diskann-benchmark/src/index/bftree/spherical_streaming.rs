/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
////////////////////////
// Streaming BfTree SQ //
////////////////////////

use std::{borrow::Cow, io::Write, num::NonZeroUsize, sync::Arc};

use diskann::graph::{DiskANNIndex, InplaceDeleteMethod};
use diskann::utils::ONE;
use diskann_benchmark_core as benchmark_core;
use diskann_benchmark_core::{recall::Rows, streaming::executors::bigann};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{quant::QuantVectorProvider, BfTreeProvider};
use diskann_providers::model::graph::provider::async_::common::Quantized;
use diskann_quantization::alloc::{AllocatorError, GlobalAllocator, Poly};
use diskann_quantization::spherical::{
    iface::{self as spherical_iface, Quantizer},
    SphericalQuantizer,
};
use diskann_utils::views::{Matrix, MatrixView};
use rand::SeedableRng;

use crate::{
    index::{
        build::{BuildKind, BuildStats},
        search::knn,
        streaming::{
            managed::{self, Managed},
            stats::{GenericStats, StreamStats},
            ManagedStream,
        },
    },
    inputs::{
        bftree::BfTreeSphericalDynamicRun,
        graph_index::{SearchPhase, TopkSearchPhase},
    },
    utils::{self, datafiles},
};

type BfTreeSQProvider = BfTreeProvider<f32, QuantVectorProvider>;
type BfTreeSQIndex = Arc<DiskANNIndex<BfTreeSQProvider>>;

fn new_quantizer<const NBITS: usize>(
    quantizer: SphericalQuantizer,
) -> Result<Poly<dyn Quantizer>, AllocatorError>
where
    spherical_iface::Impl<NBITS>: spherical_iface::Constructible + Quantizer,
{
    let imp = spherical_iface::Impl::<NBITS>::new(quantizer)?;
    diskann_quantization::poly!(Quantizer, imp, GlobalAllocator)
}

struct BfTreeSQStream {
    index: BfTreeSQIndex,
    search: TopkSearchPhase,
    runtime: tokio::runtime::Runtime,
    ntasks: NonZeroUsize,
    inplace_delete_num_to_replace: usize,
    inplace_delete_method: InplaceDeleteMethod,
}

impl BfTreeSQStream {
    fn insert_(&self, data: MatrixView<'_, f32>, slots: &[u32]) -> anyhow::Result<BuildStats> {
        let runner = benchmark_core::build::graph::SingleInsert::new(
            self.index.clone(),
            Arc::new(data.to_owned()),
            Quantized,
            benchmark_core::build::ids::Slice::new(slots.into()),
        );

        let results = benchmark_core::build::build(
            runner,
            benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        BuildStats::new(BuildKind::SingleInsert, results)
    }
}

impl ManagedStream<f32> for BfTreeSQStream {
    type Output = StreamStats;

    fn search(
        &self,
        queries: Arc<Matrix<f32>>,
        groundtruth: &dyn Rows<u32>,
    ) -> anyhow::Result<Self::Output> {
        let knn = benchmark_core::search::graph::KNN::new(
            self.index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(Quantized),
        )?;

        let steps = knn::SearchSteps::new(
            self.search.reps,
            &self.search.num_threads,
            &self.search.runs,
        );
        let results = knn::run(&knn, groundtruth, steps)?;
        Ok(StreamStats::Search(results))
    }

    fn insert(&self, data: MatrixView<'_, f32>, slots: &[u32]) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Insert(self.insert_(data, slots)?))
    }

    fn replace(&self, data: MatrixView<'_, f32>, slots: &[u32]) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Replace(self.insert_(data, slots)?))
    }

    fn delete(&self, slots: &[u32]) -> anyhow::Result<Self::Output> {
        let runner = benchmark_core::streaming::graph::InplaceDelete::new(
            self.index.clone(),
            Quantized,
            self.inplace_delete_num_to_replace,
            self.inplace_delete_method,
            benchmark_core::build::ids::Slice::new(slots.into()),
        );

        let results = benchmark_core::build::build(
            runner,
            benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        Ok(StreamStats::Delete(GenericStats::new(
            Cow::Borrowed("Delete"),
            results,
        )?))
    }

    fn maintain(&self) -> anyhow::Result<Self::Output> {
        // bf-tree uses hard deletes — no deferred cleanup needed.
        Ok(StreamStats::Maintain(Vec::new()))
    }
}

/// The streaming benchmark for bf_tree spherical quantization.
///
/// Dispatches `num_bits` at runtime to avoid const-generic monomorphization.
pub(super) struct StreamingSpherical;

impl StreamingSpherical {
    pub(super) fn new() -> Self {
        Self
    }
}

impl Benchmark for StreamingSpherical {
    type Input = BfTreeSphericalDynamicRun;
    type Output = Vec<managed::Stats<StreamStats>>;

    fn try_match(&self, input: &Self::Input) -> Result<MatchScore, FailureScore> {
        let mut failure_score: Option<u32> = None;

        if let Err(s) = utils::match_data_type::<f32>(input.data_type()) {
            failure_score = Some(s.0);
        }
        if !matches!(input.num_bits().get(), 1 | 2 | 4) {
            *failure_score.get_or_insert(0) += 1;
        }
        if !matches!(input.search_phase(), SearchPhase::Topk(_)) {
            *failure_score.get_or_insert(0) += 1;
        }

        match failure_score {
            None => Ok(MatchScore(0)),
            Some(score) => Err(FailureScore(score)),
        }
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Self::Input>,
    ) -> std::fmt::Result {
        match input {
            None => {
                writeln!(f, "- BfTree Streaming with spherical quantization")
            }
            Some(input) => {
                if !f32::is_match(input.data_type()) {
                    writeln!(f, "- Only `float32` supported, got {}", input.data_type())?;
                }
                Ok(())
            }
        }
    }

    fn run(
        &self,
        input: &Self::Input,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;

        super::streaming_utils::run_streaming::<f32, _>(
            input.runbook_params(),
            |max_points| bftree_sq_streaming_impl(input, max_points),
            output,
        )
    }
}

fn bftree_sq_streaming_impl(
    input: &BfTreeSphericalDynamicRun,
    max_points: usize,
) -> anyhow::Result<bigann::WithData<f32, u32, Managed<f32, StreamStats>>> {
    let topk = match input.search_phase() {
        SearchPhase::Topk(topk) => topk,
        _ => anyhow::bail!("Only TopK is currently supported by the streaming index"),
    };

    let data = datafiles::load_dataset::<f32>(datafiles::BinFile(input.build().data()))?;
    let queries = Arc::new(datafiles::load_dataset::<f32>(datafiles::BinFile(
        &topk.queries,
    ))?);

    // Train the spherical quantizer.
    let m: diskann_vector::distance::Metric = input.build().distance().into();
    let pre_scale = match input.pre_scale() {
        Some(&v) => v.try_into()?,
        None => diskann_quantization::spherical::PreScale::None,
    };

    let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
        data.as_view(),
        (input.transform_kind()).into(),
        m.try_into()?,
        pre_scale,
        &mut rand::rngs::StdRng::seed_from_u64(input.seed()),
        GlobalAllocator,
    )?;

    let quantizer_poly = match input.num_bits().get() {
        1 => new_quantizer::<1>(quantizer)?,
        2 => new_quantizer::<2>(quantizer)?,
        4 => new_quantizer::<4>(quantizer)?,
        _ => unreachable!("try_match handles bit validation"),
    };

    let config = input.try_as_config()?.build()?;
    let params = input.bftree_parameters(max_points, data.ncols());
    let start_points = input
        .build()
        .start_point_strategy()
        .compute(data.as_view())?;
    let provider = BfTreeProvider::new(params, start_points.as_view(), quantizer_poly)?;
    let index = Arc::new(DiskANNIndex::new(config, provider, None));

    let num_threads_and_tasks = NonZeroUsize::new(input.build().num_threads()).unwrap();
    let managed_stream = BfTreeSQStream {
        index,
        search: topk.clone(),
        runtime: benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
        ntasks: num_threads_and_tasks,
        inplace_delete_num_to_replace: input.runbook_params().ip_delete_num_to_replace,
        inplace_delete_method: input.runbook_params().ip_delete_method.into(),
    };

    let num_start_points = input.build().start_point_strategy().count();
    let managed = Managed::new(
        max_points + num_start_points,
        managed::SlotReclaim::Immediate,
        managed_stream,
    );

    let max_k = topk.max_k();
    let layered = bigann::WithData::new(managed, data, queries, move |path| {
        Ok(Box::new(datafiles::load_groundtruth(
            datafiles::BinFile(path),
            Some(max_k),
        )?))
    });

    Ok(layered)
}
