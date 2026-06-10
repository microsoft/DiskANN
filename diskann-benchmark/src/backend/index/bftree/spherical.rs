/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use diskann::graph::DiskANNIndex;
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{quant::QuantVectorProvider, BfTreeProvider};
use diskann_providers::model::graph::provider::async_::common::Quantized;
use diskann_quantization::{
    alloc::{AllocatorError, GlobalAllocator, Poly},
    spherical::{
        iface::{self as spherical_iface, Quantizer},
        SphericalQuantizer,
    },
};
use diskann_utils::views::Matrix;
use rand::SeedableRng;

use crate::{
    backend::index::{
        benchmarks::{QueryType, Strategy},
        build::single_or_multi_insert,
        result::BuildResult,
        search::plugins::{Plugin, Plugins},
    },
    inputs::{bftree::BfTreeSphericalBuild, graph_index::SearchPhase},
    utils::{self, datafiles},
};

type BfTreeSQProvider = BfTreeProvider<f32, QuantVectorProvider>;

impl QueryType for BfTreeSQProvider {
    type Element = f32;
}

/// A [`Benchmark`] for bf_tree spherical-quantized build + search.
///
/// Dispatches `num_bits` at runtime rather than using const generics to reduce
/// monomorphization.
pub(super) struct BfTreeSpherical {
    search: Plugins<BfTreeSQProvider, SearchPhase, Strategy<Quantized>>,
}

impl BfTreeSpherical {
    pub(super) fn new() -> Self {
        Self {
            search: Plugins::new(),
        }
    }

    pub(super) fn search<P>(mut self, plugin: P) -> Self
    where
        P: Plugin<BfTreeSQProvider, SearchPhase, Strategy<Quantized>> + 'static,
    {
        self.search.register(plugin);
        self
    }
}

fn new_quantizer<const NBITS: usize>(
    quantizer: SphericalQuantizer,
) -> Result<Poly<dyn Quantizer>, AllocatorError>
where
    spherical_iface::Impl<NBITS>: spherical_iface::Constructible + Quantizer,
{
    let imp = spherical_iface::Impl::<NBITS>::new(quantizer)?;
    diskann_quantization::poly!(Quantizer, imp, GlobalAllocator)
}

impl Benchmark for BfTreeSpherical {
    type Input = BfTreeSphericalBuild;
    type Output = BuildResult;

    fn try_match(&self, input: &BfTreeSphericalBuild) -> Result<MatchScore, FailureScore> {
        let mut failure_score: Option<u32> = None;

        if let Err(s) = utils::match_data_type::<f32>(input.data_type()) {
            failure_score = Some(s.0);
        }
        if !matches!(input.num_bits().get(), 1 | 2 | 4) {
            *failure_score.get_or_insert(0) += 1;
        }
        if !self.search.is_match(input.search_phase()) {
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
        input: Option<&BfTreeSphericalBuild>,
    ) -> std::fmt::Result {
        match input {
            None => {
                writeln!(
                    f,
                    "- BfTree Index Build and Search using spherical quantization"
                )?;
                writeln!(f, "- Requires `float32` data")?;
                writeln!(f, "- Search Kinds: {}", self.search.format_kinds())?;
            }
            Some(input) => {
                if !f32::is_match(input.data_type()) {
                    writeln!(
                        f,
                        "- Only `float32` data type is supported. Instead, got {}",
                        input.data_type()
                    )?;
                }

                if !self.search.is_match(input.search_phase()) {
                    writeln!(
                        f,
                        "- Unsupported search phase: \"{}\" - expected one of {}",
                        input.search_phase().kind(),
                        self.search.format_kinds(),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn run(
        &self,
        input: &BfTreeSphericalBuild,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<BuildResult> {
        writeln!(output, "{}", input)?;

        let build = input.build();
        let data: Arc<Matrix<f32>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(build.data()))?);

        // 1. Train the spherical quantizer.
        let start = std::time::Instant::now();
        let m: diskann_vector::distance::Metric = build.distance().into();
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

        let training_time = start.elapsed();
        writeln!(output, "Training time: {:.2}s", training_time.as_secs_f64())?;

        // 2. Dispatch on num_bits to create the type-erased quantizer.
        let quantizer_poly = match input.num_bits().get() {
            1 => new_quantizer::<1>(quantizer)?,
            2 => new_quantizer::<2>(quantizer)?,
            4 => new_quantizer::<4>(quantizer)?,
            _ => unreachable!("try_match handles bit validation"),
        };

        // 3. Build the bf_tree provider with quantization.
        let config = input.try_as_config()?.build()?;
        let params = input.bftree_parameters(data.nrows(), data.ncols());
        let start_points = input
            .build()
            .start_point_strategy()
            .compute(data.as_view())?;
        let provider = BfTreeProvider::new(params, start_points.as_view(), quantizer_poly)?;
        let index = Arc::new(DiskANNIndex::new(config, provider, None));

        // 4. Insert all vectors using Quantized strategy.
        let build_stats =
            single_or_multi_insert(index.clone(), Quantized, data.clone(), build, output)?;

        checkpoint.checkpoint(&build_stats)?;

        // 5. Search using Quantized strategy.
        let search_results =
            self.search
                .run(index, input.search_phase(), &Strategy::new(Quantized))?;

        let result = BuildResult::new(Some(build_stats), search_results);
        writeln!(output, "\n\n{}", result)?;
        Ok(result)
    }
}
