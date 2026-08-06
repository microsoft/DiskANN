/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use diskann::graph::DiskANNIndex;
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    output::Output,
    Benchmark, Checkpoint,
};
use diskann_bftree::{quant::QuantVectorProvider, BfTreeProvider};
use diskann_providers::{
    model::graph::provider::async_::common::Quantized,
    storage::{FileStorageProvider, SaveWith},
};
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
    index::{
        benchmarks::{QueryType, Strategy},
        build::single_or_multi_insert,
        result::BuildResult,
        search::plugins::{Plugin, Plugins},
    },
    inputs::{bftree::BfTreeSphericalBuild, graph_index::SearchPhase},
    utils::{self, datafiles, tokio},
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

    fn try_match(&self, input: &BfTreeSphericalBuild, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        utils::match_data_type::<f32>(&mut score, input.data_type());

        if !matches!(input.num_bits().get(), 1 | 2 | 4) {
            score.fail(
                1,
                &format_args!(
                    "Only 1, 2, or 4 bits are supported, instead got \"{}\"",
                    input.num_bits(),
                ),
            );
        }
        if !self.search.is_match(input.search_phase()) {
            score.fail(
                1,
                &format_args!(
                    "Unsupported search phase: \"{}\" - expected one of {}",
                    input.search_phase().kind(),
                    self.search.format_kinds(),
                ),
            );
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "- BfTree Index Build and Search using spherical quantization"
        )?;
        writeln!(f, "- Requires `float32` data")?;
        writeln!(f, "- Search Kinds: {}", self.search.format_kinds())?;
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
        let params = input.bftree_parameters(data.nrows(), data.ncols())?;
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

        // save the index if requested
        if let Some(save_path) = build.save_path() {
            tokio::block_on(
                index
                    .provider()
                    .save_with(&FileStorageProvider, &save_path.to_string()),
            )?;
        }

        // 5. Search using Quantized strategy.
        let search_results =
            self.search
                .run(index, input.search_phase(), &Strategy::new(Quantized))?;

        let result = BuildResult::new(Some(build_stats), search_results);
        writeln!(output, "\n\n{}", result)?;
        Ok(result)
    }
}
