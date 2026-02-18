/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

// Create a stub-module if the "spherical-quantization" feature is disabled.
crate::utils::stub_impl!(
    "spherical-quantization",
    inputs::async_::SphericalQuantBuild
);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    const NAME: &str = "async-spherical-quantization";

    // Spherical - requires feature "spherical-quantization"
    #[cfg(feature = "spherical-quantization")]
    benchmarks.register::<imp::SphericalQ<'static, 1>>(NAME, |object, checkpoint, output| {
        use crate::backend::index::benchmarks::BuildAndSearch;

        match object.run(checkpoint, output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });

    #[cfg(feature = "spherical-quantization")]
    benchmarks.register::<imp::SphericalQ<'static, 2>>(NAME, |object, checkpoint, output| {
        use crate::backend::index::benchmarks::BuildAndSearch;

        match object.run(checkpoint, output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });

    #[cfg(feature = "spherical-quantization")]
    benchmarks.register::<imp::SphericalQ<'static, 4>>(NAME, |object, checkpoint, output| {
        use crate::backend::index::benchmarks::BuildAndSearch;

        match object.run(checkpoint, output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });

    // Stub implementation
    #[cfg(not(feature = "spherical-quantization"))]
    imp::register(NAME, benchmarks)
}

////////////////
// SphericalQ //
////////////////

#[cfg(feature = "spherical-quantization")]
mod imp {
    use diskann::graph::StartPointStrategy;
    use diskann_benchmark_core as benchmark_core;
    use diskann_benchmark_runner::{
        describeln,
        dispatcher::{self, DispatchRule, FailureScore, MatchScore},
        utils::{datatype, MicroSeconds},
        Any, Checkpoint, Output,
    };
    use diskann_providers::{
        index::diskann_async::{self},
        model::graph::provider::async_::{common::NoDeletes, inmem},
    };
    use diskann_quantization::alloc::GlobalAllocator;
    use diskann_utils::views::Matrix;
    use rand::SeedableRng;
    use serde::Serialize;
    use std::{io::Write, sync::Arc};

    use crate::{
        backend::index::{
            benchmarks::BuildAndSearch,
            build::{self, only_single_insert, BuildStats},
            result::AggregatedSearchResults,
            search,
        },
        inputs::{
            async_::{SearchPhase, SphericalQuantBuild},
            exhaustive,
        },
        utils::{
            self, datafiles,
            filters::{generate_bitmaps, setup_filter_strategies},
        },
    };

    /// The dispatcher target for `spherical-quantization` operations.
    pub(super) struct SphericalQ<'a, const NBITS: usize> {
        input: &'a SphericalQuantBuild,
    }

    impl<'a, const NBITS: usize> SphericalQ<'a, NBITS> {
        pub(super) fn new(input: &'a SphericalQuantBuild) -> Self {
            Self { input }
        }
    }

    impl<const NBITS: usize> dispatcher::Map for SphericalQ<'static, NBITS> {
        type Type<'a> = SphericalQ<'a, NBITS>;
    }

    impl<'a, const NBITS: usize> DispatchRule<&'a SphericalQuantBuild> for SphericalQ<'a, NBITS> {
        type Error = std::convert::Infallible;

        fn try_match(from: &&'a SphericalQuantBuild) -> Result<MatchScore, FailureScore> {
            // If this is multi-insert, return a very-close failure.
            let mut failure_score: Option<u32> = None;
            if from.build.multi_insert.is_some() {
                failure_score = Some(1);
            }

            // Ensure the data type is compatible (float32).
            if let Err(FailureScore(_)) = datatype::Type::<f32>::try_match(&from.build.data_type) {
                *failure_score.get_or_insert(0) += 1;
            }

            // Match the number of bits.
            let num_bits = from.num_bits.get();
            if num_bits != NBITS {
                *failure_score.get_or_insert(0) +=
                    NBITS.abs_diff(num_bits).try_into().unwrap_or(u32::MAX);
            }

            match failure_score {
                None => Ok(MatchScore(0)),
                Some(score) => Err(FailureScore(score)),
            }
        }

        fn convert(from: &'a SphericalQuantBuild) -> Result<Self, Self::Error> {
            assert_eq!(from.num_bits.get(), NBITS);
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a SphericalQuantBuild>,
        ) -> std::fmt::Result {
            match from {
                None => {
                    describeln!(
                        f,
                        "- Index Build and Search using {}-bit spherical quantization",
                        NBITS
                    )?;
                    describeln!(f, "- Requires `float32` data")?;
                    describeln!(f, "- Implements `squared_l2` or `inner_product` distance",)?;
                    describeln!(f, "- Does not support multi-insert")?;
                }
                Some(input) => {
                    let num_bits = input.num_bits.get();
                    if num_bits != NBITS {
                        describeln!(f, "- Expected {} bits, got {}", NBITS, num_bits)?;
                    }

                    if input.build.multi_insert.is_some() {
                        describeln!(f, "- Spherical Quantization does not support multi-insert")?;
                    }

                    if datatype::Type::<f32>::try_match(&input.build.data_type).is_err() {
                        describeln!(
                            f,
                            "- Only `float32` data type is supported. Instead, got {}",
                            input.build.data_type
                        )?;
                    }
                }
            }
            Ok(())
        }
    }

    impl<'a, const NBITS: usize> DispatchRule<&'a Any> for SphericalQ<'a, NBITS> {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<SphericalQuantBuild, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<SphericalQuantBuild, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<SphericalQuantBuild, Self>(f, from, SphericalQuantBuild::tag())
        }
    }

    macro_rules! write_field {
        ($f:ident, $field:tt, $fmt:literal, $($expr:tt)*) => {
            writeln!($f, concat!("{:>12}: ", $fmt), $field, $($expr)*)
        }
    }

    #[derive(Debug, Serialize)]
    struct SearchRun {
        layout: exhaustive::SphericalQuery,
        results: AggregatedSearchResults,
    }

    #[derive(Debug, Serialize)]
    pub struct SphericalBuildResult {
        training_time: MicroSeconds,
        quantized_dim: usize,
        quantized_bytes: usize,
        original_dim: usize,
        build: BuildStats,
        runs: Vec<SearchRun>,
    }

    impl SphericalBuildResult {
        fn append(&mut self, run: SearchRun) {
            self.runs.push(run)
        }
    }

    impl std::fmt::Display for SphericalBuildResult {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write_field!(f, "Training Time", "{}s", self.training_time.as_seconds())?;
            write_field!(f, "Quantized Dim", "{}", self.quantized_dim)?;
            write_field!(f, "Quantized Bytes", "{}", self.quantized_bytes)?;
            write_field!(f, "Original Dim", "{}\n\n", self.original_dim)?;

            writeln!(f, "{}", self.build)?;

            for (i, v) in self.runs.iter().enumerate() {
                write_field!(f, "Run", "{} of {}", i + 1, self.runs.len())?;
                write_field!(f, "Query Layout", "{}", v.layout)?;
                v.results.fmt(f)?;
            }
            Ok(())
        }
    }

    macro_rules! build_and_search {
        ($N:literal) => {
            impl<'a> BuildAndSearch<'a> for SphericalQ<'a, $N> {
                type Data = SphericalBuildResult;
                fn run(
                    self,
                    _checkpoint: Checkpoint<'_>,
                    mut output: &mut dyn Output,
                ) -> Result<Self::Data, anyhow::Error> {
                    writeln!(output, "{}", self.input)?;

                    let build = &self.input.build;

                    let data: Arc<Matrix<f32>> =
                        Arc::new(datafiles::load_dataset(datafiles::BinFile(&build.data))?);

                    let start = std::time::Instant::now();
                    let m: diskann_vector::distance::Metric = build.distance.into();
                    let pre_scale = match self.input.pre_scale {
                        Some(v) => v.try_into()?,
                        None => diskann_quantization::spherical::PreScale::None,
                    };

                    let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
                        data.as_view(),
                        (&self.input.transform_kind).into(),
                        m.try_into()?,
                        pre_scale,
                        &mut rand::rngs::StdRng::seed_from_u64(self.input.seed),
                        GlobalAllocator,
                    )?;

                    let training_time: MicroSeconds = start.elapsed().into();

                    // We manually inline the build and search loops because we support
                    // multiple different kinds of searches.
                    let index = diskann_async::new_quant_index::<f32, _, _>(
                        self.input.try_as_config()?.build()?,
                        self.input.inmem_parameters(data.nrows(), data.ncols()),
                        diskann_quantization::spherical::iface::Impl::<$N>::new(quantizer)?,
                        NoDeletes,
                    )?;

                    build::set_start_points(
                        index.provider(),
                        data.as_view(),
                        StartPointStrategy::Medoid,
                    )?;

                    let original_dim = data.ncols();
                    let build_stats = only_single_insert(
                        index.clone(),
                        inmem::spherical::Quantized::build(),
                        data.clone(),
                        &build,
                        output,
                    )?;

                    let mut result = SphericalBuildResult {
                        training_time,
                        quantized_dim: index.provider().aux_vectors.output_dim(),
                        quantized_bytes: index.provider().aux_vectors.bytes(),
                        original_dim,
                        build: build_stats,
                        runs: Vec::new(),
                    };

                    match &self.input.search_phase {
                        SearchPhase::Topk(search_phase) => {
                            // Handle Topk search phase

                            // Save construction stats before running queries.
                            _checkpoint.checkpoint(&result)?;

                            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(
                                datafiles::BinFile(&search_phase.queries),
                            )?);

                            let groundtruth = datafiles::load_groundtruth(datafiles::BinFile(
                                &search_phase.groundtruth,
                            ))?;

                            let steps = search::knn::SearchSteps::new(
                                search_phase.reps,
                                &search_phase.num_threads,
                                &search_phase.runs,
                            );

                            for &layout in self.input.query_layouts.iter() {
                                let knn = benchmark_core::search::graph::KNN::new(
                                    index.clone(),
                                    queries.clone(),
                                    benchmark_core::search::graph::Strategy::broadcast(
                                        inmem::spherical::Quantized::search(layout.into()),
                                    ),
                                )?;

                                let search_results = search::knn::run(&knn, &groundtruth, steps)?;
                                result.append(SearchRun {
                                    layout,
                                    results: AggregatedSearchResults::Topk(search_results),
                                });
                            }
                            writeln!(output, "\n\n{}", result)?;
                            Ok(result)
                        }
                        SearchPhase::Range(search_phase) => {
                            // Handle Range search phase

                            // Save construction stats before running queries.
                            _checkpoint.checkpoint(&result)?;

                            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(
                                datafiles::BinFile(&search_phase.queries),
                            )?);

                            let groundtruth = datafiles::load_range_groundtruth(
                                datafiles::BinFile(&search_phase.groundtruth),
                            )?;

                            let steps = search::range::RangeSearchSteps::new(
                                search_phase.reps,
                                &search_phase.num_threads,
                                &search_phase.runs,
                            );

                            for &layout in self.input.query_layouts.iter() {
                                let range = benchmark_core::search::graph::Range::new(
                                    index.clone(),
                                    queries.clone(),
                                    benchmark_core::search::graph::Strategy::broadcast(
                                        inmem::spherical::Quantized::search(layout.into()),
                                    ),
                                )?;

                                let search_results =
                                    search::range::run(&range, &groundtruth, steps)?;

                                result.append(SearchRun {
                                    layout,
                                    results: AggregatedSearchResults::Range(search_results),
                                });
                            }

                            writeln!(output, "\n\n{}", result)?;
                            Ok(result)
                        }
                        SearchPhase::TopkBetaFilter(search_phase) => {
                            // Handle Beta Filtered Topk search phase

                            // Save construction stats before running queries.
                            _checkpoint.checkpoint(&result)?;

                            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(
                                datafiles::BinFile(&search_phase.queries),
                            )?);

                            let groundtruth = datafiles::load_range_groundtruth(
                                datafiles::BinFile(&search_phase.groundtruth),
                            )?;

                            let steps = search::knn::SearchSteps::new(
                                search_phase.reps,
                                &search_phase.num_threads,
                                &search_phase.runs,
                            );

                            let bit_maps = generate_bitmaps(
                                &search_phase.query_predicates,
                                &search_phase.data_labels,
                            )?;

                            let label_providers: Vec<_> = bit_maps
                                .into_iter()
                                .map(utils::filters::as_query_label_provider)
                                .collect();

                            for &layout in self.input.query_layouts.iter() {
                                let strategy = inmem::spherical::Quantized::search(layout.into());
                                let search_strategies = setup_filter_strategies(
                                    search_phase.beta,
                                    label_providers.iter().cloned(),
                                    strategy.clone(),
                                );

                                let knn = benchmark_core::search::graph::KNN::new(
                                    index.clone(),
                                    queries.clone(),
                                    benchmark_core::search::graph::Strategy::Collection(
                                        search_strategies.into(),
                                    ),
                                )?;

                                let search_results = search::knn::run(&knn, &groundtruth, steps)?;

                                result.append(SearchRun {
                                    layout,
                                    results: AggregatedSearchResults::Topk(search_results),
                                });
                            }
                            writeln!(output, "\n\n{}", result)?;
                            Ok(result)
                        }
                        SearchPhase::TopkMultihopFilter(search_phase) => {
                            // Handle Beta Filtered Topk search phase

                            // Save construction stats before running queries.
                            _checkpoint.checkpoint(&result)?;

                            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(
                                datafiles::BinFile(&search_phase.queries),
                            )?);

                            let groundtruth = datafiles::load_groundtruth(datafiles::BinFile(
                                &search_phase.groundtruth,
                            ))?;

                            let steps = search::knn::SearchSteps::new(
                                search_phase.reps,
                                &search_phase.num_threads,
                                &search_phase.runs,
                            );

                            let bit_maps = generate_bitmaps(
                                &search_phase.query_predicates,
                                &search_phase.data_labels,
                            )?;

                            let bit_map_filters: Arc<[_]> = bit_maps
                                .into_iter()
                                .map(utils::filters::as_query_label_provider)
                                .collect();

                            for &layout in self.input.query_layouts.iter() {
                                let multihop = benchmark_core::search::graph::MultiHop::new(
                                    index.clone(),
                                    queries.clone(),
                                    benchmark_core::search::graph::Strategy::broadcast(
                                        inmem::spherical::Quantized::search(layout.into()),
                                    ),
                                    bit_map_filters.clone(),
                                )?;

                                let search_results =
                                    search::knn::run(&multihop, &groundtruth, steps)?;
                                result.append(SearchRun {
                                    layout,
                                    results: AggregatedSearchResults::Topk(search_results),
                                });
                            }
                            writeln!(output, "\n\n{}", result)?;
                            Ok(result)
                        }
                    }
                }
            }
        };
    }

    build_and_search!(1);
    build_and_search!(2);
    build_and_search!(4);
}
