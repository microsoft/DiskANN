/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    const NAME: &str = "graph-index-spherical-quantization";

    #[cfg(feature = "spherical-quantization")]
    {
        use crate::index::search::plugins;

        // NOTE: Since the spherical provider is not generic on the number of bits, the
        // implementations of the search-plugins are shared by all bit-widths. Registering
        // all plugins for all bit widths does not meaningfully increase compilation time.
        registry.register(
            NAME,
            imp::SphericalQ::<1>::new()
                .search(plugins::Topk)
                .search(plugins::Range)
                .search(plugins::TopkBetaFilter)
                .search(plugins::TopkMultihopFilter)
                .search(plugins::TopkInlineFilter),
        )?;

        registry.register(
            NAME,
            imp::SphericalQ::<2>::new()
                .search(plugins::Topk)
                .search(plugins::Range)
                .search(plugins::TopkBetaFilter)
                .search(plugins::TopkMultihopFilter)
                .search(plugins::TopkInlineFilter),
        )?;

        registry.register(
            NAME,
            imp::SphericalQ::<4>::new()
                .search(plugins::Topk)
                .search(plugins::Range)
                .search(plugins::TopkBetaFilter)
                .search(plugins::TopkMultihopFilter)
                .search(plugins::TopkInlineFilter),
        )?;
    }

    #[cfg(not(feature = "spherical-quantization"))]
    registry.register_partially_gated::<crate::inputs::graph_index::SphericalQuantBuild>(
        NAME,
        diskann_benchmark_runner::Features::new("spherical-quantization"),
        "Spherical quantized (RabitQ) graph build and search",
    )?;

    Ok(())
}

////////////////
// SphericalQ //
////////////////

#[cfg(feature = "spherical-quantization")]
mod imp {
    use diskann::graph::{DiskANNIndex, StartPointStrategy};
    use diskann_benchmark_core as benchmark_core;
    use diskann_benchmark_core::recall::GroundTruthMode;
    use diskann_benchmark_runner::{
        benchmark::{MatchContext, Score},
        utils::{datatype::AsDataType, MicroSeconds},
        Benchmark, Checkpoint, Output,
    };
    use diskann_providers::{
        index::diskann_async,
        model::graph::provider::async_::{common, inmem},
    };
    use diskann_quantization::alloc::GlobalAllocator;
    use diskann_utils::Matrix;
    use rand::SeedableRng;
    use serde::Serialize;
    use std::{io::Write, sync::Arc};

    use crate::{
        index::{
            benchmarks::QueryType,
            build::{self, only_single_insert, BuildStats},
            result::AggregatedSearchResults,
            search,
        },
        inputs::{
            exhaustive,
            graph_index::{SearchPhase, SearchPhaseKind, SphericalQuantBuild},
        },
        utils::{
            self, datafiles,
            filters::{generate_bitmaps, setup_filter_strategies},
        },
    };

    type SQProvider = inmem::DefaultProvider<
        inmem::FullPrecisionStore<f32>,
        inmem::spherical::SphericalStore,
        common::NoDeletes,
        diskann::provider::DefaultContext,
    >;

    impl QueryType for SQProvider {
        type Element = f32;
    }

    /// A [`Benchmark`] for spherical-quantized searches containing a dynamic list of search
    /// types.
    pub(crate) struct SphericalQ<const NBITS: usize> {
        search: search::plugins::Plugins<SQProvider, SearchPhase, exhaustive::SphericalQuery>,
    }

    impl<const NBITS: usize> SphericalQ<NBITS> {
        pub(crate) fn new() -> Self {
            Self {
                search: search::plugins::Plugins::new(),
            }
        }

        pub(crate) fn search<P>(mut self, plugin: P) -> Self
        where
            P: search::plugins::Plugin<SQProvider, SearchPhase, exhaustive::SphericalQuery>
                + 'static,
        {
            self.search.register(plugin);
            self
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
            impl Benchmark for SphericalQ<$N> {
                type Input = SphericalQuantBuild;
                type Output = SphericalBuildResult;

                fn try_match(&self, input: &SphericalQuantBuild, context: &MatchContext) -> Score {
                    let mut score = context.success(0);

                    if input.build.multi_insert().is_some() {
                        score.fail(1, &"Spherical Quantization does not support multi-insert");
                    }

                    if !f32::is_match(input.build.data_type()) {
                        score.fail(
                            1,
                            &format_args!(
                                "Only `float32` data type is supported. Instead, got {}",
                                input.build.data_type()
                            ),
                        );
                    }

                    if !self.search.is_match(&input.search_phase) {
                        score.fail(
                            1,
                            &format_args!(
                                "Unsupported search phase: \"{}\" - expected one of {}",
                                input.search_phase.kind(),
                                self.search.format_kinds(),
                            ),
                        )
                    }

                    let num_bits = input.num_bits.get();
                    if num_bits != $N {
                        let penalty = ($N as usize)
                            .abs_diff(num_bits)
                            .try_into()
                            .unwrap_or(u32::MAX);

                        score.fail(
                            penalty,
                            &format_args!("Expected {} bits, got {}", $N, num_bits),
                        );
                    }

                    score
                }

                fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    writeln!(
                        f,
                        "- Index Build and Search using {}-bit spherical quantization",
                        $N
                    )?;
                    writeln!(f, "- Requires `float32` data")?;
                    writeln!(f, "- Implements `squared_l2` or `inner_product` distance")?;
                    writeln!(f, "- Does not support multi-insert")?;
                    writeln!(f, "- Search Kinds: {}", self.search.format_kinds())?;
                    Ok(())
                }

                fn run(
                    &self,
                    input: &SphericalQuantBuild,
                    checkpoint: Checkpoint<'_>,
                    mut output: &mut dyn Output,
                ) -> anyhow::Result<SphericalBuildResult> {
                    assert_eq!(
                        input.num_bits.get(),
                        $N,
                        "INTERNAL ERROR: this should not have passed the match check"
                    );

                    writeln!(output, "{}", input)?;

                    let build = &input.build;

                    let data: Arc<Matrix<f32>> =
                        Arc::new(datafiles::load_dataset(datafiles::BinFile(build.data()))?);

                    let start = std::time::Instant::now();
                    let m: diskann_vector::distance::Metric = build.distance().into();
                    let pre_scale = match input.pre_scale {
                        Some(v) => v.try_into()?,
                        None => diskann_quantization::spherical::PreScale::None,
                    };

                    let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
                        data.as_view(),
                        (&input.transform_kind).into(),
                        m.try_into()?,
                        pre_scale,
                        &mut rand::rngs::StdRng::seed_from_u64(input.seed),
                        GlobalAllocator,
                    )?;

                    let training_time: MicroSeconds = start.elapsed().into();

                    // We manually inline the build and search loops because we support
                    // multiple different kinds of searches.
                    let index = diskann_async::new_quant_index::<f32, _, _>(
                        input.try_as_config()?.build()?,
                        input.inmem_parameters(data.nrows(), data.ncols()),
                        diskann_quantization::spherical::iface::Impl::<$N>::new(quantizer)?,
                        common::NoDeletes,
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

                    // Save construction stats before running queries.
                    checkpoint.checkpoint(&result)?;

                    for layout in input.query_layouts.iter() {
                        let search = self
                            .search
                            .run(index.clone(), &input.search_phase, layout)?;

                        result.append(SearchRun {
                            layout: *layout,
                            results: search,
                        });
                    }

                    writeln!(output, "\n\n{}", result)?;
                    Ok(result)
                }
            }
        };
    }

    build_and_search!(1);
    build_and_search!(2);
    build_and_search!(4);

    impl search::plugins::Plugin<SQProvider, SearchPhase, exhaustive::SphericalQuery>
        for search::plugins::Topk
    {
        fn is_match(&self, phase: &SearchPhase) -> bool {
            search::plugins::Topk::is_match(phase)
        }

        fn kind(&self) -> &'static str {
            SearchPhaseKind::Topk.as_str()
        }

        fn run(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            phase: &SearchPhase,
            query_layout: &exhaustive::SphericalQuery,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let topk = phase.as_topk()?;

            // compute the maximum value of k used in any search
            let max_k = topk.max_k();

            let queries: Arc<Matrix<f32>> =
                Arc::new(datafiles::load_dataset(datafiles::BinFile(&topk.queries))?);

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&topk.groundtruth), Some(max_k))?;

            let steps = search::knn::SearchSteps::new(
                topk.reps,
                &topk.num_threads,
                &topk.runs,
                GroundTruthMode::Fixed,
            );

            let knn = benchmark_core::search::graph::KNN::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::broadcast(
                    inmem::spherical::Quantized::search((*query_layout).into()),
                ),
            )?;

            let result = search::knn::run(&knn, &groundtruth, steps)?;
            Ok(AggregatedSearchResults::Topk(result))
        }
    }

    impl search::plugins::Plugin<SQProvider, SearchPhase, exhaustive::SphericalQuery>
        for search::plugins::Range
    {
        fn is_match(&self, phase: &SearchPhase) -> bool {
            search::plugins::Range::is_match(phase)
        }

        fn kind(&self) -> &'static str {
            SearchPhaseKind::Range.as_str()
        }

        fn run(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            phase: &SearchPhase,
            query_layout: &exhaustive::SphericalQuery,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let range = phase.as_range()?;

            let queries: Arc<Matrix<f32>> =
                Arc::new(datafiles::load_dataset(datafiles::BinFile(&range.queries))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&range.groundtruth))?;

            let steps =
                search::range::RangeSearchSteps::new(range.reps, &range.num_threads, &range.runs);

            let range = benchmark_core::search::graph::Range::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::broadcast(
                    inmem::spherical::Quantized::search((*query_layout).into()),
                ),
            )?;

            let result = search::range::run(&range, &groundtruth, steps)?;

            Ok(AggregatedSearchResults::Range(result))
        }
    }

    impl search::plugins::Plugin<SQProvider, SearchPhase, exhaustive::SphericalQuery>
        for search::plugins::TopkBetaFilter
    {
        fn is_match(&self, phase: &SearchPhase) -> bool {
            search::plugins::TopkBetaFilter::is_match(phase)
        }

        fn kind(&self) -> &'static str {
            SearchPhaseKind::TopkBetaFilter.as_str()
        }

        fn run(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            phase: &SearchPhase,
            query_layout: &exhaustive::SphericalQuery,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let betafilter = phase.as_topk_beta_filter()?;

            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &betafilter.queries,
            ))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&betafilter.groundtruth))?;

            let steps = search::knn::SearchSteps::new(
                betafilter.reps,
                &betafilter.num_threads,
                &betafilter.runs,
                GroundTruthMode::Flexible,
            );

            let bit_maps = generate_bitmaps(&betafilter.query_predicates, &betafilter.data_labels)?;

            let label_providers: Vec<_> = bit_maps
                .into_iter()
                .map(utils::filters::as_query_label_provider)
                .collect();

            let strategy = inmem::spherical::Quantized::search((*query_layout).into());
            let search_strategies =
                setup_filter_strategies(betafilter.beta, label_providers.iter().cloned(), strategy);

            let knn = benchmark_core::search::graph::KNN::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::Collection(search_strategies.into()),
            )?;

            let result = search::knn::run(&knn, &groundtruth, steps)?;
            Ok(AggregatedSearchResults::Topk(result))
        }
    }

    impl search::plugins::Plugin<SQProvider, SearchPhase, exhaustive::SphericalQuery>
        for search::plugins::TopkMultihopFilter
    {
        fn is_match(&self, phase: &SearchPhase) -> bool {
            search::plugins::TopkMultihopFilter::is_match(phase)
        }

        fn kind(&self) -> &'static str {
            SearchPhaseKind::TopkMultihopFilter.as_str()
        }

        fn run(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            phase: &SearchPhase,
            query_layout: &exhaustive::SphericalQuery,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let multihop = phase.as_topk_multihop_filter()?;

            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &multihop.queries,
            ))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

            let steps = search::knn::SearchSteps::new(
                multihop.reps,
                &multihop.num_threads,
                &multihop.runs,
                GroundTruthMode::Flexible,
            );

            let bit_maps = generate_bitmaps(&multihop.query_predicates, &multihop.data_labels)?;

            let bit_map_filters: Arc<[_]> = bit_maps
                .into_iter()
                .map(utils::filters::as_query_label_provider)
                .collect();

            let multihop = benchmark_core::search::graph::MultiHop::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::broadcast(
                    inmem::spherical::Quantized::search((*query_layout).into()),
                ),
                bit_map_filters.clone(),
            )?;

            let result = search::knn::run(&multihop, &groundtruth, steps)?;
            Ok(AggregatedSearchResults::Topk(result))
        }
    }

    impl search::plugins::Plugin<SQProvider, SearchPhase, exhaustive::SphericalQuery>
        for search::plugins::TopkInlineFilter
    {
        fn is_match(&self, phase: &SearchPhase) -> bool {
            search::plugins::TopkInlineFilter::is_match(phase)
        }

        fn kind(&self) -> &'static str {
            search::plugins::TopkInlineFilter::as_str()
        }

        fn run(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            phase: &SearchPhase,
            query_layout: &exhaustive::SphericalQuery,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let inline = phase.as_topk_inline_filter()?;

            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &inline.queries,
            ))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&inline.groundtruth))?;

            let steps = search::knn::SearchSteps::new(
                inline.reps,
                &inline.num_threads,
                &inline.runs,
                GroundTruthMode::Flexible,
            );

            let bit_maps = generate_bitmaps(&inline.query_predicates, &inline.data_labels)?;

            let bit_map_filters: Arc<[_]> = bit_maps
                .into_iter()
                .map(utils::filters::as_query_label_provider)
                .collect();

            let inline = benchmark_core::search::graph::InlineFilterSearch::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::broadcast(
                    inmem::spherical::Quantized::search((*query_layout).into()),
                ),
                bit_map_filters.clone(),
                inline.adaptive_l()?,
            )?;

            let result = search::knn::run(&inline, &groundtruth, steps)?;
            Ok(AggregatedSearchResults::Topk(result))
        }
    }
}
