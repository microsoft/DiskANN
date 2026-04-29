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

    #[cfg(feature = "spherical-quantization")]
    {
        use crate::backend::index::search::plugins;

        // NOTE: Since the spherical provider is not generic on the number of bits, the
        // implementations of the search-plugins are shared by all bit-widths. Registering
        // all plugins for all bit widths does not meaningfully increase compilation time.
        benchmarks.register(
            NAME,
            imp::SphericalQ::<1>::new()
                .search(plugins::Topk)
                .search(plugins::Range)
                .search(plugins::BetaFilter)
                .search(plugins::MultihopFilter),
        );

        benchmarks.register(
            NAME,
            imp::SphericalQ::<2>::new()
                .search(plugins::Topk)
                .search(plugins::Range)
                .search(plugins::BetaFilter)
                .search(plugins::MultihopFilter),
        );

        benchmarks.register(
            NAME,
            imp::SphericalQ::<4>::new()
                .search(plugins::Topk)
                .search(plugins::Range)
                .search(plugins::BetaFilter)
                .search(plugins::MultihopFilter),
        );
    }

    // Stub implementation
    #[cfg(not(feature = "spherical-quantization"))]
    imp::register(NAME, benchmarks)
}

////////////////
// SphericalQ //
////////////////

#[cfg(feature = "spherical-quantization")]
mod imp {
    use diskann::graph::{DiskANNIndex, StartPointStrategy};
    use diskann_benchmark_core as benchmark_core;
    use diskann_benchmark_runner::{
        dispatcher::{DispatchRule, FailureScore, MatchScore},
        utils::{datatype, MicroSeconds},
        Benchmark, Checkpoint, Output,
    };
    use diskann_providers::{
        index::diskann_async,
        model::graph::provider::async_::{common, inmem},
    };
    use diskann_quantization::alloc::GlobalAllocator;
    use diskann_utils::views::Matrix;
    use rand::SeedableRng;
    use serde::Serialize;
    use std::{io::Write, sync::Arc};

    use crate::{
        backend::index::{
            benchmarks::QueryType,
            build::{self, only_single_insert, BuildStats},
            result::AggregatedSearchResults,
            search,
        },
        inputs::{
            async_::{SearchPhase, SearchPhaseKind, SphericalQuantBuild},
            exhaustive,
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
    pub(super) struct SphericalQ<const NBITS: usize> {
        search: search::plugins::Plugins<SQProvider, exhaustive::SphericalQuery>,
    }

    impl<const NBITS: usize> SphericalQ<NBITS> {
        pub(super) fn new() -> Self {
            Self {
                search: search::plugins::Plugins::new(),
            }
        }

        pub(super) fn search<P>(mut self, plugin: P) -> Self
        where
            P: search::plugins::Plugin<SQProvider, exhaustive::SphericalQuery> + 'static,
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

                fn try_match(
                    &self,
                    input: &SphericalQuantBuild,
                ) -> Result<MatchScore, FailureScore> {
                    let mut failure_score: Option<u32> = None;
                    if input.build.multi_insert.is_some() {
                        failure_score = Some(1);
                    }

                    if let Err(FailureScore(_)) =
                        datatype::Type::<f32>::try_match(&input.build.data_type)
                    {
                        *failure_score.get_or_insert(0) += 1;
                    }

                    if !self.search.is_match(input.search_phase.kind()) {
                        *failure_score.get_or_insert(0) += 1;
                    }

                    let num_bits = input.num_bits.get();
                    if num_bits != $N {
                        *failure_score.get_or_insert(0) += ($N as usize)
                            .abs_diff(num_bits)
                            .try_into()
                            .unwrap_or(u32::MAX);
                    }

                    match failure_score {
                        None => Ok(MatchScore(0)),
                        Some(score) => Err(FailureScore(score)),
                    }
                }

                fn description(
                    &self,
                    f: &mut std::fmt::Formatter<'_>,
                    input: Option<&SphericalQuantBuild>,
                ) -> std::fmt::Result {
                    match input {
                        None => {
                            writeln!(
                                f,
                                "- Index Build and Search using {}-bit spherical quantization",
                                $N
                            )?;
                            writeln!(f, "- Requires `float32` data")?;
                            writeln!(f, "- Implements `squared_l2` or `inner_product` distance",)?;
                            writeln!(f, "- Does not support multi-insert")?;
                            writeln!(f, "- Search Kinds: {}", self.search.format_kinds())?;
                        }
                        Some(input) => {
                            let num_bits = input.num_bits.get();
                            if num_bits != $N {
                                writeln!(f, "- Expected {} bits, got {}", $N, num_bits)?;
                            }

                            if input.build.multi_insert.is_some() {
                                writeln!(
                                    f,
                                    "- Spherical Quantization does not support multi-insert"
                                )?;
                            }

                            if datatype::Type::<f32>::try_match(&input.build.data_type).is_err() {
                                writeln!(
                                    f,
                                    "- Only `float32` data type is supported. Instead, got {}",
                                    input.build.data_type
                                )?;
                            }

                            if !self.search.is_match(input.search_phase.kind()) {
                                writeln!(
                                    f,
                                    "- Unsupported search phase: \"{}\" - expected one of {}",
                                    input.search_phase.kind(),
                                    self.search.format_kinds(),
                                )?;
                            }
                        }
                    }
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
                        Arc::new(datafiles::load_dataset(datafiles::BinFile(&build.data))?);

                    let start = std::time::Instant::now();
                    let m: diskann_vector::distance::Metric = build.distance.into();
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
                            .run(index.clone(), layout, &input.search_phase)?;
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

    impl search::plugins::Plugin<SQProvider, exhaustive::SphericalQuery> for search::plugins::Topk {
        fn kind(&self) -> SearchPhaseKind {
            Self::kind()
        }

        fn search(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            query_layout: &exhaustive::SphericalQuery,
            phase: &SearchPhase,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let topk = phase.as_topk()?;

            let queries: Arc<Matrix<f32>> =
                Arc::new(datafiles::load_dataset(datafiles::BinFile(&topk.queries))?);

            let groundtruth = datafiles::load_groundtruth(datafiles::BinFile(&topk.groundtruth))?;

            let steps = search::knn::SearchSteps::new(topk.reps, &topk.num_threads, &topk.runs);

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

    impl search::plugins::Plugin<SQProvider, exhaustive::SphericalQuery> for search::plugins::Range {
        fn kind(&self) -> SearchPhaseKind {
            Self::kind()
        }

        fn search(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            query_layout: &exhaustive::SphericalQuery,
            phase: &SearchPhase,
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

    impl search::plugins::Plugin<SQProvider, exhaustive::SphericalQuery>
        for search::plugins::BetaFilter
    {
        fn kind(&self) -> SearchPhaseKind {
            Self::kind()
        }

        fn search(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            query_layout: &exhaustive::SphericalQuery,
            phase: &SearchPhase,
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

    impl search::plugins::Plugin<SQProvider, exhaustive::SphericalQuery>
        for search::plugins::MultihopFilter
    {
        fn kind(&self) -> SearchPhaseKind {
            Self::kind()
        }

        fn search(
            &self,
            index: Arc<DiskANNIndex<SQProvider>>,
            query_layout: &exhaustive::SphericalQuery,
            phase: &SearchPhase,
        ) -> anyhow::Result<AggregatedSearchResults> {
            let multihop = phase.as_topk_multihop_filter()?;

            let queries: Arc<Matrix<f32>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &multihop.queries,
            ))?);

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

            let steps =
                search::knn::SearchSteps::new(multihop.reps, &multihop.num_threads, &multihop.runs);

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
}
