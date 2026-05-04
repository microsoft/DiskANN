/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

// Create a stub-module if the "spherical-quantization" feature is disabled.
crate::utils::stub_impl!(
    "product-quantization",
    inputs::graph_index::IndexPQOperation
);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    #[cfg(feature = "product-quantization")]
    {
        use crate::backend::index::search::plugins;
        use half::f16;

        // NOTE: Try to balance search plugins with the needed functionality.
        //
        // Feel free to add search plugins, but be mindful of the monomorphization cost.

        benchmarks.register(
            "graph-index-pq-f32",
            imp::ProductQuantized::<f32>::new()
                .search(plugins::Topk)
                .search(plugins::Range),
        );
        benchmarks.register(
            "graph-index-pq-f16",
            imp::ProductQuantized::<f16>::new().search(plugins::Topk),
        );
    }

    // Stub implementation
    #[cfg(not(feature = "product-quantization"))]
    imp::register("graph-index-pq", benchmarks);
}

#[cfg(feature = "product-quantization")]
mod imp {
    use std::{io::Write, sync::Arc};

    use diskann::utils::VectorRepr;
    use diskann_providers::{
        index::diskann_async::{self},
        model::{
            graph::provider::async_::{common, inmem},
            IndexConfiguration,
        },
    };
    use diskann_utils::views::{Matrix, MatrixView};

    use diskann_benchmark_runner::{
        dispatcher::{DispatchRule, FailureScore, MatchScore},
        utils::{datatype, MicroSeconds},
        Benchmark, Checkpoint, Output,
    };
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        backend::index::{
            benchmarks::{run_build, QueryType, Strategy},
            build::{self, load_index, save_index, single_or_multi_insert, BuildStats},
            result::{BuildResult, QuantBuildResult},
            search::plugins,
        },
        inputs::graph_index::{IndexPQOperation, IndexSource, SearchPhase},
        utils::{self, datafiles},
    };

    type PQProvider<T> = inmem::DefaultProvider<
        inmem::FullPrecisionStore<T>,
        inmem::DefaultQuant,
        common::NoDeletes,
        diskann::provider::DefaultContext,
    >;

    impl<T> QueryType for PQProvider<T>
    where
        T: VectorRepr,
    {
        type Element = T;
    }

    /// A [`Benchmark`] for product-quantized searches containing a dynamic list of search
    /// types.
    ///
    /// The kinds of quantized and full-precision searches are kept in-sync.
    pub(super) struct ProductQuantized<T>
    where
        T: VectorRepr,
    {
        quant_search: plugins::Plugins<PQProvider<T>, SearchPhase, Strategy<common::Hybrid>>,
        full_search: plugins::Plugins<PQProvider<T>, SearchPhase, Strategy<common::FullPrecision>>,
    }

    impl<T> ProductQuantized<T>
    where
        T: VectorRepr,
    {
        pub(super) fn new() -> Self {
            Self {
                quant_search: plugins::Plugins::new(),
                full_search: plugins::Plugins::new(),
            }
        }

        pub(super) fn search<P>(mut self, plugin: P) -> Self
        where
            P: plugins::Plugin<PQProvider<T>, SearchPhase, Strategy<common::Hybrid>>
                + plugins::Plugin<PQProvider<T>, SearchPhase, Strategy<common::FullPrecision>>
                + Clone
                + 'static,
        {
            self.quant_search.register(plugin.clone());
            self.full_search.register(plugin);
            self
        }
    }

    impl<T> Benchmark for ProductQuantized<T>
    where
        T: VectorRepr
            + diskann_utils::sampling::WithApproximateNorm
            + diskann::graph::SampleableForStart,
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Input = IndexPQOperation;
        type Output = QuantBuildResult;

        fn try_match(&self, input: &IndexPQOperation) -> Result<MatchScore, FailureScore> {
            let score = datatype::Type::<T>::try_match(input.index_operation.source.data_type());

            if self
                .quant_search
                .is_match(&input.index_operation.search_phase)
            {
                score
            } else {
                match score {
                    Ok(_) => Err(FailureScore(0)),
                    Err(score) => Err(score),
                }
            }
        }

        fn description(
            &self,
            f: &mut std::fmt::Formatter<'_>,
            input: Option<&IndexPQOperation>,
        ) -> std::fmt::Result {
            use diskann_benchmark_runner::dispatcher::{Description, Why};

            match input {
                Some(arg) => {
                    let data_type = arg.index_operation.source.data_type();
                    if datatype::Type::<T>::try_match(data_type).is_err() {
                        writeln!(
                            f,
                            "Data/Query Type: {}",
                            Why::<datatype::DataType, datatype::Type<T>>::new(data_type)
                        )?;
                    }

                    if !self
                        .quant_search
                        .is_match(&arg.index_operation.search_phase)
                    {
                        writeln!(
                            f,
                            "Unsupported search phase: \"{}\" - expected one of {}",
                            arg.index_operation.search_phase.kind(),
                            self.quant_search.format_kinds(),
                        )?;
                    }
                    Ok(())
                }
                None => {
                    writeln!(
                        f,
                        "Data/Query Type: {}",
                        Description::<datatype::DataType, datatype::Type<T>>::new()
                    )?;

                    writeln!(f, "Search Kinds: {}", self.quant_search.format_kinds())
                }
            }
        }

        fn run(
            &self,
            input: &IndexPQOperation,
            checkpoint: Checkpoint<'_>,
            mut output: &mut dyn Output,
        ) -> anyhow::Result<QuantBuildResult> {
            writeln!(output, "{}", input)?;

            let hybrid = common::Hybrid::new(input.max_fp_vecs_per_prune);

            let (index, build_stats, quant_training_time) = match &input.index_operation.source {
                IndexSource::Load(load) => {
                    let index_config: &IndexConfiguration = &input.to_config()?;

                    let index =
                        { utils::tokio::block_on(load_index::<_>(&load.load_path, index_config))? };

                    (Arc::new(index), None::<BuildStats>, MicroSeconds::new(0))
                }
                IndexSource::Build(build) => {
                    let data: Arc<Matrix<T>> =
                        Arc::new(datafiles::load_dataset(datafiles::BinFile(&build.data))?);

                    let start = std::time::Instant::now();
                    let table = {
                        let train_data = Matrix::try_from(
                            (&*T::as_f32(data.as_slice())?).into(),
                            data.nrows(),
                            data.ncols(),
                        )?;

                        diskann_async::train_pq(
                            train_data.as_view(),
                            input.num_pq_chunks,
                            &mut StdRng::seed_from_u64(input.seed),
                            diskann_providers::utils::create_thread_pool(build.num_threads)?
                                .as_ref(),
                        )?
                    };

                    let create_index = |data_view: MatrixView<T>| {
                        let index = diskann_async::new_quant_index::<T, _, _>(
                            input.try_as_config()?.build()?,
                            input.inmem_parameters(data_view.nrows(), data_view.ncols())?,
                            table,
                            common::NoDeletes,
                        )?;
                        build::set_start_points(
                            index.provider(),
                            data_view,
                            build.start_point_strategy,
                        )?;
                        Ok(index)
                    };
                    let quant_training_time: MicroSeconds = start.elapsed().into();

                    let (index, build_stats) = run_build(
                        build,
                        hybrid,
                        None,
                        output,
                        create_index,
                        single_or_multi_insert,
                    )?;

                    // Save the index if requested
                    if let Some(save_path) = &build.save_path {
                        utils::tokio::block_on(save_index(index.clone(), save_path))?;
                    }

                    (index, Some(build_stats), quant_training_time)
                }
            };

            // Save construction stats before running queries.
            checkpoint.checkpoint(&build_stats)?;

            let search_phase = &input.index_operation.search_phase;
            let search = if input.use_fp_for_search {
                self.full_search
                    .run(index, search_phase, &Strategy::new(common::FullPrecision))?
            } else {
                self.quant_search
                    .run(index, search_phase, &Strategy::new(hybrid))?
            };

            let result = QuantBuildResult {
                quant_training_time,
                build: BuildResult::new(build_stats, search),
            };

            writeln!(output, "\n\n{}", result)?;
            Ok(result)
        }
    }
}
