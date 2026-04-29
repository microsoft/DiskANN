/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

// Create a stub-module if the "spherical-quantization" feature is disabled.
crate::utils::stub_impl!("product-quantization", inputs::async_::IndexPQOperation);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    #[cfg(feature = "product-quantization")]
    {
        use half::f16;

        benchmarks.register::<imp::ProductQuantized<'static, f32>>("async-pq-f32");
        benchmarks.register::<imp::ProductQuantized<'static, f16>>("async-pq-f16");
    }

    // Stub implementation
    #[cfg(not(feature = "product-quantization"))]
    imp::register("async-pq", benchmarks);
}

#[cfg(feature = "product-quantization")]
mod imp {
    use std::{io::Write, sync::Arc};

    use diskann::utils::VectorRepr;
    use diskann_providers::{
        index::diskann_async::{self},
        model::{graph::provider::async_::common, IndexConfiguration},
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
            benchmarks::{run_build, run_search_outer, BuildAndSearch, FullPrecision},
            build::{self, load_index, save_index, single_or_multi_insert, BuildStats},
            result::QuantBuildResult,
        },
        inputs::async_::{IndexPQOperation, IndexSource},
        utils::{self, datafiles},
    };

    pub(super) struct ProductQuantized<'a, T> {
        input: &'a IndexPQOperation,
        _type: std::marker::PhantomData<T>,
    }

    impl<'a, T> ProductQuantized<'a, T> {
        fn new(input: &'a IndexPQOperation) -> Self {
            Self {
                input,
                _type: std::marker::PhantomData,
            }
        }
    }

    impl<T> Benchmark for ProductQuantized<'static, T>
    where
        T: VectorRepr
            + diskann_utils::sampling::WithApproximateNorm
            + diskann::graph::SampleableForStart,
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Input = IndexPQOperation;
        type Output = QuantBuildResult;

        fn try_match(input: &IndexPQOperation) -> Result<MatchScore, FailureScore> {
            <FullPrecision<'static, T> as Benchmark>::try_match(&input.index_operation)
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            input: Option<&IndexPQOperation>,
        ) -> std::fmt::Result {
            <FullPrecision<'static, T> as Benchmark>::description(
                f,
                input.map(|f| &f.index_operation),
            )
        }

        fn run(
            input: &IndexPQOperation,
            checkpoint: Checkpoint<'_>,
            output: &mut dyn Output,
        ) -> anyhow::Result<QuantBuildResult> {
            let pq = ProductQuantized::<T>::new(input);
            BuildAndSearch::run(pq, checkpoint, output)
        }
    }

    impl<'a, T> BuildAndSearch<'a> for ProductQuantized<'a, T>
    where
        T: VectorRepr
            + diskann_utils::sampling::WithApproximateNorm
            + diskann::graph::SampleableForStart,
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Data = QuantBuildResult;
        fn run(
            self,
            checkpoint: Checkpoint<'_>,
            mut output: &mut dyn Output,
        ) -> Result<Self::Data, anyhow::Error> {
            writeln!(output, "{}", self.input)?;

            let hybrid = common::Hybrid::new(self.input.max_fp_vecs_per_prune);

            let (index, build_stats, quant_training_time) = match &self.input.index_operation.source
            {
                IndexSource::Load(load) => {
                    let index_config: &IndexConfiguration = &self.input.to_config()?;

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
                            self.input.num_pq_chunks,
                            &mut StdRng::seed_from_u64(self.input.seed),
                            diskann_providers::utils::create_thread_pool(build.num_threads)?
                                .as_ref(),
                        )?
                    };

                    let create_index = |data_view: MatrixView<T>| {
                        let index = diskann_async::new_quant_index::<T, _, _>(
                            self.input.try_as_config()?.build()?,
                            self.input
                                .inmem_parameters(data_view.nrows(), data_view.ncols())?,
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

            let build = if self.input.use_fp_for_search {
                run_search_outer(
                    &self.input.index_operation.search_phase,
                    common::FullPrecision,
                    index,
                    build_stats,
                    checkpoint,
                )?
            } else {
                run_search_outer(
                    &self.input.index_operation.search_phase,
                    hybrid,
                    index,
                    build_stats,
                    checkpoint,
                )?
            };

            let result = QuantBuildResult {
                quant_training_time,
                build,
            };

            writeln!(output, "\n\n{}", result)?;
            Ok(result)
        }
    }
}
