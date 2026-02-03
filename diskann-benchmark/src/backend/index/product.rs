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

        use crate::backend::index::benchmarks::register;

        register!(
            benchmarks,
            "async-pq-f32",
            imp::ProductQuantized<'static, f32>
        );
        register!(
            benchmarks,
            "async-pq-f16",
            imp::ProductQuantized<'static, f16>
        );
    }

    // Stub implementation
    #[cfg(not(feature = "product-quantization"))]
    imp::register("async-pq", benchmarks);
}

#[cfg(feature = "product-quantization")]
mod imp {
    use std::{io::Write, sync::Arc};

    use diskann_providers::{
        model::{graph::provider::async_::common, IndexConfiguration},
    };
    use diskann_inmem::diskann_async;
    use diskann_utils::views::{Matrix, MatrixView};

    use diskann_benchmark_runner::{
        dispatcher::{self, DispatchRule, FailureScore, MatchScore},
        utils::MicroSeconds,
        Any, Checkpoint, Output,
    };
    use half::f16;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::{
        backend::index::{
            benchmarks::{
                run_build, run_search_outer, single_or_multi_insert, BuildAndSearch, FullPrecision,
            },
            build::{self, load_index, save_index},
            result::{BuildStats, QuantBuildResult},
        },
        inputs::async_::{IndexOperation, IndexPQOperation, IndexSource},
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

    impl<T> dispatcher::Map for ProductQuantized<'static, T>
    where
        T: 'static,
    {
        type Type<'a> = ProductQuantized<'a, T>;
    }

    impl<'a, T> DispatchRule<&'a IndexPQOperation> for ProductQuantized<'a, T>
    where
        FullPrecision<'a, T>: DispatchRule<&'a IndexOperation>,
    {
        type Error = std::convert::Infallible;

        // Matching simply requires that we match the inner type.
        fn try_match(from: &&'a IndexPQOperation) -> Result<MatchScore, FailureScore> {
            FullPrecision::<'a, T>::try_match(&&from.index_operation)
        }

        fn convert(from: &'a IndexPQOperation) -> Result<Self, Self::Error> {
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a IndexPQOperation>,
        ) -> std::fmt::Result {
            FullPrecision::<'a, T>::description(f, from.map(|f| &f.index_operation).as_ref())
        }
    }

    impl<'a, T> DispatchRule<&'a Any> for ProductQuantized<'a, T>
    where
        ProductQuantized<'a, T>:
            DispatchRule<&'a IndexPQOperation, Error = std::convert::Infallible>,
    {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<IndexPQOperation, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<IndexPQOperation, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<IndexPQOperation, Self>(f, from, IndexPQOperation::tag())
        }
    }

    macro_rules! impl_pq_build {
        ($T:ty) => {
            impl<'a> BuildAndSearch<'a> for ProductQuantized<'a, $T> {
                type Data = QuantBuildResult;
                fn run(
                    self,
                    checkpoint: Checkpoint<'_>,
                    mut output: &mut dyn Output,
                ) -> Result<Self::Data, anyhow::Error> {
                    writeln!(output, "{}", self.input)?;

                    let hybrid = common::Hybrid::new(self.input.max_fp_vecs_per_prune);

                    let (index, build_stats, quant_training_time) = match &self
                        .input
                        .index_operation
                        .source
                    {
                        IndexSource::Load(load) => {
                            let index_config: &IndexConfiguration = &self.input.to_config()?;

                            let index = {
                                utils::tokio::block_on(load_index::<_>(
                                    &load.load_path,
                                    index_config,
                                ))?
                            };

                            (Arc::new(index), None::<BuildStats>, MicroSeconds::new(0))
                        }
                        IndexSource::Build(build) => {
                            let data: Arc<Matrix<$T>> =
                                Arc::new(datafiles::load_dataset(datafiles::BinFile(&build.data))?);

                            let start = std::time::Instant::now();
                            let table = {
                                let train_data = Matrix::try_from(
                                    data.as_slice().iter().copied().map(f32::from).collect(),
                                    data.nrows(),
                                    data.ncols(),
                                )?;

                                diskann_async::train_pq(
                                    train_data.as_view(),
                                    self.input.num_pq_chunks,
                                    &mut StdRng::seed_from_u64(self.input.seed),
                                    build.num_threads,
                                )?
                            };

                            let create_index = |data_view: MatrixView<$T>| {
                                let index = diskann_async::new_quant_index::<$T, _, _>(
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
        };
    }

    impl_pq_build!(f32);
    impl_pq_build!(f16);
}
