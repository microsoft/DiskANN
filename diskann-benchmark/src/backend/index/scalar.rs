/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

// Create a stub-module if the "scalar-quantization" feature is disabled.
crate::utils::stub_impl!("scalar-quantization", inputs::async_::IndexSQOperation);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    #[cfg(feature = "scalar-quantization")]
    {
        use half::f16;

        // f32
        benchmarks.register("async-sq-8-bit-f32", imp::ScalarQuantized::<8, f32>::new());
        benchmarks.register("async-sq-4-bit-f32", imp::ScalarQuantized::<4, f32>::new());
        benchmarks.register("async-sq-2-bit-f32", imp::ScalarQuantized::<2, f32>::new());
        benchmarks.register("async-sq-1-bit-f32", imp::ScalarQuantized::<1, f32>::new());
        // f16
        benchmarks.register("async-sq-8-bit-f16", imp::ScalarQuantized::<8, f16>::new());
        benchmarks.register("async-sq-4-bit-f16", imp::ScalarQuantized::<4, f16>::new());
        benchmarks.register("async-sq-2-bit-f16", imp::ScalarQuantized::<2, f16>::new());
        benchmarks.register("async-sq-1-bit-f16", imp::ScalarQuantized::<1, f16>::new());
        // i8
        benchmarks.register("async-sq-1-bit-i8", imp::ScalarQuantized::<1, i8>::new());
    }

    // Stub implementation
    #[cfg(not(feature = "scalar-quantization"))]
    imp::register("async-pq", benchmarks);
}

#[cfg(feature = "scalar-quantization")]
mod imp {
    use std::{io::Write, sync::Arc};

    use anyhow::Context;
    use diskann_benchmark_runner::{
        describeln,
        dispatcher::{Description, DispatchRule, FailureScore, MatchScore},
        utils::{datatype, MicroSeconds},
        Benchmark, Checkpoint, Output,
    };
    use diskann_providers::{
        index::diskann_async::{self},
        model::{
            configuration::IndexConfiguration,
            graph::provider::async_::{common, inmem},
        },
    };
    use diskann_utils::views::{Matrix, MatrixView};
    use half::f16;

    use crate::{
        backend::index::{
            benchmarks::{run_build, run_search_outer, FullPrecision},
            build::{self, load_index, only_single_insert, save_index, BuildStats},
            result::QuantBuildResult,
        },
        inputs::async_::{IndexSQOperation, IndexSource},
        utils::{self, datafiles},
    };

    // Scalar Quantized
    pub(super) struct ScalarQuantized<const NBITS: usize, T> {
        _type: std::marker::PhantomData<T>,
    }

    impl<const NBITS: usize, T> ScalarQuantized<NBITS, T> {
        pub(super) fn new() -> Self {
            Self {
                _type: std::marker::PhantomData,
            }
        }
    }

    macro_rules! impl_sq_build {
        ($N:literal, $T: ty) => {
            impl Benchmark for ScalarQuantized<$N, $T> {
                type Input = IndexSQOperation;
                type Output = QuantBuildResult;

                fn try_match(&self, input: &IndexSQOperation) -> Result<MatchScore, FailureScore> {
                    let mut failure_score: Option<u32> = None;
                    match input.index_operation.source {
                        IndexSource::Load(_) => {}
                        IndexSource::Build(ref build) => {
                            if build.multi_insert.is_some() {
                                failure_score = Some(1);
                            }
                        }
                    }

                    if FullPrecision::<$T>::new().try_match(&input.index_operation)
                        .is_err()
                    {
                        *failure_score.get_or_insert(0) += 1;
                    }

                    if input.num_bits != $N {
                        *failure_score.get_or_insert(0) += 10 + ($N as usize).abs_diff(input.num_bits) as u32;
                    }

                    match failure_score {
                        None => Ok(MatchScore(0)),
                        Some(score) => Err(FailureScore(score)),
                    }
                }

                fn description(
                    &self,
                    f: &mut std::fmt::Formatter<'_>,
                    input: Option<&IndexSQOperation>,
                ) -> std::fmt::Result {
                    match input {
                        None => {
                            describeln!(
                                f,
                                "- Index Build and Search using {} scalar quantized bits",
                                $N
                            )?;
                            describeln!(
                                f,
                                "- Requires `{}` data",
                                Description::<datatype::DataType, datatype::Type<$T>>::new(),
                            )?;
                            describeln!(f, "- Implements `squared_l2` or `inner_product` distance",)?;
                            describeln!(f, "- Does not support multi-insert")?;
                        }
                        Some(input) => {
                            if input.num_bits != $N {
                                describeln!(
                                    f,
                                    "- Expected {} bits, instead got {}",
                                    $N,
                                    input.num_bits
                                )?;
                            }

                            let mut check_match = |data_type: &datatype::DataType| {
                                if datatype::Type::<$T>::try_match(data_type).is_err() {
                                    describeln!(
                                        f,
                                        "- Only `{}` data type is supported. Instead, got {}",
                                        Description::<datatype::DataType, datatype::Type<$T>>::new(),
                                        data_type
                                    ).unwrap();
                                }
                            };

                            match &input.index_operation.source {
                                IndexSource::Load(load) => {
                                    check_match(&load.data_type);
                                }
                                IndexSource::Build(build) => {
                                    check_match(&build.data_type);

                                    if build.multi_insert.is_some() {
                                        describeln!(
                                            f,
                                            "- Scalar Quantization does not support multi-insert"
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                    Ok(())
                }

                fn run(
                    &self,
                    input: &IndexSQOperation,
                    checkpoint: Checkpoint<'_>,
                    mut output: &mut dyn Output,
                ) -> anyhow::Result<QuantBuildResult> {
                    assert_eq!(
                        input.num_bits,
                        $N,
                        "INTERNAL ERROR: this should not have passed the match check"
                    );

                    writeln!(output, "{}", input)?;

                    let (index, build_stats, quant_training_time) = match &input.index_operation.source {
                        IndexSource::Load(load) => {
                            let index_config: &IndexConfiguration = &load.to_config()?;


                            let index = {
                                utils::tokio::block_on(load_index::<_>(&load.load_path, index_config))?
                            };

                            (Arc::new(index), None::<BuildStats>, MicroSeconds::new(0))
                        }
                        IndexSource::Build(build) => {
                            let data: Arc<Matrix<$T>> =
                                Arc::new(datafiles::load_dataset(datafiles::BinFile(&build.data))?);

                        let start = std::time::Instant::now();
                        let quantizer = diskann_quantization::scalar::train::ScalarQuantizationParameters::new(
                            diskann_quantization::num::Positive::new(input.standard_deviations).context(
                                "please file a bug report, this should not have made it past the\
                                    front end",
                            )?,
                        )
                        .train(data.as_view());
                                            let create_index = |data_view: MatrixView<$T>| {
                        let index = diskann_async::new_quant_index::<$T, _, _>(
                            input.try_as_config()?.build()?,
                            input
                                .inmem_parameters(data_view.nrows(), data_view.ncols())?,
                            inmem::WithBits::<$N>::new(quantizer),
                            common::NoDeletes,
                        )?;
                        build::set_start_points(index.provider(), data_view, build.start_point_strategy)?;
                        Ok(index)
                    };

                        let quant_training_time: MicroSeconds = start.elapsed().into();

                        let (index, build_stats) = run_build(
                            &build,
                            common::Quantized,
                            None,
                            output,
                            create_index,
                            only_single_insert,
                        )?;

                        if let Some(save_path) = &build.save_path {
                            utils::tokio::block_on(save_index(index.clone(), save_path))?;
                        }

                        (index, Some(build_stats), quant_training_time)
                    }

                    };


                    let build = if input.use_fp_for_search {
                        run_search_outer(
                            &input.index_operation.search_phase,
                            common::FullPrecision,
                            index,
                            build_stats,
                            checkpoint,
                        )?
                    } else {
                        run_search_outer(
                            &input.index_operation.search_phase,
                            common::Quantized,
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

    impl_sq_build!(8, f32);
    impl_sq_build!(4, f32);
    impl_sq_build!(2, f32);
    impl_sq_build!(1, f32);

    impl_sq_build!(8, f16);
    impl_sq_build!(4, f16);
    impl_sq_build!(2, f16);
    impl_sq_build!(1, f16);

    impl_sq_build!(1, i8);
}
