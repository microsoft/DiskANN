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

        use crate::backend::index::benchmarks::register;

        // f32
        register!(
            benchmarks,
            "async-sq-8-bit-f32",
            imp::ScalarQuantized<'static, 8, f32>
        );
        register!(
            benchmarks,
            "async-sq-4-bit-f32",
            imp::ScalarQuantized<'static, 4, f32>
        );
        register!(
            benchmarks,
            "async-sq-2-bit-f32",
            imp::ScalarQuantized<'static, 2, f32>
        );
        register!(
            benchmarks,
            "async-sq-1-bit-f32",
            imp::ScalarQuantized<'static, 1, f32>
        );
        // f16
        register!(
            benchmarks,
            "async-sq-8-bit-f16",
            imp::ScalarQuantized<'static, 8, f16>
        );
        register!(
            benchmarks,
            "async-sq-4-bit-f16",
            imp::ScalarQuantized<'static, 4, f16>
        );
        register!(
            benchmarks,
            "async-sq-2-bit-f16",
            imp::ScalarQuantized<'static, 2, f16>
        );
        register!(
            benchmarks,
            "async-sq-1-bit-f16",
            imp::ScalarQuantized<'static, 1, f16>
        );
        // i8
        register!(
            benchmarks,
            "async-sq-1-bit-i8",
            imp::ScalarQuantized<'static, 1, i8>
        );
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
        dispatcher::{self, DispatchRule, FailureScore, MatchScore},
        utils::{datatype, MicroSeconds},
        Any, Checkpoint, Output,
    };
    use diskann_providers::{
        model::{
            configuration::IndexConfiguration,
            graph::provider::async_::common,
        },
    };
    use diskann_inmem::{self as inmem, diskann_async};
    use diskann_utils::views::{Matrix, MatrixView};
    use half::f16;

    use crate::{
        backend::index::{
            benchmarks::{
                only_single_insert, run_build, run_search_outer, BuildAndSearch, FullPrecision,
            },
            build::{self, load_index, save_index},
            result::{BuildStats, QuantBuildResult},
        },
        inputs::async_::{IndexSQOperation, IndexSource},
        utils::{self, datafiles},
    };

    // Scalar Quantized
    pub(super) struct ScalarQuantized<'a, const NBITS: usize, T> {
        input: &'a IndexSQOperation,
        _type: std::marker::PhantomData<T>,
    }

    impl<'a, const NBITS: usize, T> ScalarQuantized<'a, NBITS, T> {
        fn new(input: &'a IndexSQOperation) -> Self {
            assert_eq!(input.num_bits, NBITS);
            Self {
                input,
                _type: std::marker::PhantomData,
            }
        }
    }

    impl<const NBITS: usize, T> dispatcher::Map for ScalarQuantized<'static, NBITS, T>
    where
        T: 'static,
    {
        type Type<'a> = ScalarQuantized<'a, NBITS, T>;
    }

    impl<'a, const NBITS: usize, T> DispatchRule<&'a IndexSQOperation> for ScalarQuantized<'a, NBITS, T>
    where
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Error = std::convert::Infallible;

        fn try_match(from: &&'a IndexSQOperation) -> Result<MatchScore, FailureScore> {
            // If this is multi-insert, return a very-close failure.
            let mut failure_score: Option<u32> = None;
            match from.index_operation.source {
                IndexSource::Load(_) => {}
                IndexSource::Build(ref build) => {
                    // If the build is not compatible, return a failure score.
                    if build.multi_insert.is_some() {
                        failure_score = Some(1);
                    }
                }
            }

            // make sure the data type is correct
            if let Err(FailureScore(_)) = FullPrecision::<'a, T>::try_match(&&from.index_operation)
            {
                *failure_score.get_or_insert(0) += 1;
            }

            // Make sure the number of bits is correct.
            if from.num_bits != NBITS {
                *failure_score.get_or_insert(0) += 10 + NBITS.abs_diff(from.num_bits) as u32;
            }

            match failure_score {
                None => Ok(MatchScore(0)),
                Some(score) => Err(FailureScore(score)),
            }
        }

        fn convert(from: &'a IndexSQOperation) -> Result<Self, Self::Error> {
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a IndexSQOperation>,
        ) -> std::fmt::Result {
            match from {
                None => {
                    describeln!(
                        f,
                        "- Index Build and Search using {} scalar quantized bits",
                        NBITS
                    )?;
                    describeln!(
                        f,
                        "- Requires `{}` data",
                        dispatcher::Description::<datatype::DataType, datatype::Type<T>>::new(),
                    )?;
                    describeln!(f, "- Implements `squared_l2` or `inner_product` distance",)?;
                    describeln!(f, "- Does not support multi-insert")?;
                }
                Some(input) => {
                    if input.num_bits != NBITS {
                        describeln!(
                            f,
                            "- Expected {} bits, instead got {}",
                            NBITS,
                            input.num_bits
                        )?;
                    }

                    let mut check_match = |data_type: &datatype::DataType| {
                        if datatype::Type::<T>::try_match(data_type).is_err() {
                            describeln!(
                                f,
                                "- Only `{}` data type is supported. Instead, got {}",
                                dispatcher::Description::<datatype::DataType, datatype::Type<T>>::new(),
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
    }

    impl<'a, const NBITS: usize, T> DispatchRule<&'a Any> for ScalarQuantized<'a, NBITS, T>
    where
        datatype::Type<T>: DispatchRule<datatype::DataType>,
    {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<IndexSQOperation, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<IndexSQOperation, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<IndexSQOperation, Self>(f, from, IndexSQOperation::tag())
        }
    }

    macro_rules! impl_sq_build {
        ($N:literal, $T: ty) => {
            impl<'a> BuildAndSearch<'a> for ScalarQuantized<'a, $N, $T> {
                type Data = QuantBuildResult;
                fn run(
                    self,
                    checkpoint: Checkpoint<'_>,
                    mut output: &mut dyn Output,
                ) -> Result<Self::Data, anyhow::Error> {
                    writeln!(output, "{}", self.input)?;

                    let (index, build_stats, quant_training_time) = match &self.input.index_operation.source {
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
                            diskann_quantization::num::Positive::new(self.input.standard_deviations).context(
                                "please file a bug report, this should not have made it past the\
                                    front end",
                            )?,
                        )
                        .train(data.as_view());
                                            let create_index = |data_view: MatrixView<$T>| {
                        let index = diskann_async::new_quant_index::<$T, _, _>(
                            self.input.try_as_config()?.build()?,
                            self.input
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
