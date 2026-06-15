/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

// Create a stub-module if the "scalar-quantization" feature is disabled.
crate::utils::stub_impl!("scalar-quantization", inputs::graph_index::IndexSQOperation);

pub(crate) fn register_benchmarks(benchmarks: &mut Registry) -> anyhow::Result<()> {
    #[cfg(feature = "scalar-quantization")]
    {
        use crate::index::search::plugins::Topk;

        // NOTE: This benchmark is heavily monomorphized. Each `(NBITS, T)` pair
        // generates a full `Benchmark` impl/build path for
        // `ScalarQuantized<NBITS, T>` via the `impl_sq_build!` macro in `mod imp`,
        // which materially impacts compile time. We intentionally keep the registered
        // set minimal (`f32` at 1, 4, and 8 bits) to cover the common cases used by
        // `example/scalar.json`.
        //
        // To add a new variant (e.g. another bit-width or element type):
        //   1. Add a `benchmarks.register("graph-index-sq-<N>-bit-<T>",
        //      imp::ScalarQuantized::<N, T>::new().search(Topk));` call here.
        //   2. Add a matching `impl_sq_build!(N, T);` invocation at the bottom of
        //      `mod imp` below.
        //
        // Search plugins (e.g. `Range`, filter variants) are also monomorphized per
        // variant, so additions multiply compile cost across every registered variant.
        // Only `Topk` is registered today; add others sparingly.

        benchmarks.register(
            "graph-index-sq-8-bit-f32",
            imp::ScalarQuantized::<8, f32>::new().search(Topk),
        )?;
        benchmarks.register(
            "graph-index-sq-4-bit-f32",
            imp::ScalarQuantized::<4, f32>::new().search(Topk),
        )?;
        benchmarks.register(
            "graph-index-sq-1-bit-f32",
            imp::ScalarQuantized::<1, f32>::new().search(Topk),
        )?;
    }

    // Stub implementation
    #[cfg(not(feature = "scalar-quantization"))]
    imp::register("graph-index-sq", benchmarks)?;

    Ok(())
}

#[cfg(feature = "scalar-quantization")]
mod imp {
    use std::{io::Write, sync::Arc};

    use anyhow::Context;
    use diskann::utils::VectorRepr;
    use diskann_benchmark_runner::{
        benchmark::{FailureScore, MatchScore},
        utils::{datatype::AsDataType, MicroSeconds},
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

    use crate::{
        index::{
            benchmarks::{run_build, QueryType, Strategy},
            build::{self, load_index, only_single_insert, save_index, BuildStats},
            result::{BuildResult, QuantBuildResult},
            search::plugins,
        },
        inputs::graph_index::{IndexSQOperation, IndexSource, SearchPhase},
        utils::{self, datafiles},
    };

    type SQProvider<const NBITS: usize, T> = inmem::DefaultProvider<
        inmem::FullPrecisionStore<T>,
        inmem::SQStore<NBITS>,
        common::NoDeletes,
        diskann::provider::DefaultContext,
    >;

    impl<const NBITS: usize, T> QueryType for SQProvider<NBITS, T>
    where
        T: VectorRepr,
    {
        type Element = T;
    }

    /// A [`Benchmark`] for scalar-quantized searches containing a dynamic list of search
    /// types.
    ///
    /// The kinds of quantized and full-precision searches are kept in-sync.
    pub(crate) struct ScalarQuantized<const NBITS: usize, T>
    where
        T: VectorRepr,
    {
        quant_search:
            plugins::Plugins<SQProvider<NBITS, T>, SearchPhase, Strategy<common::Quantized>>,
        full_search:
            plugins::Plugins<SQProvider<NBITS, T>, SearchPhase, Strategy<common::FullPrecision>>,
    }

    impl<const NBITS: usize, T> ScalarQuantized<NBITS, T>
    where
        T: VectorRepr,
    {
        pub(crate) fn new() -> Self {
            Self {
                quant_search: plugins::Plugins::new(),
                full_search: plugins::Plugins::new(),
            }
        }

        pub(crate) fn search<P>(mut self, plugin: P) -> Self
        where
            P: plugins::Plugin<SQProvider<NBITS, T>, SearchPhase, Strategy<common::Quantized>>
                + plugins::Plugin<SQProvider<NBITS, T>, SearchPhase, Strategy<common::FullPrecision>>
                + Clone
                + 'static,
        {
            self.quant_search.register(plugin.clone());
            self.full_search.register(plugin);
            self
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
                            if build.multi_insert().is_some() {
                                failure_score = Some(1);
                            }
                        }
                    }

                    if !<$T>::is_match(*input.index_operation.source.data_type()) {
                        *failure_score.get_or_insert(0) += 1;
                    }

                    if !self.quant_search.is_match(&input.index_operation.search_phase) {
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
                            writeln!(
                                f,
                                "- Index Build and Search using {} scalar quantized bits",
                                $N
                            )?;
                            writeln!(
                                f,
                                "- Requires `{}` data",
                                <$T>::DATA_TYPE,
                            )?;
                            writeln!(f, "- Implements `squared_l2` or `inner_product` distance",)?;
                            writeln!(f, "- Does not support multi-insert")?;
                            writeln!(f, "- Search Kinds: {}", self.quant_search.format_kinds())?;
                        }
                        Some(input) => {
                            if input.num_bits != $N {
                                writeln!(
                                    f,
                                    "- Expected {} bits, instead got {}",
                                    $N,
                                    input.num_bits
                                )?;
                            }

                            let data_type = *input.index_operation.source.data_type();
                            if !<$T>::is_match(data_type) {
                                writeln!(
                                    f,
                                    "- Only `{}` data type is supported. Instead, got {}",
                                    <$T>::DATA_TYPE,
                                    data_type
                                )?;
                            }

                            if let IndexSource::Build(ref build) = input.index_operation.source {
                                if build.multi_insert().is_some() {
                                    writeln!(
                                        f,
                                        "- Scalar Quantization does not support multi-insert"
                                    )?;
                                }
                            }

                            if !self.quant_search.is_match(&input.index_operation.search_phase) {
                                writeln!(
                                    f,
                                    "- Unsupported search phase: \"{}\" - expected one of {}",
                                    input.index_operation.search_phase.kind(),
                                    self.quant_search.format_kinds(),
                                )?;
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
                                Arc::new(datafiles::load_dataset(datafiles::BinFile(build.data()))?);

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
                        build::set_start_points(index.provider(), data_view, *build.start_point_strategy())?;
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

                        if let Some(save_path) = build.save_path() {
                            utils::tokio::block_on(save_index(index.clone(), save_path))?;
                        }

                        (index, Some(build_stats), quant_training_time)
                    }

                    };


                    // Save construction stats before running queries.
                    checkpoint.checkpoint(&build_stats)?;

                    let search = if input.use_fp_for_search {
                        self.full_search.run(
                            index,
                            &input.index_operation.search_phase,
                            &Strategy::new(common::FullPrecision),
                        )?
                    } else {
                        self.quant_search.run(
                            index,
                            &input.index_operation.search_phase,
                            &Strategy::new(common::Quantized),
                        )?
                    };

                    let result = QuantBuildResult {
                        quant_training_time,
                        build: BuildResult::new(build_stats, search),
                    };

                    writeln!(output, "\n\n{}", result)?;
                    Ok(result)
                }
            }
        };
    }

    // See the doc comment in `register_benchmarks` above for the policy on
    // adding/removing variants. Each invocation here generates a full `Benchmark`
    // impl and materially affects compile time.
    impl_sq_build!(8, f32);
    impl_sq_build!(4, f32);
    impl_sq_build!(1, f32);
}
