/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

const NAME: &str = "minmax-exhaustive-search";

crate::utils::stub_impl!("minmax-quantization", inputs::exhaustive::MinMax);

// MinMax - requires feature "minmax-quantization"
#[cfg(feature = "minmax-quantization")]
pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    benchmarks.register(NAME, imp::MinMaxQ::<1>);
    benchmarks.register(NAME, imp::MinMaxQ::<2>);
    benchmarks.register(NAME, imp::MinMaxQ::<4>);
    benchmarks.register(NAME, imp::MinMaxQ::<8>);
}

// Stub implementation
#[cfg(not(feature = "minmax-quantization"))]
pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    imp::register(NAME, benchmarks)
}

/////////////
// MinMaxQ //
/////////////

#[cfg(feature = "minmax-quantization")]
mod imp {
    use std::{io::Write, num::NonZeroUsize};

    use diskann_benchmark_runner::{
        dispatcher::{FailureScore, MatchScore},
        utils::{percentiles, MicroSeconds},
        Benchmark, Output,
    };
    use diskann_quantization::{
        algorithms::transforms::Transform,
        bits::{Representation, Unsigned},
        distances,
        minmax::{
            self, Data, DataRef, MinMaxCosine, MinMaxCosineNormalized, MinMaxIP, MinMaxL2Squared,
            MinMaxQuantizer,
        },
        num::Positive,
        AsFunctor, CompressInto,
    };
    use diskann_utils::{Reborrow, ReborrowMut};
    use diskann_vector::{PreprocessedDistanceFunction, PureDistanceFunction};
    use indicatif::{ProgressBar, ProgressStyle};
    use rand::{rngs::StdRng, SeedableRng};
    use serde::Serialize;

    use crate::{
        backend::exhaustive::algos::{self, LinearSearch},
        inputs::{self, exhaustive::MinMaxQuery},
        utils::{
            datafiles::{self, ConvertingLoad},
            recall, SimilarityMeasure,
        },
    };

    macro_rules! write_field {
        ($f:ident, $field:tt, $fmt:literal, $($expr:tt)*) => {
            writeln!($f, concat!("{:>19}: ", $fmt), $field, $($expr)*)
        }
    }

    type RF32 = diskann_quantization::distances::Result<f32>;

    fn make_progress_bar(
        message: &str,
        count: usize,
        draw_target: indicatif::ProgressDrawTarget,
    ) -> anyhow::Result<ProgressBar> {
        let progress = ProgressBar::with_draw_target(Some(count as u64), draw_target);
        progress.set_style(ProgressStyle::with_template(&format!(
            "{} [{{elapsed_precise}}] {{wide_bar}} {{percent}}",
            message
        ))?);
        Ok(progress)
    }

    /// The dispatcher target for `minmax-quantization` operations.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct MinMaxQ<const NBITS: usize>;

    impl<const NBITS: usize> MinMaxQ<NBITS> {
        pub(super) fn run(
            &self,
            input: &inputs::exhaustive::MinMax,
            mut output: &mut dyn Output,
        ) -> anyhow::Result<Results>
        where
            Unsigned: Representation<NBITS>,
            Plan: algos::CreateQuantComputer<Store<NBITS>>,
        {
            writeln!(output, "{}", input)?;

            // Training
            let data = f32::converting_load(datafiles::BinFile(&input.data), input.data_type)?;
            let start = std::time::Instant::now();

            let mut rng = StdRng::seed_from_u64(input.seed);

            let dim = NonZeroUsize::new(data.ncols()).unwrap();
            let transform = Transform::new(
                (&input.transform_kind).into(),
                dim,
                Some(&mut rng),
                diskann_quantization::alloc::GlobalAllocator,
            )?;

            let quantizer = MinMaxQuantizer::new(transform, Positive::new(input.scale)?);

            let training_time: MicroSeconds = start.elapsed().into();

            // Compressing
            let start = std::time::Instant::now();
            let store = {
                let compression_progress =
                    make_progress_bar("compressing", data.nrows(), output.draw_target())?;
                let store = Store::<NBITS>::new(data.as_view(), quantizer, &compression_progress)?;
                compression_progress.finish();
                store
            };
            let compression_time: MicroSeconds = start.elapsed().into();

            // Search
            let queries =
                f32::converting_load(datafiles::BinFile(&input.search.queries), input.data_type)?;

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth))?;
            let mut search_results = Vec::<SearchResults>::new();
            let threadpool = rayon::ThreadPoolBuilder::new()
                .num_threads(input.search.num_threads.get())
                .build()?;

            for layout in &input.query_layouts {
                let search_progress =
                    make_progress_bar("running search", queries.nrows(), output.draw_target())?;

                let recall_n =
                    input.search.recalls.recall_n.last().ok_or_else(|| {
                        anyhow::anyhow!("expected at least one value for `recall_n`")
                    })?;

                let plan = Plan {
                    measure: input.distance,
                    layout: *layout,
                };

                let r = threadpool.install(|| {
                    algos::linear_search(
                        &store,
                        queries.as_view(),
                        &plan,
                        *recall_n,
                        &search_progress,
                    )
                })?;

                let recalls = recall::compute_multiple_recalls(
                    &r.ids,
                    &groundtruth,
                    &input.search.recalls.recall_k,
                    &input.search.recalls.recall_n,
                )?;

                search_results.push(SearchResults::new(
                    r,
                    input.search.num_threads.get(),
                    *layout,
                    recalls,
                )?);

                search_progress.finish();
            }

            // Aggregate and print results.
            let result = Results {
                training_time,
                compression_time,
                search_results,
                quantized_bytes: store.quantized_bytes(),
                original_dim: data.ncols(),
            };

            writeln!(output, "\n\n{}", result)?;
            Ok(result)
        }
    }

    impl<const NBITS: usize> Benchmark for MinMaxQ<NBITS>
    where
        Unsigned: Representation<NBITS>,
        Plan: algos::CreateQuantComputer<Store<NBITS>>,
    {
        type Input = inputs::exhaustive::MinMax;
        type Output = Results;

        fn try_match(
            &self,
            input: &inputs::exhaustive::MinMax,
        ) -> Result<MatchScore, FailureScore> {
            let num_bits = input.num_bits.get();
            if num_bits == NBITS {
                Ok(MatchScore(0))
            } else {
                Err(FailureScore(
                    NBITS.abs_diff(num_bits).try_into().unwrap_or(u32::MAX),
                ))
            }
        }

        fn description(
            &self,
            f: &mut std::fmt::Formatter<'_>,
            input: Option<&inputs::exhaustive::MinMax>,
        ) -> std::fmt::Result {
            match input {
                None => {
                    writeln!(
                        f,
                        "- Exhaustive search for {}-bit minmax quantization",
                        NBITS
                    )?;
                    writeln!(f, "- Requires `float32` data")?;
                    writeln!(f, "- Implements `squared_l2` or `inner_product` distance")?;
                }
                Some(from) => {
                    if from.num_bits.get() != NBITS {
                        writeln!(
                            f,
                            "- Expected \"num_bits = {}\", instead got {}",
                            NBITS,
                            from.num_bits.get(),
                        )?;
                    }
                }
            }
            Ok(())
        }

        fn run(
            &self,
            input: &inputs::exhaustive::MinMax,
            _checkpoint: diskann_benchmark_runner::Checkpoint<'_>,
            output: &mut dyn Output,
        ) -> anyhow::Result<Results> {
            self.run(input, output)
        }
    }

    /// Results from an end-to-end run of Scalar Quantization.
    #[derive(Debug, Serialize)]
    pub(super) struct Results {
        /// The time it takes to generate the base quantizer.
        training_time: MicroSeconds,
        /// How long it takes to compress the raw data.
        compression_time: MicroSeconds,
        /// The minimum number of bytes required for each compressed vecto4r.
        quantized_bytes: usize,
        /// The dimensionality of the uncompressed data.
        original_dim: usize,
        /// Results for each search kind (varying over query layouts).
        search_results: Vec<SearchResults>,
    }

    impl std::fmt::Display for Results {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write_field!(f, "Training Time", "{}s", self.training_time.as_seconds())?;
            write_field!(
                f,
                "Compression Time",
                "{}s",
                self.compression_time.as_seconds()
            )?;
            write_field!(f, "Quantized Bytes", "{}", self.quantized_bytes)?;
            write_field!(f, "Original Dim", "{}\n\n", self.original_dim)?;

            for (i, v) in self.search_results.iter().enumerate() {
                write_field!(f, "Run", "{} of {}", i + 1, self.search_results.len())?;
                writeln!(f, "{}", v)?;
            }
            Ok(())
        }
    }

    #[derive(Debug, Serialize)]
    struct SearchResults {
        num_threads: usize,
        time: MicroSeconds,
        qps: f64,

        layout: MinMaxQuery,

        // Latencies
        mean_preprocess: f64,
        p90_preprocess: MicroSeconds,
        p99_preprocess: MicroSeconds,

        mean_search: f64,
        p90_search: MicroSeconds,
        p99_search: MicroSeconds,

        // Values for each combination of recalls.
        recalls: Vec<recall::RecallMetrics>,
    }

    impl SearchResults {
        fn new(
            mut search: LinearSearch,
            num_threads: usize,
            layout: MinMaxQuery,
            recalls: Vec<recall::RecallMetrics>,
        ) -> Result<Self, percentiles::CannotBeEmpty> {
            let preprocess_latency = percentiles::compute_percentiles(&mut search.preprocess)?;
            let search_latency = percentiles::compute_percentiles(&mut search.search)?;

            let time = search.total;
            Ok(Self {
                num_threads,
                time,
                qps: (search.ids.nrows() as f64) / time.as_seconds(),
                layout,
                mean_preprocess: preprocess_latency.mean,
                p90_preprocess: preprocess_latency.p90,
                p99_preprocess: preprocess_latency.p99,
                mean_search: search_latency.mean,
                p90_search: search_latency.p90,
                p99_search: search_latency.p99,
                recalls,
            })
        }
    }

    impl std::fmt::Display for SearchResults {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write_field!(f, "Query Layout", "{:?}", self.layout)?;
            write_field!(f, "Total", "{:.2}s", self.time.as_seconds())?;
            write_field!(f, "QPS", "{:.3}", self.qps)?;
            write_field!(f, "Num Threads", "{}", self.num_threads)?;
            write_field!(
                f,
                "Preprocess Latency",
                "{:.1}us ({:.1})",
                self.mean_preprocess,
                self.p99_preprocess.as_f64(),
            )?;
            write_field!(
                f,
                "Search Latency",
                "{:.1}us ({:.1})",
                self.mean_search,
                self.p99_search.as_f64(),
            )?;

            writeln!(f)?;

            let header = ["K", "N", "Recall (%)"];
            let mut table =
                diskann_benchmark_runner::utils::fmt::Table::new(header, self.recalls.len());
            self.recalls.iter().enumerate().for_each(|(row, r)| {
                let mut row = table.row(row);
                row.insert(r.recall_k, 0);
                row.insert(r.recall_n, 1);
                row.insert(format!("{:.3}", 100.0 * r.average), 2);
            });

            write!(f, "{}", table)
        }
    }

    /// A store for quantized data.
    pub(super) struct Store<const NBITS: usize> {
        // The number of bytes to take from each row.
        bytes: usize,
        data: diskann_utils::views::Matrix<u8>,
        quantizer: diskann_quantization::minmax::MinMaxQuantizer,
    }

    impl<const NBITS: usize> Store<NBITS>
    where
        Unsigned: Representation<NBITS>,
    {
        fn new(
            input: diskann_utils::views::MatrixView<f32>,
            quantizer: diskann_quantization::minmax::MinMaxQuantizer,
            progress: &ProgressBar,
        ) -> anyhow::Result<Self> {
            // Both the dimensionality of the compressed vector and the exact number of bytes
            // needed to store it can vary.
            //
            // The APIs below should correctly handle these variables.
            let output_dim = quantizer.output_dim();
            let bytes = Data::<NBITS>::canonical_bytes(output_dim);
            let mut data = diskann_utils::views::Matrix::new(0, input.nrows(), bytes);

            // Compress the data.
            //
            // NOTE: If this gets too slow, we can parallelize it.
            std::iter::zip(data.row_iter_mut(), input.row_iter()).try_for_each(
                |(d, i)| -> anyhow::Result<()> {
                    let c = diskann_quantization::minmax::DataMutRef::<NBITS>::from_canonical_front_mut(
                        &mut d[..bytes],
                        output_dim,
                    )?;
                    quantizer.compress_into(i, c)?;
                    progress.inc(1);
                    Ok(())
                },
            )?;

            Ok(Self {
                bytes,
                data,
                quantizer,
            })
        }

        /// Return the number of bytes required to store each quantized vector.
        fn quantized_bytes(&self) -> usize {
            self.bytes
        }
    }

    impl<const NBITS: usize> algos::QuantStore for Store<NBITS>
    where
        Unsigned: Representation<NBITS>,
    {
        type Item<'a>
            = DataRef<'a, NBITS>
        where
            Self: 'a;

        fn iter(&self) -> impl Iterator<Item = Self::Item<'_>> {
            let output_dim = self.quantizer.output_dim();
            self.data.row_iter().map(move |r| {
                DataRef::<NBITS>::from_canonical_front(&r[..self.bytes], output_dim).unwrap()
            })
        }
    }

    /// To handle multiple query bit-widths, we use type erasure on the actual distance
    /// function implementation.
    ///
    /// This struct represents the partial application of the `inner` distance function with
    /// `query` in a generic way so we only have one level of dynamic dispatch when computing
    /// distances.
    struct Curried<D, Q> {
        _inner: D,
        query: Q,
    }

    impl<D, Q> Curried<D, Q> {
        fn new(inner: D, query: Q) -> Self {
            Self {
                _inner: inner,
                query,
            }
        }
    }

    impl<D, Q, T, R> PreprocessedDistanceFunction<T, R> for Curried<D, Q>
    where
        Q: for<'a> Reborrow<'a>,
        D: for<'a> PureDistanceFunction<<Q as Reborrow<'a>>::Target, T, R>,
    {
        fn evaluate_similarity(&self, x: T) -> R {
            D::evaluate(self.query.reborrow(), x)
        }
    }

    /// A thin wrapper around `Box<dyn PreprocessedDistanceFunction<...>.`.
    /// right now, this is not needed for minmax quantizer but keeping for extensibility.
    pub(super) struct Boxed<const NBITS: usize>
    where
        Unsigned: Representation<NBITS>,
    {
        inner: Box<dyn for<'a> PreprocessedDistanceFunction<DataRef<'a, NBITS>, RF32>>,
    }

    impl<const NBITS: usize> Boxed<NBITS>
    where
        Unsigned: Representation<NBITS>,
    {
        fn new<T>(inner: T) -> Self
        where
            T: for<'a> PreprocessedDistanceFunction<DataRef<'a, NBITS>, RF32> + 'static,
        {
            Self {
                inner: Box::new(inner),
            }
        }
    }

    impl<const NBITS: usize> PreprocessedDistanceFunction<DataRef<'_, NBITS>> for Boxed<NBITS>
    where
        Unsigned: Representation<NBITS>,
    {
        fn evaluate_similarity(&self, x: DataRef<'_, NBITS>) -> f32 {
            self.inner.evaluate_similarity(x).unwrap()
        }
    }

    pub(super) struct Plan {
        measure: SimilarityMeasure,
        layout: MinMaxQuery,
    }

    impl<const NBITS: usize> algos::CreateQuantComputer<Store<NBITS>> for Plan
    where
        Unsigned: Representation<NBITS>,
        MinMaxL2Squared: for<'a, 'b> PureDistanceFunction<
                DataRef<'a, NBITS>,
                DataRef<'b, NBITS>,
                distances::Result<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                minmax::FullQueryRef<'a>,
                DataRef<'b, NBITS>,
                distances::Result<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                DataRef<'a, 8>,
                DataRef<'b, NBITS>,
                distances::Result<f32>,
            >,
        MinMaxIP: for<'a, 'b> PureDistanceFunction<
                DataRef<'a, NBITS>,
                DataRef<'b, NBITS>,
                distances::Result<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                minmax::FullQueryRef<'a>,
                DataRef<'b, NBITS>,
                distances::Result<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                DataRef<'a, NBITS>,
                DataRef<'b, NBITS>,
                distances::MathematicalResult<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                minmax::FullQueryRef<'a>,
                DataRef<'b, NBITS>,
                distances::MathematicalResult<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                DataRef<'a, 8>,
                DataRef<'b, NBITS>,
                distances::Result<f32>,
            > + for<'a, 'b> PureDistanceFunction<
                DataRef<'a, 8>,
                DataRef<'b, NBITS>,
                distances::MathematicalResult<f32>,
            >,
    {
        type Computer<'a> = Boxed<NBITS>;

        fn create_quant_computer<'a>(
            &self,
            store: &Store<NBITS>,
            query: &[f32],
        ) -> anyhow::Result<Self::Computer<'_>> {
            let quantizer = &store.quantizer;
            let output_dim = quantizer.output_dim();

            // Pair the freshly-compressed query with the right `MinMax*` distance functor
            // for `self.measure` and erase the concrete query type behind a `Boxed<NBITS>`.
            //
            // Implemented as a macro because expressing it as a generic function would need
            // higher-ranked bounds on `<Q as Reborrow<'a>>::Target` that the current trait
            // solver cannot discharge for the concrete `Q`s used below.
            macro_rules! box_for_measure {
                ($compressed:expr) => {
                    match self.measure {
                        SimilarityMeasure::SquaredL2 => {
                            let inner: MinMaxL2Squared = quantizer.as_functor();
                            Boxed::new(Curried::new(inner, $compressed))
                        }
                        SimilarityMeasure::InnerProduct => {
                            let inner: MinMaxIP = quantizer.as_functor();
                            Boxed::new(Curried::new(inner, $compressed))
                        }
                        SimilarityMeasure::Cosine => {
                            let inner: MinMaxCosine = quantizer.as_functor();
                            Boxed::new(Curried::new(inner, $compressed))
                        }
                        SimilarityMeasure::CosineNormalized => {
                            let inner: MinMaxCosineNormalized = quantizer.as_functor();
                            Boxed::new(Curried::new(inner, $compressed))
                        }
                    }
                };
            }

            match self.layout {
                MinMaxQuery::SameAsData => {
                    let mut compressed = Data::<NBITS>::new_boxed(output_dim);
                    quantizer.compress_into(query, compressed.reborrow_mut())?;
                    Ok(box_for_measure!(compressed))
                }
                MinMaxQuery::FullPrecision => {
                    let mut compressed = minmax::FullQuery::new_in(
                        output_dim,
                        diskann_quantization::alloc::GlobalAllocator,
                    )?;
                    quantizer.compress_into(query, compressed.reborrow_mut())?;
                    Ok(box_for_measure!(compressed))
                }
                MinMaxQuery::EightBit => {
                    let mut compressed = Data::<8>::new_boxed(output_dim);
                    quantizer.compress_into(query, compressed.reborrow_mut())?;
                    Ok(box_for_measure!(compressed))
                }
            }
        }
    }
}
