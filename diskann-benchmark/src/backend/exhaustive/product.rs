/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

const NAME: &str = "product-exhaustive-search";

crate::utils::stub_impl!("product-quantization", inputs::exhaustive::Product);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    #[cfg(feature = "product-quantization")]
    benchmarks.register::<imp::ProductQ<'static>>(NAME, |object, _checkpoint, output| match object
        .run(output)
    {
        Ok(v) => Ok(serde_json::to_value(v)?),
        Err(err) => Err(err),
    });

    #[cfg(not(feature = "product-quantization"))]
    imp::register(NAME, benchmarks)
}

//////////////
// ProductQ //
//////////////

#[cfg(feature = "product-quantization")]
mod imp {
    use std::io::Write;

    use diskann_benchmark_runner::{
        describeln,
        dispatcher::{self, DispatchRule, FailureScore, MatchScore},
        utils::{percentiles, MicroSeconds},
        Any, Output,
    };
    use diskann_quantization::{product::train::TrainQuantizer, CompressInto};
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use serde::Serialize;

    use crate::{
        backend::exhaustive::algos::{self, LinearSearch},
        inputs,
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

    /// The dispatcher target for `spherical-quantization` operations.
    pub(super) struct ProductQ<'a> {
        input: &'a inputs::exhaustive::Product,
    }

    impl<'a> ProductQ<'a> {
        pub(super) fn new(input: &'a inputs::exhaustive::Product) -> Self {
            Self { input }
        }

        pub(super) fn run(self, mut output: &mut dyn Output) -> anyhow::Result<Results> {
            let input = &self.input;
            writeln!(output, "{}", self.input)?;

            // Training
            let data = f32::converting_load(datafiles::BinFile(&input.data), input.data_type)?;
            let start = std::time::Instant::now();

            let parameters = diskann_quantization::product::train::LightPQTrainingParameters::new(
                input.num_pq_centers.get(),
                5,
            );

            let offsets = diskann_providers::model::pq::calculate_chunk_offsets_auto(
                data.ncols(),
                input.num_pq_chunks.get(),
            );

            let base = {
                let threadpool = rayon::ThreadPoolBuilder::new()
                    .num_threads(input.compression_threads.get())
                    .build()?;
                threadpool.install(|| -> anyhow::Result<_> {
                    Ok(parameters.train(
                        data.as_view(),
                        diskann_quantization::views::ChunkOffsetsView::new(offsets.as_slice())?,
                        diskann_quantization::Parallelism::Rayon,
                        &diskann_quantization::random::StdRngBuilder::new(input.seed),
                        &diskann_quantization::cancel::DontCancel,
                    )?)
                })?
            };

            let quantizer = diskann_providers::model::pq::FixedChunkPQTable::new(
                data.ncols(),
                base.flatten().into(),
                vec![0.0; data.ncols()].into(),
                offsets.into(),
                None,
            )?;

            let training_time: MicroSeconds = start.elapsed().into();

            // Compressing
            let start = std::time::Instant::now();
            let store = {
                let threadpool = rayon::ThreadPoolBuilder::new()
                    .num_threads(input.compression_threads.get())
                    .build()?;

                let compression_progress =
                    make_progress_bar("compressing", data.nrows(), output.draw_target())?;
                let store = threadpool
                    .install(|| Store::new(data.as_view(), quantizer, &compression_progress))?;
                compression_progress.finish();
                store
            };
            let compression_time: MicroSeconds = start.elapsed().into();

            // Search
            let queries =
                f32::converting_load(datafiles::BinFile(&input.search.queries), input.data_type)?;

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth))?;

            let search_progress =
                make_progress_bar("running search", queries.nrows(), output.draw_target())?;

            let threadpool = rayon::ThreadPoolBuilder::new()
                .num_threads(input.search.num_threads.get())
                .build()?;

            let recall_n = input
                .search
                .recalls
                .recall_n
                .last()
                .ok_or_else(|| anyhow::anyhow!("expected at least one value for `recall_n`"))?;

            let plan = Plan {
                measure: input.distance,
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

            let search_results = SearchResults::new(r, input.search.num_threads.get(), recalls)?;

            search_progress.finish();

            // Aggregate and print results.
            let result = Results {
                training_time,
                compression_time,
                search_results,
            };

            writeln!(output, "\n\n{}", result)?;
            Ok(result)
        }
    }

    impl dispatcher::Map for ProductQ<'static> {
        type Type<'a> = ProductQ<'a>;
    }

    impl<'a> DispatchRule<&'a inputs::exhaustive::Product> for ProductQ<'a> {
        type Error = std::convert::Infallible;

        fn try_match(_from: &&'a inputs::exhaustive::Product) -> Result<MatchScore, FailureScore> {
            Ok(MatchScore(0))
        }

        fn convert(from: &'a inputs::exhaustive::Product) -> Result<Self, Self::Error> {
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a inputs::exhaustive::Product>,
        ) -> std::fmt::Result {
            if from.is_none() {
                describeln!(f, "- Exhaustive search for product quantization",)?;
                describeln!(f, "- Requires `float32` data")?;
            }
            Ok(())
        }
    }

    impl<'a> DispatchRule<&'a Any> for ProductQ<'a> {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<inputs::exhaustive::Product, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<inputs::exhaustive::Product, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<inputs::exhaustive::Product, Self>(
                f,
                from,
                inputs::exhaustive::Product::tag(),
            )
        }
    }

    /// Results from an end-to-end run of Product Quantization.
    #[derive(Debug, Serialize)]
    pub(super) struct Results {
        /// The time it takes to generate the base quantizer.
        training_time: MicroSeconds,
        /// How long it takes to compress the raw data.
        compression_time: MicroSeconds,
        /// Results for each search kind (varying over query layouts).
        search_results: SearchResults,
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
            writeln!(f, "{}", self.search_results)?;

            Ok(())
        }
    }

    #[derive(Debug, Serialize)]
    struct SearchResults {
        num_threads: usize,
        time: MicroSeconds,
        qps: f64,

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
            recalls: Vec<recall::RecallMetrics>,
        ) -> Result<Self, percentiles::CannotBeEmpty> {
            let preprocess_latency = percentiles::compute_percentiles(&mut search.preprocess)?;
            let search_latency = percentiles::compute_percentiles(&mut search.search)?;

            let time = search.total;
            Ok(Self {
                num_threads,
                time,
                qps: (search.ids.nrows() as f64) / time.as_seconds(),
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
    pub(super) struct Store {
        data: diskann_utils::views::Matrix<u8>,
        quantizer: diskann_providers::model::pq::FixedChunkPQTable,
    }

    impl Store {
        fn new(
            input: diskann_utils::views::MatrixView<f32>,
            quantizer: diskann_providers::model::pq::FixedChunkPQTable,
            progress: &ProgressBar,
        ) -> anyhow::Result<Self> {
            let mut data =
                diskann_utils::views::Matrix::new(0, input.nrows(), quantizer.get_num_chunks());

            // Compress the data.
            #[allow(clippy::disallowed_methods)]
            data.par_row_iter_mut()
                .zip(input.par_row_iter())
                .try_for_each(|(d, i)| -> anyhow::Result<()> {
                    quantizer.compress_into(i, d)?;
                    progress.inc(1);
                    Ok(())
                })?;

            Ok(Self { data, quantizer })
        }
    }

    struct Plan {
        measure: SimilarityMeasure,
    }

    impl algos::QuantStore for Store {
        type Item<'a>
            = &'a [u8]
        where
            Self: 'a;

        fn iter(&self) -> impl Iterator<Item = Self::Item<'_>> {
            self.data.row_iter()
        }
    }

    impl algos::CreateQuantComputer<Store> for Plan {
        type Computer<'a> = diskann_providers::model::pq::distance::QueryComputer<
            &'a diskann_providers::model::pq::FixedChunkPQTable,
        >;

        fn create_quant_computer<'a>(
            &self,
            store: &'a Store,
            query: &[f32],
        ) -> anyhow::Result<Self::Computer<'a>> {
            Ok(diskann_providers::model::pq::distance::QueryComputer::new(
                &store.quantizer,
                self.measure.into(),
                query,
                None,
            )?)
        }
    }
}
