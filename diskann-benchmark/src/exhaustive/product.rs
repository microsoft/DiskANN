/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::Registry;

const NAME: &str = "product-exhaustive-search";

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    #[cfg(feature = "product-quantization")]
    registry.register(NAME, imp::ProductQ)?;

    #[cfg(not(feature = "product-quantization"))]
    registry.register_partially_gated::<crate::inputs::exhaustive::Product>(
        NAME,
        diskann_benchmark_runner::Features::new("product-quantization"),
        "Product quantization exhaustive search",
    )?;

    Ok(())
}

//////////////
// ProductQ //
//////////////

#[cfg(feature = "product-quantization")]
mod imp {
    use std::io::Write;

    use diskann_benchmark_runner::{
        benchmark::{MatchContext, Score},
        utils::{percentiles, MicroSeconds},
        Benchmark, Output,
    };
    use diskann_providers::model::pq::FixedChunkPQTable;
    use diskann_quantization::{
        product::{tables, train::TrainQuantizer},
        CompressInto,
    };
    use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
    use diskann_vector::distance::Metric;
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use serde::Serialize;

    use crate::{
        exhaustive::algos::{self, LinearSearch},
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
    #[derive(Debug, Clone, Copy)]
    pub(super) struct ProductQ;

    impl ProductQ {
        pub(super) fn run(
            &self,
            input: &inputs::exhaustive::Product,
            mut output: &mut dyn Output,
        ) -> anyhow::Result<Results> {
            writeln!(output, "{}", input)?;

            // Training
            let data = f32::converting_load(datafiles::BinFile(&input.data), input.data_type)?;
            let start = std::time::Instant::now();

            let parameters = diskann_quantization::product::train::LightPQTrainingParameters::new(
                input.num_pq_centers.get(),
                5,
            );

            let dim = std::num::NonZeroUsize::new(data.ncols())
                .ok_or_else(|| anyhow::anyhow!("data has zero columns"))?;
            let offsets =
                diskann_quantization::views::ChunkOffsets::partition(dim, input.num_pq_chunks)?;

            let base = {
                let threadpool = rayon::ThreadPoolBuilder::new()
                    .num_threads(input.compression_threads.get())
                    .build()?;
                threadpool.install(|| -> anyhow::Result<_> {
                    Ok(parameters.train(
                        data.as_view(),
                        offsets.as_view(),
                        diskann_quantization::Parallelism::Rayon,
                        &diskann_quantization::random::StdRngBuilder::new(input.seed),
                        &diskann_quantization::cancel::DontCancel,
                    )?)
                })?
            };

            // TODO: Training should return a `BasicTable` directly.
            let table = tables::BasicTable::new(
                rowmajor::Owned::try_from_data(
                    base.flatten().into(),
                    input.num_pq_centers.get(),
                    data.ncols(),
                )?,
                offsets,
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
                let store = threadpool.install(|| {
                    Store::new(
                        data.as_view(),
                        table,
                        input.table_style,
                        &compression_progress,
                    )
                })?;
                compression_progress.finish();
                store
            };
            let compression_time: MicroSeconds = start.elapsed().into();

            // Search
            let queries =
                f32::converting_load(datafiles::BinFile(&input.search.queries), input.data_type)?;

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth), None)?;

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

    impl Benchmark for ProductQ {
        type Input = inputs::exhaustive::Product;
        type Output = Results;

        fn try_match(&self, _input: &inputs::exhaustive::Product, context: &MatchContext) -> Score {
            context.success(0)
        }

        fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "- Exhaustive search for product quantization",)?;
            writeln!(f, "- Requires `float32` data")?;
            Ok(())
        }

        fn run(
            &self,
            input: &inputs::exhaustive::Product,
            _checkpoint: diskann_benchmark_runner::Checkpoint<'_>,
            output: &mut dyn Output,
        ) -> anyhow::Result<Results> {
            self.run(input, output)
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

    //------------//
    // Compressor //
    //------------//

    #[derive(Debug)]
    enum Compressor {
        FixedChunk(FixedChunkPQTable),
        Transposed(tables::TransposedTable),
    }

    impl Compressor {
        fn new(
            table: tables::BasicTable,
            style: inputs::exhaustive::PQTableStyle,
        ) -> anyhow::Result<Self> {
            use inputs::exhaustive::PQTableStyle;
            match style {
                PQTableStyle::FixedChunk => Ok(Self::FixedChunk(table.try_into()?)),
                PQTableStyle::Padded | PQTableStyle::Transposed => {
                    let table = tables::TransposedTable::from_parts(
                        table.view_pivots(),
                        table.view_offsets().to_owned(),
                    )?;

                    Ok(Self::Transposed(table))
                }
            }
        }

        fn compress(&self, storage: &mut [u8], data: &[f32]) -> anyhow::Result<()> {
            match self {
                Self::FixedChunk(table) => table.compress_into(data, storage)?,
                Self::Transposed(table) => table.compress_into(data, storage)?,
            }
            Ok(())
        }
    }

    //-----------//
    // Distances //
    //-----------//

    trait ComputerImpl: std::fmt::Debug {
        fn evaluate(&self, x: &[u8]) -> anyhow::Result<f32>;
    }

    #[derive(Debug)]
    struct Computer<'a>(Box<dyn ComputerImpl + 'a>);

    impl<'a> Computer<'a> {
        fn new<C>(inner: C) -> Self
        where
            C: ComputerImpl + 'a,
        {
            Self(Box::new(inner))
        }
    }

    impl diskann_vector::PreprocessedDistanceFunction<&[u8], f32> for Computer<'_> {
        fn evaluate_similarity(&self, x: &[u8]) -> f32 {
            match self.0.evaluate(x) {
                Ok(v) => v,
                Err(err) => panic!("distance failed with {:#}", err),
            }
        }
    }

    //-----------//
    // Computers //
    //-----------//

    impl ComputerImpl for diskann_providers::model::pq::distance::QueryComputer<'_> {
        fn evaluate(&self, x: &[u8]) -> anyhow::Result<f32> {
            Ok(<Self as diskann_vector::PreprocessedDistanceFunction<
                &[u8],
                f32,
            >>::evaluate_similarity(self, x))
        }
    }

    #[derive(Debug)]
    struct PaddedComputer<'a> {
        table: &'a tables::PaddedTable,
        vtable: tables::padded::VTable,
        query: Vec<f32>,
    }

    impl ComputerImpl for PaddedComputer<'_> {
        fn evaluate(&self, x: &[u8]) -> anyhow::Result<f32> {
            Ok(self.vtable.distance(self.table, &self.query, x)?)
        }
    }

    #[derive(Debug)]
    struct LookupTable {
        lookup: rowmajor::Owned<f32>,
    }

    impl ComputerImpl for LookupTable {
        fn evaluate(&self, x: &[u8]) -> anyhow::Result<f32> {
            Ok(tables::lookup::lookup_single(
                tables::lookup::Sum,
                self.lookup.as_view(),
                x,
            )?)
        }
    }

    #[derive(Debug)]
    struct CosineLookupTable {
        lookup: rowmajor::Owned<tables::lookup::DotAndNorm>,
        query_norm: f32,
    }

    impl ComputerImpl for CosineLookupTable {
        fn evaluate(&self, x: &[u8]) -> anyhow::Result<f32> {
            let partial =
                tables::lookup::lookup_single(tables::lookup::Sum, self.lookup.as_view(), x)?;
            Ok(partial.finish_cosine(self.query_norm).into_inner())
        }
    }

    #[derive(Debug)]
    enum Distance {
        FixedChunk(FixedChunkPQTable),
        Padded(tables::PaddedTable),
        Transposed(tables::TransposedTable),
    }

    impl Distance {
        fn new(
            basic: tables::BasicTable,
            style: inputs::exhaustive::PQTableStyle,
        ) -> anyhow::Result<Self> {
            use inputs::exhaustive::PQTableStyle;
            match style {
                PQTableStyle::FixedChunk => Ok(Self::FixedChunk(basic.try_into()?)),
                PQTableStyle::Padded => Ok(Self::Padded(tables::PaddedTable::from_basic(
                    basic.as_view(),
                ))),
                PQTableStyle::Transposed => {
                    Ok(Self::Transposed(tables::TransposedTable::from_parts(
                        basic.view_pivots(),
                        basic.view_offsets().to_owned(),
                    )?))
                }
            }
        }

        fn computer(&self, query: &[f32], metric: Metric) -> anyhow::Result<Computer<'_>> {
            match self {
                Self::FixedChunk(table) => {
                    let inner = diskann_providers::model::pq::distance::QueryComputer::new(
                        table.into(),
                        metric,
                        query,
                        None,
                    )?;
                    Ok(Computer::new(inner))
                }
                Self::Padded(table) => {
                    let inner = PaddedComputer {
                        table,
                        vtable: table.vtable(metric.into()),
                        query: query.into(),
                    };

                    Ok(Computer::new(inner))
                }
                Self::Transposed(table) => match metric {
                    Metric::L2 => {
                        let mut lookup =
                            rowmajor::Owned::from_element(table.nchunks(), table.ncenters(), 0.0);
                        table.process_into::<diskann_quantization::distances::SquaredL2, _>(
                            query,
                            lookup.as_view_mut(),
                        );
                        Ok(Computer::new(LookupTable { lookup }))
                    }
                    Metric::InnerProduct => {
                        let mut lookup =
                            rowmajor::Owned::from_element(table.nchunks(), table.ncenters(), 0.0);
                        table.process_into::<diskann_quantization::distances::InnerProduct, _>(
                            query,
                            lookup.as_view_mut(),
                        );
                        Ok(Computer::new(LookupTable { lookup }))
                    }
                    Metric::Cosine | Metric::CosineNormalized => {
                        let mut lookup = rowmajor::Owned::from_element(
                            table.nchunks(),
                            table.ncenters(),
                            tables::lookup::DotAndNorm::default(),
                        );

                        let query_norm = <_ as diskann_vector::Norm<&[f32]>>::evaluate(
                            &diskann_vector::norm::FastL2Norm,
                            query,
                        );

                        table.process_into::<diskann_quantization::distances::Cosine, _>(
                            query,
                            lookup.as_view_mut(),
                        );
                        Ok(Computer::new(CosineLookupTable { lookup, query_norm }))
                    }
                },
            }
        }
    }

    /// A store for quantized data.
    pub(super) struct Store {
        data: rowmajor::Owned<u8>,
        distance: Distance,
    }

    impl Store {
        fn new(
            input: rowmajor::Ref<f32>,
            table: tables::BasicTable,
            style: inputs::exhaustive::PQTableStyle,
            progress: &ProgressBar,
        ) -> anyhow::Result<Self> {
            let mut data = rowmajor::Owned::try_from_element(input.nrows(), table.nchunks(), 0)?;

            let compressor = Compressor::new(table.clone(), style)?;

            // Compress the data.
            #[expect(clippy::disallowed_methods)]
            data.par_rows_mut().zip(input.par_rows()).try_for_each(
                |(d, i)| -> anyhow::Result<()> {
                    compressor.compress(d, i)?;
                    progress.inc(1);
                    Ok(())
                },
            )?;

            let distance = Distance::new(table, style)?;
            Ok(Self { data, distance })
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
            self.data.rows()
        }
    }

    impl algos::CreateQuantComputer<Store> for Plan {
        type Computer<'a> = Computer<'a>;

        fn create_quant_computer<'a>(
            &self,
            store: &'a Store,
            query: &[f32],
        ) -> anyhow::Result<Self::Computer<'a>> {
            store.distance.computer(query, self.measure.into())
        }
    }
}
