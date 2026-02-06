/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::registry::Benchmarks;

const NAME: &str = "spherical-exhaustive-search";

crate::utils::stub_impl!("spherical-quantization", inputs::exhaustive::Spherical);

// Spherical - requires feature "spherical-quantization"
#[cfg(feature = "spherical-quantization")]
pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    benchmarks.register::<imp::SphericalQ<'static, 1>>(NAME, |object, _checkpoint, output| {
        match object.run(output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });

    benchmarks.register::<imp::SphericalQ<'static, 2>>(NAME, |object, _checkpoint, output| {
        match object.run(output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });

    benchmarks.register::<imp::SphericalQ<'static, 4>>(NAME, |object, _checkpoint, output| {
        match object.run(output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });

    benchmarks.register::<imp::SphericalQ<'static, 8>>(NAME, |object, _checkpoint, output| {
        match object.run(output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });
}

// Stub implementation
#[cfg(not(feature = "spherical-quantization"))]
pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    imp::register(NAME, benchmarks)
}

////////////////
// SphericalQ //
////////////////

#[cfg(feature = "spherical-quantization")]
mod imp {
    use std::io::Write;

    use diskann_benchmark_runner::{
        describeln,
        dispatcher::{self, DispatchRule, FailureScore, MatchScore},
        output::Output,
        utils::{percentiles, MicroSeconds},
        Any,
    };
    use diskann_providers::model::graph::provider::async_::distances::UnwrapErr;
    use diskann_quantization::{
        alloc::{GlobalAllocator, ScopedAllocator},
        bits::{Representation, Unsigned},
        spherical::{DataMut, SphericalQuantizer},
        CompressIntoWith,
    };
    use indicatif::{ProgressBar, ProgressStyle};
    use rand::SeedableRng;
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
    pub(super) struct SphericalQ<'a, const NBITS: usize> {
        input: &'a inputs::exhaustive::Spherical,
    }

    impl<'a, const NBITS: usize> SphericalQ<'a, NBITS> {
        pub(super) fn new(input: &'a inputs::exhaustive::Spherical) -> Self {
            Self { input }
        }

        pub(super) fn run(self, mut output: &mut dyn Output) -> anyhow::Result<Results>
        where
            Unsigned: Representation<NBITS>,
            Plan: algos::CreateQuantComputer<Store<NBITS>>,
            diskann_quantization::spherical::iface::Impl<NBITS>:
                diskann_quantization::spherical::iface::Constructible,
            SphericalQuantizer:
                for<'x> CompressIntoWith<&'x [f32], DataMut<'x, NBITS>, ScopedAllocator<'x>>,
        {
            let input = &self.input;
            writeln!(output, "{}", self.input)?;

            // Training
            let data = f32::converting_load(datafiles::BinFile(&input.data), input.data_type)?;
            let start = std::time::Instant::now();
            let metric: diskann_vector::distance::Metric = input.distance.into();
            let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
                data.as_view(),
                (&input.transform_kind).into(),
                metric.try_into()?,
                input.pre_scale.try_into()?,
                &mut rand::rngs::StdRng::seed_from_u64(input.seed),
                GlobalAllocator,
            )?;

            let training_time: MicroSeconds = start.elapsed().into();

            // Compressing
            let start = std::time::Instant::now();
            let store = {
                let compression_progress =
                    make_progress_bar("compressing", data.nrows(), output.draw_target())?;
                let store = Store::new(
                    data.as_view(),
                    diskann_quantization::spherical::iface::Impl::<NBITS>::new(quantizer)?,
                    &compression_progress,
                )?;
                compression_progress.finish();
                store
            };
            let compression_time: MicroSeconds = start.elapsed().into();

            // Search
            let queries =
                f32::converting_load(datafiles::BinFile(&input.search.queries), input.data_type)?;

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth))?;

            let search_progress = make_progress_bar(
                "running search",
                input.query_layouts.len() * queries.nrows(),
                output.draw_target(),
            )?;

            let mut search_results = Vec::<SearchResults>::new();
            let threadpool = rayon::ThreadPoolBuilder::new()
                .num_threads(input.search.num_threads.get())
                .build()?;

            let recall_n = input
                .search
                .recalls
                .recall_n
                .last()
                .ok_or_else(|| anyhow::anyhow!("expected at least one value for `recall_n`"))?;

            for layout in input.query_layouts.iter() {
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
                    *layout,
                    input.search.num_threads.get(),
                    recalls,
                )?);
            }
            search_progress.finish();

            // Aggregate and print results.
            let result = Results {
                training_time,
                compression_time,
                search_results,
                quantized_dim: store.quantized_dim(),
                quantized_bytes: store.quantized_bytes(),
                original_dim: data.ncols(),
            };

            writeln!(output, "\n\n{}", result)?;
            Ok(result)
        }
    }

    impl<const NBITS: usize> dispatcher::Map for SphericalQ<'static, NBITS> {
        type Type<'a> = SphericalQ<'a, NBITS>;
    }

    impl<'a, const NBITS: usize> DispatchRule<&'a inputs::exhaustive::Spherical>
        for SphericalQ<'a, NBITS>
    {
        type Error = std::convert::Infallible;

        fn try_match(from: &&'a inputs::exhaustive::Spherical) -> Result<MatchScore, FailureScore> {
            let num_bits = from.num_bits.get();
            if num_bits == NBITS {
                Ok(MatchScore(0))
            } else {
                Err(FailureScore(
                    NBITS.abs_diff(num_bits).try_into().unwrap_or(u32::MAX),
                ))
            }
        }

        fn convert(from: &'a inputs::exhaustive::Spherical) -> Result<Self, Self::Error> {
            assert_eq!(
                from.num_bits.get(),
                NBITS,
                "This should not have occurred. Please file a bug report"
            );
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a inputs::exhaustive::Spherical>,
        ) -> std::fmt::Result {
            match from {
                None => {
                    describeln!(
                        f,
                        "- Exhaustive search for {}-bit spherical quantization",
                        NBITS
                    )?;
                    describeln!(f, "- Requires `float32` data")?;
                    describeln!(f, "- Implements `squared_l2` or `inner_product` distance")?;
                }
                Some(from) => {
                    if from.num_bits.get() != NBITS {
                        describeln!(
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
    }

    impl<'a, const NBITS: usize> DispatchRule<&'a Any> for SphericalQ<'a, NBITS> {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<inputs::exhaustive::Spherical, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<inputs::exhaustive::Spherical, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<inputs::exhaustive::Spherical, Self>(
                f,
                from,
                inputs::exhaustive::Spherical::tag(),
            )
        }
    }

    /// Results from an end-to-end run of Spherical Quantization.
    #[derive(Debug, Serialize)]
    pub(super) struct Results {
        /// The time it takes to generate the base quantizer.
        training_time: MicroSeconds,
        /// How long it takes to compress the raw data.
        compression_time: MicroSeconds,
        /// The effective dimensionality of the post-transformed data.
        ///
        /// This dimensionality can be higher or lower than the original dimensionality,
        /// depending on the mechanics of the distance-preserving transform.
        quantized_dim: usize,
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
            write_field!(f, "Quantized Dim", "{}", self.quantized_dim)?;
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
        layout: inputs::exhaustive::SphericalQuery,

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
            layout: inputs::exhaustive::SphericalQuery,
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
            write_field!(f, "Query Layout", "{}", self.layout)?;
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

            let mut table = diskann_benchmark_runner::utils::fmt::Table::new(
                ["K", "N", "Recall (%)"],
                self.recalls.len(),
            );
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
        plan: diskann_quantization::spherical::iface::Impl<NBITS>,
    }

    impl<const NBITS: usize> Store<NBITS>
    where
        Unsigned: Representation<NBITS>,
        SphericalQuantizer:
            for<'a> CompressIntoWith<&'a [f32], DataMut<'a, NBITS>, ScopedAllocator<'a>>,
    {
        fn new(
            input: diskann_utils::views::MatrixView<f32>,
            plan: diskann_quantization::spherical::iface::Impl<NBITS>,
            progress: &ProgressBar,
        ) -> anyhow::Result<Self> {
            // Both the dimensionality of the compressed vector and the exact number of bytes
            // needed to store it can vary.
            //
            // The APIs below should correctly handle these variables.
            let output_dim = plan.quantizer().output_dim();
            let bytes =
                diskann_quantization::spherical::DataRef::<NBITS>::canonical_bytes(output_dim);
            let mut data = diskann_utils::views::Matrix::new(0, input.nrows(), bytes);

            // Compress the data.
            #[allow(clippy::disallowed_methods)]
            data.par_row_iter_mut()
                .zip(input.par_row_iter())
                .try_for_each(|(d, i)| -> anyhow::Result<()> {
                    let c =
                        diskann_quantization::spherical::DataMut::<NBITS>::from_canonical_back_mut(
                            &mut d[..bytes],
                            output_dim,
                        )?;
                    plan.quantizer()
                        .compress_into_with(i, c, ScopedAllocator::global())?;
                    progress.inc(1);
                    Ok(())
                })?;

            Ok(Self { bytes, data, plan })
        }

        /// Return the effective dimensionality of the compressed data.
        fn quantized_dim(&self) -> usize {
            self.plan.quantizer().output_dim()
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
            = diskann_quantization::spherical::iface::Opaque<'a>
        where
            Self: 'a;

        fn iter(&self) -> impl Iterator<Item = Self::Item<'_>> {
            self.data
                .row_iter()
                .map(move |r| diskann_quantization::spherical::iface::Opaque::new(&r[..self.bytes]))
        }
    }

    /// Exhaustive searching depends both on the similarity measure used and the number
    /// of bits use to compress the query.
    ///
    /// This struct uses a factor pattern to realize a distance computer for each valid
    /// combination.
    pub(super) struct Plan {
        measure: SimilarityMeasure,
        layout: inputs::exhaustive::SphericalQuery,
    }

    impl<const NBITS: usize> algos::CreateQuantComputer<Store<NBITS>> for Plan
    where
        Unsigned: Representation<NBITS>,
        diskann_quantization::spherical::iface::Impl<NBITS>:
            diskann_quantization::spherical::iface::Quantizer,
    {
        type Computer<'a> = UnwrapErr<
            diskann_quantization::spherical::iface::QueryComputer,
            diskann_quantization::spherical::iface::QueryDistanceError,
        >;

        fn create_quant_computer<'a>(
            &'a self,
            store: &'a Store<NBITS>,
            query: &[f32],
        ) -> anyhow::Result<Self::Computer<'a>> {
            use diskann_quantization::spherical::iface::Quantizer;

            let allow_rescale = self.measure == SimilarityMeasure::InnerProduct;
            Ok(store
                .plan
                .fused_query_computer(
                    query,
                    self.layout.into(),
                    allow_rescale,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .map(UnwrapErr::new)?)
        }
    }
}
