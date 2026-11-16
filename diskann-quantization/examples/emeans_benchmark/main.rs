/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod data;
mod metrics;
mod train;

use diskann_utils::views::rowmajor::Matrix;
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fs::File,
    io::{self, BufWriter, Write},
    time::Instant,
};
use train::{Algorithm, Mode, Run};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    seeds: Vec<u64>,
    repeats: usize,
    output: String,
    runs: Vec<Run>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    mode: Mode,
    train: String,
    test: String,
    train_rows: usize,
    test_rows: usize,
    clusters: usize,
    partitions: usize,
    database: Option<String>,
    queries: Option<String>,
    query_rows: usize,
}

#[derive(Clone, Copy, Serialize)]
struct Quality {
    train_mse: f64,
    test_mse: f64,
    recall_at10: Option<f64>,
}

#[derive(Serialize)]
struct Measurement<'a> {
    case: &'a str,
    mode: Mode,
    seed: u64,
    repeat: usize,
    run_order: usize,
    algorithm: Algorithm,
    epochs: usize,
    train_rows: usize,
    test_rows: usize,
    dimensions: usize,
    partitions: usize,
    clusters: usize,
    database_rows: Option<usize>,
    query_rows: usize,
    seconds: f64,
    batch_size: Option<usize>,
    initial_step: Option<f64>,
    revived_clusters: usize,
    #[serde(flatten)]
    quality: Quality,
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: emeans_benchmark CONFIG.json")?;
    let config: Config = serde_json::from_reader(File::open(path)?)?;
    if config.seeds.is_empty()
        || config.repeats == 0
        || config.runs.is_empty()
        || config.runs.iter().any(|r| r.epochs == 0)
    {
        return Err("invalid run configuration".into());
    }
    let mut output = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(&config.output)?,
    );
    for case in &config.cases {
        let train_data = data::load(&case.train)?;
        let test_data = data::load(&case.test)?;
        let training = data::prefix(train_data.as_view(), case.train_rows)?;
        let testing = data::prefix(test_data.as_view(), case.test_rows)?;
        if training.ncols() != testing.ncols()
            || case.clusters == 0
            || case.clusters > training.nrows()
            || case.partitions == 0
            || !training.ncols().is_multiple_of(case.partitions)
            || (case.mode == Mode::Clustering && case.partitions != 1)
            || (case.mode == Mode::Pq && case.clusters > 256)
            || case.database.is_some() != case.queries.is_some()
        {
            return Err(io::Error::other(format!("invalid case {}", case.name)).into());
        }
        let database = case.database.as_deref().map(data::load).transpose()?;
        let queries = case.queries.as_deref().map(data::load).transpose()?;
        let query_view = queries
            .as_ref()
            .map(|q| data::prefix(q.as_view(), case.query_rows))
            .transpose()?;
        let truth = if let (Some(database), Some(queries)) = (&database, query_view) {
            if case.mode != Mode::Pq
                || database.ncols() != training.ncols()
                || queries.ncols() != training.ncols()
                || database.nrows() < 10
            {
                return Err("invalid recall inputs".into());
            }
            Some(metrics::ground_truth(database.as_view(), queries, 10))
        } else {
            None
        };
        let warmup = data::prefix(training, training.nrows().min(2048))?;
        for algorithm in [Algorithm::Lloyd, Algorithm::Emeans] {
            train::fit(
                warmup,
                case.mode,
                case.clusters.min(32),
                case.partitions,
                Run {
                    algorithm,
                    epochs: 2,
                },
                0,
            )?;
        }
        for &seed in &config.seeds {
            // Repeat quality only once per deterministic codebook; verify every repeat is bit-identical.
            let mut reference: Vec<Option<(Vec<f32>, Quality)>> = vec![None; config.runs.len()];
            for repeat in 0..config.repeats {
                let mut order: Vec<_> = (0..config.runs.len()).collect();
                order.shuffle(&mut StdRng::seed_from_u64(
                    seed ^ (repeat as u64).wrapping_mul(1_000_003),
                ));
                for (run_order, index) in order.into_iter().enumerate() {
                    let run = config.runs[index];
                    let begin = Instant::now();
                    let trained = train::fit(
                        training,
                        case.mode,
                        case.clusters,
                        case.partitions,
                        run,
                        seed,
                    )?;
                    let seconds = begin.elapsed().as_secs_f64();
                    let quality = if let Some((centers, quality)) = &reference[index] {
                        if !centers
                            .iter()
                            .zip(trained.table.view_pivots().as_slice())
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                        {
                            return Err("non-deterministic codebook between timing repeats".into());
                        }
                        *quality
                    } else {
                        let (train_mse, test_mse) = if case.mode == Mode::Clustering {
                            (
                                metrics::clustering_mse(training, &trained.table),
                                metrics::clustering_mse(testing, &trained.table),
                            )
                        } else {
                            (
                                metrics::encode(training, &trained.table)?.1,
                                metrics::encode(testing, &trained.table)?.1,
                            )
                        };
                        let recall_at10 = if let (Some(database), Some(queries), Some(truth)) =
                            (&database, query_view, &truth)
                        {
                            let codes = metrics::encode(database.as_view(), &trained.table)?.0;
                            Some(metrics::recall(&codes, queries, &trained.table, truth))
                        } else {
                            None
                        };
                        let quality = Quality {
                            train_mse,
                            test_mse,
                            recall_at10,
                        };
                        reference[index] =
                            Some((trained.table.view_pivots().as_slice().to_vec(), quality));
                        quality
                    };
                    let measurement = Measurement {
                        case: &case.name,
                        mode: case.mode,
                        seed,
                        repeat,
                        run_order,
                        algorithm: run.algorithm,
                        epochs: run.epochs,
                        train_rows: training.nrows(),
                        test_rows: testing.nrows(),
                        dimensions: training.ncols(),
                        partitions: case.partitions,
                        clusters: case.clusters,
                        database_rows: database.as_ref().map(|d| d.nrows()),
                        query_rows: case.query_rows,
                        seconds,
                        batch_size: trained.batch_size,
                        initial_step: trained.initial_step,
                        revived_clusters: trained.revived_clusters,
                        quality,
                    };
                    serde_json::to_writer(&mut output, &measurement)?;
                    writeln!(output)?;
                    output.flush()?;
                    eprintln!("{}", serde_json::to_string(&measurement)?);
                }
            }
        }
    }
    Ok(())
}
