/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Release microbenchmark using the checked-in disk search fixture.
//! Usage: cargo run --release -p diskann-disk --example search_diagnostics -- off cached 100000
//! Replace `off` with `on`, or `cached` with `storage`. Output is CSV.
//! Includes span-close text formatting to a sink, but no terminal/disk/network export.

use std::{hint::black_box, time::Instant};

use diskann_disk::{
    data_model::{AdHoc, CachingStrategy},
    search::{
        provider::{
            disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        search_mode::SearchMode,
        traits::{VertexProvider, VertexProviderFactory},
    },
    storage::disk_index_reader::DiskIndexReader,
};
use diskann_providers::storage::{
    FileStorageProvider, get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file,
};
use diskann_utils::{
    io::read_bin,
    views::rowmajor::{self, Matrix},
};
use diskann_vector::distance::Metric;
use tracing::Instrument;
use tracing_subscriber::{
    filter::{LevelFilter, Targets},
    fmt::format::FmtSpan,
    prelude::*,
};

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 3,
        "usage: search_diagnostics <off|on> <cached|storage> <queries>"
    );
    let enabled = match args[0].as_str() {
        "off" => false,
        "on" => true,
        value => anyhow::bail!("unknown tracing mode: {value}"),
    };
    let caching = match args[1].as_str() {
        "cached" => CachingStrategy::StaticCacheWithBfsNodes(256),
        "storage" => CachingStrategy::None,
        value => anyhow::bail!("unknown cache mode: {value}"),
    };
    let iterations: usize = args[2].parse()?;
    anyhow::ensure!(iterations > 0, "queries must be positive");
    let targets = Targets::new().with_default(LevelFilter::TRACE).with_target(
        "diskann::disk::search::diagnostics",
        if enabled {
            LevelFilter::DEBUG
        } else {
            LevelFilter::OFF
        },
    );
    let subscriber = tracing_subscriber::registry().with(targets).with(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::sink),
    );
    let _subscriber = tracing::subscriber::set_default(subscriber);

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test_data")
        .join("disk_index_search");
    let prefix = root
        .join("disk_index_sift_learn_R4_L50_A1.2_truth_search")
        .to_string_lossy()
        .into_owned();
    let reader = DiskIndexReader::new(
        get_pq_pivot_file(&prefix),
        get_compressed_pq_file(&prefix),
        &FileStorageProvider,
    )?;
    let factory =
        DiskVertexProviderFactory::from_disk_index_path(get_disk_index_file(&prefix), caching)?;
    let header = factory.get_header()?;
    let mut ids = vec![header.metadata().medoid as u32];
    let mut visited = vec![false; header.metadata().num_pts as usize];
    visited[ids[0] as usize] = true;
    let mut probe = factory.create_vertex_provider(1, &header)?;
    let mut position = 0;
    while position < ids.len() {
        let id = ids[position];
        probe.load_vertices(&[id])?;
        probe.process_loaded_node(&id, 0)?;
        for &neighbor in probe.get_adjacency_list(&id)? {
            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                ids.push(neighbor);
            }
        }
        position += 1;
    }
    let read_ops = probe.io_operations();
    anyhow::ensure!(
        (args[1] == "cached" && read_ops == 0) || (args[1] == "storage" && read_ops > 0),
        "cache preflight failed: {read_ops} native read operations"
    );
    eprintln!(
        "cache preflight: {} reachable nodes, {read_ops} native read operations",
        ids.len()
    );
    drop(probe);
    let searcher =
        DiskIndexSearcher::<AdHoc<f32>, _>::new(1, usize::MAX, &reader, factory, Metric::L2)?;
    let queries: rowmajor::Owned<f32> = read_bin(&mut std::fs::File::open(
        root.join("disk_index_sample_query_10pts.fbin"),
    )?)?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let run = |i: usize| {
        runtime.block_on(
            searcher
                .search(
                    queries.row(i % queries.nrows()),
                    10,
                    20,
                    Some(4),
                    SearchMode::graph(),
                )
                .instrument(tracing::info_span!(
                    "request",
                    k = 10,
                    l = 20,
                    beam_width = 4
                )),
        )
    };
    for i in 0..1000 {
        black_box(run(i)?);
    }
    let mut latencies = Vec::with_capacity(iterations);
    let mut comparisons = 0u64;
    let mut hops = 0u64;
    let mut checksum = 0u64;
    let start = Instant::now();
    for i in 0..iterations {
        let query_start = Instant::now();
        let result = run(i)?;
        latencies.push(query_start.elapsed().as_nanos() as u64);
        comparisons += u64::from(result.stats.cmps);
        hops += u64::from(result.stats.query_statistics.search_hops);
        for r in &result.results {
            checksum =
                checksum.wrapping_add(u64::from(r.vertex_id) + u64::from(r.distance.to_bits()));
        }
        black_box(result);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mean_us = latencies.iter().sum::<u64>() as f64 / iterations as f64 / 1000.0;
    latencies.sort_unstable();
    let percentile = |p: f64| {
        latencies[((iterations as f64 * p).ceil() as usize).saturating_sub(1)] as f64 / 1000.0
    };
    println!("tracing,cache,queries,qps,mean_us,p50_us,p95_us,p99_us,p999_us,cmps,hops,checksum");
    println!(
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{}",
        args[0],
        args[1],
        iterations,
        iterations as f64 / elapsed,
        mean_us,
        percentile(0.5),
        percentile(0.95),
        percentile(0.99),
        percentile(0.999),
        comparisons,
        hops,
        checksum,
    );
    Ok(())
}
