/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_providers::storage::FileStorageProvider;
use diskann_tools::utils::{
    build_pq, get_num_threads, init_subscriber, BuildPQParameters, CMDToolError, DataType,
};
use diskann_vector::distance::Metric;
use diskann_vector::Half;
use tracing::{error, info};

fn main() -> Result<(), CMDToolError> {
    init_subscriber();

    let args: BuildPQArgs = BuildPQArgs::parse();

    let threads = get_num_threads(args.num_threads);

    let storage_provider = FileStorageProvider;

    let parameters = BuildPQParameters {
        metric: args.dist_fn,
        data_path: &args.data_path,
        index_path_prefix: &args.index_path_prefix,
        num_threads: threads,
        p_val: args.p_val,
        pq_bytes: args.pq_bytes as f64,
    };

    let err = match args.data_type {
        DataType::Int8 => build_pq::<i8>(&storage_provider, parameters),
        DataType::Uint8 => build_pq::<u8>(&storage_provider, parameters),
        DataType::Float => build_pq::<f32>(&storage_provider, parameters),
        DataType::Fp16 => build_pq::<Half>(&storage_provider, parameters),
    };

    match err {
        Ok(_) => {
            info!("PQ build completed successfully");
            Ok(())
        }
        Err(err) => {
            error!("PQ build failed - see diagnostic");
            Err(err.into())
        }
    }
}

#[derive(Debug, Parser)]
struct BuildPQArgs {
    /// data type <int8/uint8/float / fp16> (required)
    #[arg(long = "data-type", default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist-fn", default_value = "l2")]
    pub dist_fn: Metric,

    /// Path to the data file. The file should be in the format specified by the `data_type` argument.
    #[arg(long = "data-file", short, required = true)]
    pub data_path: String,

    /// Path to the index file. The index will be saved to this prefixed name.
    #[arg(long = "index-path-prefix", short, required = true)]
    pub index_path_prefix: String,

    /// Number of threads to use.
    #[arg(long = "num-threads", short = 'T')]
    pub num_threads: Option<usize>,

    /// Ratio of PQ training set size to data size
    #[arg(long = "p-val", short = 'p', default_value = "0.1")]
    pub p_val: f64,

    /// Number of PQ bytee
    #[arg(long = "pq-bytes", short, default_value = "10")]
    pub pq_bytes: usize,
}
