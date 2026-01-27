/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_providers::storage::FileStorageProvider;
use diskann_providers::utils::random;
use diskann_tools::utils::{
    init_subscriber, relative_contrast::compute_relative_contrast, CMDResult, DataType,
    GraphDataF32Vector, GraphDataHalfVector, GraphDataInt8Vector, GraphDataU8Vector,
};

/// Command Usage:
///
/// Description:
/// This program calculates the relative contrast based on the provided inputs.
///  (More details is in https://www.ee.columbia.edu/~jh2700/ICML_NNSDifficulty.pdf)
/// The required inputs include:
/// - Raw vector dataset
/// - Query vector dataset
/// - Ground truth data
/// - Number of samples
/// - Desired number of returned vector searches
///
/// Expected Behavior:
/// - For a dataset that is well-suited for Approximate Nearest Neighbor (ANN) search,
///   the relative contrast is expected to be greater than 1.5.
/// - For a dataset that is not well-suited, the relative contrast approaches 1.0.
///
/// Example Command:
/// cargo run --bin relative_contrast -- --data_type fp16 --data_file data.bin
/// --query_file query.bin --gt_file gt.bin
/// --recall_at 1000 --search_list 1000
fn main() -> CMDResult<()> {
    init_subscriber();

    let args = RelativeContrastArgs::parse();
    let storage_provider = FileStorageProvider;
    let mut rng = random::create_rnd();

    let result = match args.data_type {
        DataType::Float => compute_relative_contrast::<GraphDataF32Vector, _, _>(
            &storage_provider,
            &args.data_file,
            &args.query_file,
            &args.gt_file,
            args.recall_at,
            args.search_list,
            &mut rng,
        ),
        DataType::Fp16 => compute_relative_contrast::<GraphDataHalfVector, _, _>(
            &storage_provider,
            &args.data_file,
            &args.query_file,
            &args.gt_file,
            args.recall_at,
            args.search_list,
            &mut rng,
        ),
        DataType::Uint8 => compute_relative_contrast::<GraphDataU8Vector, _, _>(
            &storage_provider,
            &args.data_file,
            &args.query_file,
            &args.gt_file,
            args.recall_at,
            args.search_list,
            &mut rng,
        ),
        DataType::Int8 => compute_relative_contrast::<GraphDataInt8Vector, _, _>(
            &storage_provider,
            &args.data_file,
            &args.query_file,
            &args.gt_file,
            args.recall_at,
            args.search_list,
            &mut rng,
        ),
    };

    match result {
        Ok(_) => {
            tracing::info!("Relative contrast computation completed successfully");
            Ok(())
        }
        Err(err) => {
            tracing::error!("Error: {:?}", err);
            Err(err)
        }
    }
}

#[derive(Debug, Parser)]
struct RelativeContrastArgs {
    /// Data type <int8/uint8/float/fp16>
    #[arg(long = "data_type", default_value = "fp16")]
    pub data_type: DataType,

    /// Vector data file path
    #[arg(long = "data_file", short, required = true)]
    pub data_file: String,

    /// Query file in binary format
    #[arg(long = "query_file", short, required = true)]
    pub query_file: String,

    /// Ground truth file for the queryset
    #[arg(long = "gt_file", required = true)]
    pub gt_file: String,

    /// Number of neighbors to use from ground truth
    #[arg(long = "recall_at", short = 'K', default_value = "10")]
    pub recall_at: usize,

    /// Number of random distances to average per query
    #[arg(long = "search_list", short = 'L', default_value = "10")]
    pub search_list: usize,
}
