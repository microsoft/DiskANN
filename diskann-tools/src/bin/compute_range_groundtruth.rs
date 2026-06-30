/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use clap::Parser;
use diskann_providers::{storage::FileStorageProvider, utils::Timer};
use diskann_tools::utils::{
    compute_range_ground_truth_from_datafiles, init_subscriber, CMDResult, DataType,
    GraphDataF32Vector, GraphDataHalfVector, GraphDataInt8Vector, GraphDataU8Vector,
};
use diskann_vector::distance::Metric;

fn main() -> CMDResult<()> {
    init_subscriber();
    let timer = Timer::new();

    let args = ComputeRangeGroundTruthArgs::parse();

    tracing::info!("Computing range-search ground truth file");

    let storage_provider = FileStorageProvider;

    let err = match args.data_type {
        DataType::Float => {
            compute_range_ground_truth_from_datafiles::<GraphDataF32Vector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.radius,
                args.filter_bitmap_file.as_deref(),
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
        DataType::Fp16 => {
            compute_range_ground_truth_from_datafiles::<GraphDataHalfVector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.radius,
                args.filter_bitmap_file.as_deref(),
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
        DataType::Uint8 => {
            compute_range_ground_truth_from_datafiles::<GraphDataU8Vector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.radius,
                args.filter_bitmap_file.as_deref(),
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
        DataType::Int8 => {
            compute_range_ground_truth_from_datafiles::<GraphDataInt8Vector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.radius,
                args.filter_bitmap_file.as_deref(),
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
    };

    match err {
        Ok(_) => {
            tracing::info!(
                "Compute range ground-truth completed successfully in {:?}",
                timer.elapsed()
            );
            Ok(())
        }
        Err(err) => {
            tracing::error!("Error: {:?}", err);
            Err(err)
        }
    }
}

#[derive(Debug, Parser)]
struct ComputeRangeGroundTruthArgs {
    /// data type <int8/uint8/float/fp16>
    #[arg(long = "data-type", default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist-fn", default_value = "l2")]
    pub distance_function: Metric,

    /// File containing the base vectors in binary format
    #[arg(long = "base-file", short, required = true)]
    pub base_file: String,

    /// Optional labels file for base vectors
    #[arg(long = "base-file-labels", default_value = None)]
    pub base_file_labels: Option<String>,

    /// File containing the query vectors in binary format
    #[arg(long = "query-file", short, required = true)]
    pub query_file: String,

    /// Optional labels file for query vectors
    #[arg(long = "query-file-labels", default_value = None)]
    pub query_file_labels: Option<String>,

    /// Path of the file to write range ground truth to in binary format
    #[arg(long = "gt-file", short, required = true)]
    pub ground_truth_file: String,

    /// Filter bitmap file in range ground truth format
    #[arg(long = "filter-bitmap-file", short, default_value = None)]
    pub filter_bitmap_file: Option<String>,

    /// Radius threshold used to include neighbors in range-groundtruth
    #[arg(long = "radius", required = true)]
    pub radius: f32,
}
