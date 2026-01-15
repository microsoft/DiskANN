/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use clap::Parser;
use diskann_providers::{storage::FileStorageProvider, utils::Timer};
use diskann_tools::utils::{
    compute_multivec_ground_truth_from_datafiles, init_subscriber, CMDResult, DataType,
    GraphDataF32Vector, GraphDataHalfVector, GraphDataInt8Vector, GraphDataU8Vector,
    MultivecAggregationMethod,
};
use diskann_vector::distance::Metric;

fn main() -> CMDResult<()> {
    init_subscriber();
    let timer = Timer::new();

    let args = ComputeMultivecGroundTruthArgs::parse();

    tracing::info!("Computing ground truth file");

    let storage_provider = FileStorageProvider;

    let err = match args.data_type {
        DataType::Float => {
            compute_multivec_ground_truth_from_datafiles::<GraphDataF32Vector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                args.aggregation,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.recall_at,
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
        DataType::Fp16 => {
            compute_multivec_ground_truth_from_datafiles::<GraphDataHalfVector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                args.aggregation,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.recall_at,
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
        DataType::Uint8 => {
            compute_multivec_ground_truth_from_datafiles::<GraphDataU8Vector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                args.aggregation,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.recall_at,
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
        DataType::Int8 => {
            compute_multivec_ground_truth_from_datafiles::<GraphDataInt8Vector, FileStorageProvider>(
                &storage_provider,
                args.distance_function,
                args.aggregation,
                &args.base_file,
                &args.query_file,
                &args.ground_truth_file,
                args.recall_at,
                args.base_file_labels.as_deref(),
                args.query_file_labels.as_deref(),
            )
        }
    };

    match err {
        Ok(_) => {
            tracing::info!(
                "Compute ground-truth completed successfully in {:?}",
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
struct ComputeMultivecGroundTruthArgs {
    /// data type <int8/uint8/float / fp16> (required)
    #[arg(long = "data_type", default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist_fn", default_value = "l2")]
    pub distance_function: Metric,

    /// Whether to use average or min aggregation
    #[arg(long = "aggregation", default_value = "average")]
    pub aggregation: MultivecAggregationMethod,

    /// File containing the base vectors in binary format
    #[arg(long = "base_file", short, required = true)]
    pub base_file: String,

    #[arg(long = "base_file_labels", default_value = None)]
    pub base_file_labels: Option<String>,

    /// File containing the query vectors in binary format
    #[arg(long = "query_file", short, required = true)]
    pub query_file: String,

    #[arg(long = "query_file_labels", default_value = None)]
    pub query_file_labels: Option<String>,

    /// Path of the file to write the ground truth to in binary format.  Please don't append .bin at the end if no filter_label or filter_label_file is provided.  It will save the file with '.bin' at the end.  Otherwise it will save the file as filename_label.bin.
    #[arg(long = "gt_file", short, required = true)]
    pub ground_truth_file: String,

    /// Number of ground truth nearest neighbors to compute
    #[arg(long = "recall_at", short = 'K', default_value = "10")]
    pub recall_at: u32,
}
