/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use clap::Parser;
use diskann_providers::{storage::FileStorageProvider, utils::Timer};
use diskann_tools::utils::{
    compute_ground_truth_from_datafiles, init_subscriber, CMDResult, DataType,
};
use diskann_vector::distance::Metric;
use diskann_vector::Half;

fn main() -> CMDResult<()> {
    init_subscriber();
    let timer = Timer::new();

    let args = ComputeGroundTruthArgs::parse();

    tracing::info!("Computing ground truth file");

    let insert_file = None;
    let skip_base = None;

    let storage_provider = FileStorageProvider;

    let err = match args.data_type {
        DataType::Float => compute_ground_truth_from_datafiles::<f32, (), FileStorageProvider>(
            &storage_provider,
            args.distance_function,
            &args.base_file,
            &args.query_file,
            &args.ground_truth_file,
            args.filter_bitmap_file.as_deref(),
            args.recall_at,
            insert_file,
            skip_base,
            args.associated_data_file,
            args.base_file_labels.as_deref(),
            args.query_file_labels.as_deref(),
        ),
        DataType::Fp16 => compute_ground_truth_from_datafiles::<Half, (), FileStorageProvider>(
            &storage_provider,
            args.distance_function,
            &args.base_file,
            &args.query_file,
            &args.ground_truth_file,
            args.filter_bitmap_file.as_deref(),
            args.recall_at,
            insert_file,
            skip_base,
            args.associated_data_file,
            args.base_file_labels.as_deref(),
            args.query_file_labels.as_deref(),
        ),
        DataType::Uint8 => compute_ground_truth_from_datafiles::<u8, (), FileStorageProvider>(
            &storage_provider,
            args.distance_function,
            &args.base_file,
            &args.query_file,
            &args.ground_truth_file,
            args.filter_bitmap_file.as_deref(),
            args.recall_at,
            insert_file,
            skip_base,
            args.associated_data_file,
            args.base_file_labels.as_deref(),
            args.query_file_labels.as_deref(),
        ),
        DataType::Int8 => compute_ground_truth_from_datafiles::<i8, (), FileStorageProvider>(
            &storage_provider,
            args.distance_function,
            &args.base_file,
            &args.query_file,
            &args.ground_truth_file,
            args.filter_bitmap_file.as_deref(),
            args.recall_at,
            insert_file,
            skip_base,
            args.associated_data_file,
            args.base_file_labels.as_deref(),
            args.query_file_labels.as_deref(),
        ),
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
struct ComputeGroundTruthArgs {
    /// data type <int8/uint8/float / fp16> (required)
    #[arg(long = "data-type", default_value = "float")]
    pub data_type: DataType,

    /// Distance function to use.
    #[arg(long = "dist-fn", default_value = "l2")]
    pub distance_function: Metric,

    /// File containing the base vectors in binary format
    #[arg(long = "base-file", short, required = true)]
    pub base_file: String,

    #[arg(long = "base-file-labels", default_value = None)]
    pub base_file_labels: Option<String>,

    /// File containing the query vectors in binary format
    #[arg(long = "query-file", short, required = true)]
    pub query_file: String,

    #[arg(long = "query-file-labels", default_value = None)]
    pub query_file_labels: Option<String>,

    /// Path of the file to write the ground truth to in binary format.  Please don't append .bin at the end if no filter_label or filter_label_file is provided.  It will save the file with '.bin' at the end.  Otherwise it will save the file as filename_label.bin.
    #[arg(long = "gt-file", short, required = true)]
    pub ground_truth_file: String,

    /// Filter bitmap file in the range ground truth format
    #[arg(long = "filter-bitmap-file", short, default_value = None)]
    pub filter_bitmap_file: Option<String>,

    /// Number of ground truth nearest neigbhors to compute
    #[arg(long = "recall-at", short = 'K', default_value = "10")]
    pub recall_at: u32,

    /// File containing the associated data in binary format
    #[arg(long = "associated-data-file")]
    pub associated_data_file: Option<String>,
}
