/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use clap::Parser;
use diskann_disk::utils::aligned_file_reader::AlignedFileReaderFactory;
use diskann_providers::storage::{get_disk_index_file, FileStorageProvider};
use diskann_tools::utils::{
    get_num_threads, init_subscriber, search_disk_index, CMDResult, DataType, GraphDataF32Vector,
    GraphDataHalfVector, GraphDataInt8Vector, GraphDataMinMaxVector, GraphDataU8Vector,
    SearchDiskIndexParameters,
};
use diskann_vector::distance::Metric;

fn main() -> CMDResult<()> {
    init_subscriber();

    let args: SearchDiskIndexArgs = SearchDiskIndexArgs::parse();

    let threads = get_num_threads(args.num_threads);

    let search_disk_index_params = SearchDiskIndexParameters {
        metric: args.dist_fn,
        index_path_prefix: &args.index_path_prefix,
        result_output_prefix: &args.result_path_prefix,
        query_file: &args.query_file,
        vector_filters_file: args.vector_filters_file.as_deref(),
        truthset_file: &args.gt_file,
        num_threads: threads,
        recall_at: args.recall_at,
        beam_width: args.beam_width,
        search_io_limit: args.search_io_limit,
        l_vec: &args.search_list,
        fail_if_recall_below: args.fail_if_recall_below,
        num_nodes_to_cache: args.num_nodes_to_cache,
        is_flat_search: args.flat_search,
    };

    let storage_provider = FileStorageProvider;
    let aligned_file_reader_factory = AlignedFileReaderFactory::new(get_disk_index_file(
        search_disk_index_params.index_path_prefix,
    ));

    let result: CMDResult<i32> = if args.use_minmax {
        search_disk_index::<GraphDataMinMaxVector, FileStorageProvider, _>(
            &storage_provider,
            search_disk_index_params,
            aligned_file_reader_factory,
        )
    } else {
        match args.data_type {
            DataType::Float => search_disk_index::<GraphDataF32Vector, FileStorageProvider, _>(
                &storage_provider,
                search_disk_index_params,
                aligned_file_reader_factory,
            ),
            DataType::Int8 => search_disk_index::<GraphDataInt8Vector, FileStorageProvider, _>(
                &storage_provider,
                search_disk_index_params,
                aligned_file_reader_factory,
            ),
            DataType::Uint8 => search_disk_index::<GraphDataU8Vector, FileStorageProvider, _>(
                &storage_provider,
                search_disk_index_params,
                aligned_file_reader_factory,
            ),
            DataType::Fp16 => search_disk_index::<GraphDataHalfVector, FileStorageProvider, _>(
                &storage_provider,
                search_disk_index_params,
                aligned_file_reader_factory,
            ),
        }
    };

    match result {
        Ok(_) => {
            println!("Index search completed successfully");
            Ok(())
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            Err(err)
        }
    }
}

#[derive(Debug, Parser)]
struct SearchDiskIndexArgs {
    /// data type <int8/uint8/float/fp16> (required)
    #[arg(long = "data_type", required = true)]
    pub data_type: DataType,

    /// Boolean on whether the index and queries use minmax vectors.
    #[arg(long = "use_minmax", default_value = "false")]
    pub use_minmax: bool,

    /// Distance function to use (l2, cosine)
    #[arg(long = "dist_fn", required = true)]
    pub dist_fn: Metric,

    /// Path to the index file
    #[arg(long = "index_path_prefix", required = true)]
    pub index_path_prefix: String,

    /// Path for saving results of the queries
    #[arg(long = "result_path", required = true)]
    pub result_path_prefix: String,

    /// Query file in binary format
    #[arg(long = "query_file", short, required = true)]
    pub query_file: String,

    /// Vector filters file in the range ground truth format
    #[arg(long = "vector_filters_file", short, default_value = None)]
    pub vector_filters_file: Option<String>,

    /// Ground truth file for the queryset
    #[arg(long = "gt_file", default_value = "")]
    pub gt_file: String,

    /// Number of neighbors to be returned
    #[arg(long = "recall_at", short = 'K', default_value = "10")]
    pub recall_at: u32,

    /// List of L values of search
    #[arg(long = "search_list", short = 'L', required = true, num_args=1..)]
    pub search_list: Vec<u32>,

    /// Beam width for beam search
    #[arg(long = "beam_width", default_value = "2")]
    pub beam_width: u32,

    /// IO limit for each beam search, the default value is u32::MAX
    #[arg(long = "search_io_limit", default_value = "4294967295")]
    pub search_io_limit: u32,

    /// Number of threads used for querying the index
    #[arg(long = "num_threads", short = 'T')]
    pub num_threads: Option<usize>,

    /// Print overall QPS divided by the number of threads in the output table
    #[arg(long = "fail_if_recall_below", default_value = "0.0")]
    pub fail_if_recall_below: f32,

    /// Number of BFS nodes around medoid(s) to cache during query warm up
    #[arg(long = "num_nodes_to_cache", default_value = "0")]
    pub num_nodes_to_cache: usize,

    /// Flat search enabled
    #[arg(long = "flat_search", default_value = "false")]
    pub flat_search: bool,
}
