/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann::{graph::config::defaults::ALPHA, ANNError};
use diskann_disk::QuantizationType;
use diskann_providers::storage::FileStorageProvider;
use diskann_tools::utils::{
    build_disk_index, get_num_threads, init_subscriber, BuildDiskIndexParameters, CMDToolError,
    DataType, DimensionValues, GraphDataF32Vector, GraphDataHalfVector, GraphDataInt8Vector,
    GraphDataMinMaxVector, GraphDataU8Vector,
};
use diskann_vector::distance::Metric;

fn main() -> Result<(), CMDToolError> {
    init_subscriber();

    let args: BuildDiskIndexArgs = BuildDiskIndexArgs::parse();

    let threads = get_num_threads(args.num_threads);

    println!(
        "Starting index build with R: {}  Lbuild: {}  alpha: {}  #threads: {} num_of_pq_chunks: {} build_DRAM_budget: {}",
        args.max_degree, args.l_build, ALPHA, threads, args.num_of_pq_chunks, args.build_dram_budget
    );

    let parameters = BuildDiskIndexParameters {
        metric: args.dist_fn,
        data_path: &args.data_path,
        r: args.max_degree,
        l: args.l_build,
        index_path_prefix: &args.index_path_prefix,
        num_threads: threads,
        index_build_ram_limit_gb: args.build_dram_budget,
        build_quantization_type: args.build_quantization_type,
        num_of_pq_chunks: args.num_of_pq_chunks,
        chunking_parameters: None,
        dim_values: DimensionValues::new(args.dim, args.dim),
    };

    let storage_provider = FileStorageProvider;

    let err = if args.use_minmax {
        if args.full_dim.is_none() {
            return Err(ANNError::log_index_config_error(
                format!("Full dim : {:?}", args.full_dim),
                "full_dim cannot be None for minmax based build".to_string(),
            )
            .into());
        }
        let parameters = BuildDiskIndexParameters {
            dim_values: DimensionValues::new(args.dim, args.full_dim.unwrap()),
            ..parameters
        };
        build_disk_index::<GraphDataMinMaxVector, FileStorageProvider>(
            &storage_provider,
            parameters,
        )
    } else {
        match args.data_type {
            DataType::Int8 => build_disk_index::<GraphDataInt8Vector, FileStorageProvider>(
                &storage_provider,
                parameters,
            ),
            DataType::Uint8 => build_disk_index::<GraphDataU8Vector, FileStorageProvider>(
                &storage_provider,
                parameters,
            ),
            DataType::Float => build_disk_index::<GraphDataF32Vector, FileStorageProvider>(
                &storage_provider,
                parameters,
            ),
            DataType::Fp16 => build_disk_index::<GraphDataHalfVector, FileStorageProvider>(
                &storage_provider,
                parameters,
            ),
        }
    };

    match err {
        Ok(_) => {
            println!("Index build completed successfully");
            Ok(())
        }
        Err(err) => {
            tracing::error!("Index build failed - see diagnostic");
            Err(err.into())
        }
    }
}

#[derive(Debug, Parser)]
struct BuildDiskIndexArgs {
    /// data type <int8/uint8/float / fp16> (required)
    #[arg(long = "data_type", default_value = "float")]
    pub data_type: DataType,

    /// Should minmax vectors be used. Hack to reuse the code without implementing a new DataType.
    #[arg(long = "use_minmax", default_value = "false")]
    pub use_minmax: bool,

    /// The dimension of the input dataset
    #[arg(long = "dim", required = true)]
    pub dim: usize,

    /// Optional value for the dimension of the dataset vectors when converted to full-precision vectors.
    /// This is only required when `use_minmax` = true.
    #[arg(long = "full_dim")]
    pub full_dim: Option<usize>,

    /// Distance function to use.
    #[arg(long = "dist_fn", default_value = "l2")]
    pub dist_fn: Metric,

    /// Path to the data file. The file should be in the format specified by the `data_type` argument.
    #[arg(long = "data_path", short, required = true)]
    pub data_path: String,

    /// Path to the index file. The index will be saved to this prefixed name.
    #[arg(long = "index_path_prefix", short, required = true)]
    pub index_path_prefix: String,

    /// Number of max out degree from a vertex.
    #[arg(long = "max_degree", short = 'R', default_value = "64")]
    pub max_degree: u32,

    /// Number of candidates to consider when building out edges
    #[arg(long = "l_build", short = 'L', default_value = "100")]
    pub l_build: u32,

    /// DRAM budget in GB for building the index
    #[arg(long = "build_DRAM_budget", short = 'M', default_value = "0")]
    pub build_dram_budget: f64,

    /// Quantization type to quantize the vector data.
    /// Supported values: 'FP', 'SQ_NBITS'(STDDEV=2.0 by default), 'SQ_NBITS_STDDEV'
    #[arg(long = "build_quantization_type", short, default_value = "FP")]
    pub build_quantization_type: QuantizationType,

    /// Number of PQ chunks to use.
    #[arg(long = "num_of_pq_chunks", required = true)]
    pub num_of_pq_chunks: usize,

    /// Number of threads to use.
    #[arg(long = "num_threads", short = 'T')]
    pub num_threads: Option<usize>,

    /// Set true for OPQ compression while using PQ distance comparisons for building the index, and false for PQ compression
    #[arg(long = "use_opq", short, default_value = "false")]
    pub use_opq: bool,
}
