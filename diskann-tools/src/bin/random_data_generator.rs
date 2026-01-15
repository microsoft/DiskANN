/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_providers::storage::FileStorageProvider;
use diskann_tools::utils::{write_random_data, CMDResult, CMDToolError, DataType};

#[derive(Debug, Parser)]
struct RandomDataGeneratorArgs {
    /// data type <int8/uint8/float/fp16> (required)
    #[arg(long = "data_type", required = true)]
    pub data_type: DataType,

    /// File name for saving the random vectors
    #[arg(long = "output_file", required = true)]
    pub output_file: String,

    /// Dimensionality of the vector
    #[arg(long = "ndims", short = 'D', required = true)]
    pub number_of_dimensions: usize,

    /// Number of vectors
    #[arg(long = "npts", short = 'N', required = true)]
    pub number_of_vectors: u64,

    /// Norm of the vectors.  Vectors are random points around a sphere.  'norm' is the radius of
    /// the sphere.  This is usually between 100 and 200.  If 'norm' is too small it can cause the
    /// random data to be all zeros.  When data_type is 'uint8' this needs to be < 128.
    #[arg(long = "norm", required = true, default_value = "150")]
    pub norm: f32,
}

fn main() -> CMDResult<()> {
    let args: RandomDataGeneratorArgs = RandomDataGeneratorArgs::parse();

    if args.norm <= 0.0 {
        return Err(CMDToolError {
            details: "Error: Norm must be a positive number".to_string(),
        });
    }

    if (args.data_type == DataType::Int8 || args.data_type == DataType::Uint8) && args.norm > 127.0
    {
        return Err(CMDToolError {
            details: "Error: for int8/uint8 datatypes, L2 norm cannot be greater than 127."
                .to_string(),
        });
    }

    let storage_provider = FileStorageProvider;
    match write_random_data(
        &storage_provider,
        &args.output_file,
        args.data_type,
        args.number_of_dimensions,
        args.number_of_vectors,
        args.norm,
    ) {
        Ok(_) => {
            println!("Successfully generated random data");
            Ok(())
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            // Return the error
            Err(err)
        }
    }
}
