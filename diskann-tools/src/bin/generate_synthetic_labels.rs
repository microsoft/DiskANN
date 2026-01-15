/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use clap::Parser;
use diskann_providers::utils::generate_synthetic_labels_utils::generate_labels;
use diskann_tools::utils::{CMDResult, CMDToolError};
use tracing::{error, info};

#[derive(Debug, Parser)]
struct GenerateSyntheticLabelsArgs {
    /// Filename for saving the label file
    #[arg(long = "output_file", required = true)]
    pub output_file: String,

    /// Number of vectors
    #[arg(long = "npts", short = 'N', required = true)]
    pub number_of_vectors: u64,

    /// Number of labels
    #[arg(long = "nlbls", short = 'L', required = true)]
    pub number_of_labels: u32,

    /// Distribution function for labels random/zipf/one_per_point, defaults to random
    #[arg(long = "ndt", short = 'D', required = true, default_value = "random")]
    pub distribution_type: String,
}

fn main() -> CMDResult<()> {
    let args: GenerateSyntheticLabelsArgs = GenerateSyntheticLabelsArgs::parse();

    if args.number_of_labels > 5000 {
        return Err(CMDToolError {
            details: "Error: num_labels must be 5000 or less".to_string(),
        });
    }

    if args.number_of_vectors == 0 {
        return Err(CMDToolError {
            details: "Error: num_points must be greater than 0".to_string(),
        });
    }

    info!(
        "Generating synthetic labels for {} points with {} unique labels.",
        args.number_of_vectors, args.number_of_labels
    );

    match generate_labels(
        &args.output_file,
        &args.distribution_type,
        args.number_of_vectors as usize,
        args.number_of_labels,
        &mut diskann_providers::utils::create_rnd_from_seed(42),
    ) {
        Ok(_) => {
            info!("Successfully generated labels");
            Ok(())
        }
        Err(err) => {
            error!("Label generation failed: {:?}", err);
            Err(err.into())
        }
    }
}
