/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::Parser;
use half::f16;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardUniform};

use diskann::utils::VectorRepr;
use diskann_providers::storage::FileStorageProvider;
use diskann_providers::storage::StorageWriteProvider;
use diskann_providers::utils::{write_metadata, SampleVectorReader, SamplingDensity};
use diskann_tools::utils::DataType;

/// Subsamples vectors from a DiskANN style binary file.
#[derive(Parser, Debug)]
#[command(name = "subsample_bin", about = "Subsample vectors from a binary file")]
struct Args {
    /// Data type of the vectors, one of: float, int8, uint8, fp16
    #[arg(value_enum)]
    data_type: DataType,

    /// Input base binary file
    base_bin_file: PathBuf,

    /// Output file for sampled vectors
    sampled_output_file: PathBuf,

    /// Sampling probability between 0 and 1, for example 0.1
    sampling_probability: f64,

    /// Optional random seed for reproducible sampling
    random_seed: Option<u64>,
}

fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut system_rng = rand::rng();
            StdRng::from_rng(&mut system_rng)
        }
    }
}

/// Runs sampling for a specific stored vector representation.
fn run_for_type<T>(args: &Args) -> Result<()>
where
    T: VectorRepr,
{
    ensure!(
        (0.0..=1.0).contains(&args.sampling_probability),
        "sampling_probability must be in the range 0 to 1"
    );

    let mut rng = create_rng(args.random_seed);
    let storage_provider = FileStorageProvider;

    let data_file = args.base_bin_file.to_string_lossy().to_string();
    let mut reader: SampleVectorReader<T, _> = SampleVectorReader::new(
        &data_file,
        SamplingDensity::from_sample_rate(args.sampling_probability),
        &storage_provider,
    )?;

    let (npts, dims) = reader.get_dataset_headers();
    println!(
        "Found base file {} with {} points of dimension {}",
        data_file, npts, dims
    );

    // Decide which indices to sample using a simple Bernoulli test.
    let distribution = StandardUniform;
    let sampled_indices = (0..npts).filter(|_| {
        let p: f64 = distribution.sample(&mut rng);
        p < args.sampling_probability
    });

    let output_file = args.sampled_output_file.to_string_lossy().to_string();
    let mut writer = storage_provider.create_for_write(&output_file)?;

    // Write metadata with a temporary count, then fix it after sampling.
    write_metadata(&mut writer, npts, dims)?;

    let mut sampled_count: u32 = 0;
    reader.read_vectors(sampled_indices, |vec_t| {
        sampled_count += 1;
        writer.write_all(bytemuck::cast_slice(vec_t))?;
        Ok(())
    })?;

    // Rewrite metadata at the start of the file with the actual sampled count.
    writer.seek(SeekFrom::Start(0))?;
    write_metadata(&mut writer, sampled_count, dims)?;

    println!(
        "Wrote {} points to sample file {}",
        sampled_count, output_file
    );

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.data_type {
        DataType::Float => run_for_type::<f32>(&args),
        DataType::Int8 => run_for_type::<i8>(&args),
        DataType::Uint8 => run_for_type::<u8>(&args),
        DataType::Fp16 => run_for_type::<f16>(&args),
    }
}
