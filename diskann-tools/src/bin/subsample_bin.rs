/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::Parser;
use half::f16;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardUniform};

use diskann::utils::VectorRepr;
use diskann_providers::storage::FileStorageProvider;
use diskann_providers::storage::StorageWriteProvider;
use diskann_providers::utils::{random, SampleVectorReader, SamplingDensity};
use diskann_tools::utils::DataType;
use diskann_utils::io::Metadata;

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
        Some(seed) => random::create_rnd_from_seed(seed),
        None => random::create_rnd(),
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
    Metadata::new(npts, dims)?.write(&mut writer)?;

    let mut sampled_count: u32 = 0;
    reader.read_vectors(sampled_indices, |vec_t| {
        sampled_count += 1;
        writer.write_all(bytemuck::cast_slice(vec_t))?;
        Ok(())
    })?;

    // Rewrite metadata at the start of the file with the actual sampled count.
    writer.seek(SeekFrom::Start(0))?;
    Metadata::new(sampled_count, dims)?.write(&mut writer)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_utils::io::Metadata;
    use tempfile::TempDir;

    /// Write a standard `.bin` file (8-byte header + row-major f32 data).
    fn write_f32_bin(path: &std::path::Path, npts: usize, dim: usize) {
        let mut f = std::fs::File::create(path).unwrap();
        Metadata::new(npts, dim).unwrap().write(&mut f).unwrap();
        let data: Vec<f32> = (0..npts * dim).map(|i| i as f32).collect();
        f.write_all(bytemuck::cast_slice(&data)).unwrap();
        f.flush().unwrap();
    }

    fn args_for(input: PathBuf, output: PathBuf, probability: f64) -> Args {
        Args {
            data_type: DataType::Float,
            base_bin_file: input,
            sampled_output_file: output,
            sampling_probability: probability,
            random_seed: Some(7),
        }
    }

    #[test]
    fn create_rng_is_deterministic_with_seed() {
        let mut a = create_rng(Some(123));
        let mut b = create_rng(Some(123));
        let dist = StandardUniform;
        let x: u64 = dist.sample(&mut a);
        let y: u64 = dist.sample(&mut b);
        assert_eq!(x, y);
    }

    #[test]
    fn create_rng_without_seed_produces_values() {
        // Smoke test the seedless branch (non-deterministic, just exercise it).
        let mut rng = create_rng(None);
        let _: u64 = StandardUniform.sample(&mut rng);
    }

    #[test]
    fn run_for_type_samples_all_with_probability_one() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in.bin");
        let output = dir.path().join("out.bin");
        write_f32_bin(&input, 12, 4);

        run_for_type::<f32>(&args_for(input, output.clone(), 1.0)).unwrap();

        let mut r = std::fs::File::open(&output).unwrap();
        let (npts, dim) = Metadata::read(&mut r).unwrap().into_dims();
        assert_eq!(npts, 12);
        assert_eq!(dim, 4);
    }

    #[test]
    fn run_for_type_samples_none_with_probability_zero() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in.bin");
        let output = dir.path().join("out.bin");
        write_f32_bin(&input, 12, 4);

        run_for_type::<f32>(&args_for(input, output.clone(), 0.0)).unwrap();

        let mut r = std::fs::File::open(&output).unwrap();
        let (npts, dim) = Metadata::read(&mut r).unwrap().into_dims();
        assert_eq!(npts, 0);
        assert_eq!(dim, 4);
    }

    #[test]
    fn run_for_type_rejects_out_of_range_probability() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("in.bin");
        let output = dir.path().join("out.bin");
        write_f32_bin(&input, 4, 2);

        let err = run_for_type::<f32>(&args_for(input, output, 1.5)).unwrap_err();
        assert!(err.to_string().contains("sampling_probability"));
    }
}
