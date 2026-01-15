/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{BufWriter, Write},
    num::NonZero,
};

use anyhow::{Context, Result};
use clap::Parser;
use diskann_providers::utils::write_metadata;
use diskann_quantization::{
    algorithms::transforms::{DoubleHadamard, TargetDim},
    alloc::GlobalAllocator,
    minmax::{DataMutRef, MinMaxQuantizer},
    num::Positive,
    CompressInto,
};
use half::f16;
use rand::{rngs::StdRng, SeedableRng};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input binary file path containing vector data
    #[arg(short, long)]
    input: String,

    /// Output binary file path for quantized vectors
    #[arg(short, long)]
    output: String,

    /// Number of bits for quantization (1, 2, 4, or 8)
    #[arg(short, long, default_value = "4")]
    bits: u8,

    #[arg(short, long, default_value = "f32")]
    precision: String,

    #[arg(short, long, default_value = "2282129662191")]
    seed: u64,

    #[arg(short, long, default_value = "1.0")]
    grid_scale: f32,
}

fn dispatch_process_file<T: Copy + Into<f32> + bytemuck::Pod>(
    bits: u8,
    input: &str,
    output: &str,
    seed: u64,
    scale: f32,
) -> Result<()> {
    match bits {
        1 => process_file::<1, T>(input, output, seed, scale),
        2 => process_file::<2, T>(input, output, seed, scale),
        4 => process_file::<4, T>(input, output, seed, scale),
        8 => process_file::<8, T>(input, output, seed, scale),
        _ => anyhow::bail!(
            "Unsupported bit width: {}. Supported values are 1, 2, 4, 8",
            bits
        ),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.precision.as_str() {
        "f32" | "float32" => dispatch_process_file::<f32>(
            args.bits,
            &args.input,
            &args.output,
            args.seed,
            args.grid_scale,
        ),
        "fp16" | "f16" => dispatch_process_file::<f16>(
            args.bits,
            &args.input,
            &args.output,
            args.seed,
            args.grid_scale,
        ),
        _ => anyhow::bail!(
            "Unsupported precision: {}. Supported values are f32, fp16, f16, float",
            args.precision
        ),
    }
}

fn process_file<const NBITS: usize, T: Copy + Into<f32> + bytemuck::Pod>(
    input_path: &str,
    output_path: &str,
    seed: u64,
    scale: f32,
) -> Result<()>
where
    diskann_quantization::bits::Unsigned: diskann_quantization::bits::Representation<NBITS>,
{
    // Load input data
    let (input_data, num_points, dim) = diskann_providers::utils::file_util::load_bin::<T, _>(
        &diskann_providers::storage::FileStorageProvider,
        input_path,
        0,
    )
    .with_context(|| format!("Failed to load data from {}", input_path))?;

    if input_data.len() != num_points * dim {
        anyhow::bail!(
            "Data size mismatch: expected {} elements, got {}",
            num_points * dim,
            input_data.len()
        );
    }

    println!("Input file: {} points, {} dimensions", num_points, dim);

    let mut rng = StdRng::seed_from_u64(seed);
    // Create MinMax quantizer
    let double_hadamard = DoubleHadamard::new(
        NonZero::new(dim).unwrap(),
        TargetDim::Same,
        &mut rng,
        GlobalAllocator,
    )
    .unwrap();
    let transform = diskann_quantization::algorithms::Transform::DoubleHadamard(double_hadamard);
    let quantizer = MinMaxQuantizer::new(transform, Positive::new(scale)?);

    let output_dim = quantizer.output_dim();

    // Calculate bytes per quantized vector
    let bytes_per_vector = diskann_quantization::minmax::Data::<NBITS>::canonical_bytes(output_dim);
    println!("Bytes per quantized vector: {}", bytes_per_vector);

    // Create output file
    let output_file = File::create(output_path)
        .with_context(|| format!("Failed to create output file {}", output_path))?;
    let mut writer = BufWriter::new(output_file);

    // Write output header: num_points (u32) and bytes_per_vector (u32)
    write_metadata(&mut writer, num_points, bytes_per_vector)
        .context("Failed to write metadata header")?;

    println!("Processing {} vectors...", num_points);

    let mut loss = 0.0;

    // Process vectors one by one
    for i in 0..num_points {
        // Get input vector
        let start_idx = i * dim;
        let end_idx = start_idx + dim;
        let input_vector: Vec<f32> = input_data[start_idx..end_idx]
            .iter()
            .map(|x| (*x).into())
            .collect();

        // Create buffer for quantized data with proper alignment
        let mut quantized_buffer = vec![0u8; bytes_per_vector];

        // Create mutable reference to quantized data
        let quantized_data =
            DataMutRef::<NBITS>::from_canonical_front_mut(&mut quantized_buffer, output_dim)
                .with_context(|| format!("Failed to create quantized data ref for vector {}", i))?;

        // Compress the vector
        let loss_x = quantizer
            .compress_into(input_vector.as_slice(), quantized_data)
            .with_context(|| format!("Failed to compress vector {}", i))?;

        loss += loss_x.as_f32();

        // Write the quantized data (only the actual bytes, not the aligned padding)
        writer
            .write_all(&quantized_buffer)
            .with_context(|| format!("Failed to write quantized vector {}", i))?;
    }

    writer.flush().context("Failed to flush output file")?;

    println!(
        "Successfully quantized {} vectors to {}",
        num_points, output_path
    );
    println!("Average l2 loss : {}", loss / (num_points as f32));
    println!("Output file format:");
    println!(
        "  Header: {} bytes (num_points: u32, bytes_per_vector: u32)",
        8
    );
    println!(
        "  Data: {} bytes ({} vectors × {} bytes each)",
        num_points * bytes_per_vector,
        num_points,
        bytes_per_vector
    );

    Ok(())
}
