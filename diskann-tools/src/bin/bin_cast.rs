/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Convert a DiskANN `.bin` vector file from one POD element type to another.
//!
//! Only widening casts (`u8 → f32`, `i8 → f32`, `f16 → f32`) and the identity
//! cast are supported. Use this when an existing dataset is in `u8` but the
//! benchmark you want to run only accepts `f32`.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use diskann_utils::io::{read_bin, Metadata};
use half::f16;

#[derive(Copy, Clone, Debug, ValueEnum)]
#[value(rename_all = "lower")]
enum InType {
    U8,
    I8,
    F16,
    F32,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
#[value(rename_all = "lower")]
enum OutType {
    F32,
}

#[derive(Parser, Debug)]
#[command(
    name = "bin_cast",
    about = "Cast a DiskANN .bin file to a different element type"
)]
struct Args {
    /// Input .bin file.
    #[arg(long)]
    input: PathBuf,

    /// Element type stored in the input file.
    #[arg(long, value_enum)]
    in_type: InType,

    /// Output .bin file.
    #[arg(long)]
    output: PathBuf,

    /// Element type to emit in the output file. Currently only `f32` is supported.
    #[arg(long, value_enum, default_value_t = OutType::F32)]
    out_type: OutType,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut reader = BufReader::new(
        File::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?,
    );
    let mut writer = BufWriter::new(
        File::create(&args.output)
            .with_context(|| format!("creating {}", args.output.display()))?,
    );

    // Read the entire input file into a typed Matrix using the existing reader.
    match (args.in_type, args.out_type) {
        (InType::U8, OutType::F32) => cast::<u8, f32>(&mut reader, &mut writer, |x| x as f32)?,
        (InType::I8, OutType::F32) => cast::<i8, f32>(&mut reader, &mut writer, |x| x as f32)?,
        (InType::F16, OutType::F32) => cast::<f16, f32>(&mut reader, &mut writer, |x| x.to_f32())?,
        (InType::F32, OutType::F32) => cast::<f32, f32>(&mut reader, &mut writer, |x| x)?,
    }

    writer.flush()?;
    Ok(())
}

fn cast<I, O>(
    reader: &mut BufReader<File>,
    writer: &mut BufWriter<File>,
    mut f: impl FnMut(I) -> O,
) -> Result<()>
where
    I: bytemuck::Pod,
    O: bytemuck::Pod,
{
    let m = read_bin::<I>(reader)?;
    let (nrows, ncols) = (m.nrows(), m.ncols());
    if nrows > u32::MAX as usize || ncols > u32::MAX as usize {
        bail!(
            "dimensions too large for the bin format (nrows={}, ncols={})",
            nrows,
            ncols
        );
    }
    Metadata::new(nrows as u32, ncols as u32)
        .map_err(|_| anyhow!("dimension overflow building bin metadata"))?
        .write(writer)?;

    // Stream the conversion row by row to avoid allocating a second full
    // matrix in memory.
    let mut out_row: Vec<O> = vec![bytemuck::Zeroable::zeroed(); ncols];
    for r in 0..nrows {
        let in_row = m.row(r);
        for c in 0..ncols {
            out_row[c] = f(in_row[c]);
        }
        writer.write_all(bytemuck::cast_slice::<O, u8>(&out_row))?;
    }

    println!("converted {} rows × {} cols", nrows, ncols);
    Ok(())
}
