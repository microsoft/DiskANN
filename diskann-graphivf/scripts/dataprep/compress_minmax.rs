/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compress a full-precision (`fp16` or `f32`) `.bin` corpus / query file into
//! 8-bit MinMax-quantized canonical rows and write them to a new `.bin` file.
//!
//! Each output row is one canonical [`MinMaxElement<8>`] vector (the quantized
//! codes followed by the embedded min/max compensation meta), so the file can
//! be loaded directly as a `Matrix<MinMaxElement<8>>` and used as the stored
//! element type `T` in graph-IVF (build via
//! [`GraphIvfIndex::build_from_compressed_seeded_profiled`], search by loading
//! the compressed queries as the same `T`).
//!
//! Quantization uses a `Null` transform (no rotation) and `grid_scale = 1.0`.
//!
//! Run (compress both the corpus and the queries):
//! ```text
//! cargo run --release --example compress_minmax -- <input.bin> <output.bin> [fp16|f32]
//! ```

use std::{
    fs::File,
    io::{BufWriter, Write},
    num::NonZeroUsize,
};

use diskann_graphivf::Half;
use diskann_providers::common::MinMaxElement;
use diskann_quantization::{
    algorithms::{transforms::NullTransform, Transform},
    minmax::{DataMutRef, DataRef, MinMaxQuantizer},
    num::Positive,
    CompressInto,
};
use diskann_utils::{
    io::{read_bin, Metadata},
    views::Matrix,
    ReborrowMut,
};

/// Target bitrate for the MinMax quantizer.
const NBITS: usize = 8;

const USAGE: &str = "usage: compress_minmax <input.bin> <output.bin> [fp16|f32]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or(USAGE)?;
    let output = args.next().ok_or(USAGE)?;
    let dtype = args.next().unwrap_or_else(|| "fp16".to_string());

    // --- Load the raw input; access rows as f32 without materializing a full
    //     f32 copy of the corpus (important for very large inputs) -----------
    let raw_f32: Option<Matrix<f32>>;
    let raw_u16: Option<Matrix<u16>>;
    let (num_points, dim): (usize, usize);
    match dtype.as_str() {
        "fp16" | "f16" => {
            let raw: Matrix<u16> = read_bin(&mut File::open(&input)?)?;
            num_points = raw.nrows();
            dim = raw.ncols();
            raw_u16 = Some(raw);
            raw_f32 = None;
        }
        "f32" => {
            let raw: Matrix<f32> = read_bin(&mut File::open(&input)?)?;
            num_points = raw.nrows();
            dim = raw.ncols();
            raw_f32 = Some(raw);
            raw_u16 = None;
        }
        other => return Err(format!("unsupported dtype {other:?} (expected fp16 or f32)").into()),
    }
    println!("input:  {num_points} x {dim} ({dtype})  {input}");

    // --- Compress each row to 8-bit MinMax canonical form --------------------
    let canonical = DataRef::<NBITS>::canonical_bytes(dim);
    let nz_dim = NonZeroUsize::new(dim).ok_or("dimension must be non-zero")?;
    let transform = Transform::Null(NullTransform::new(nz_dim));
    let quantizer =
        MinMaxQuantizer::new(transform, Positive::new(1.0).map_err(|e| format!("{e:?}"))?);

    // Stream the output: write the header once, then one canonical row at a
    // time so we never hold a second full-corpus buffer in memory.
    let mut writer = BufWriter::new(File::create(&output)?);
    Metadata::new(num_points, canonical)?.write(&mut writer)?;

    // Reusable per-row buffers: the source f32 row (only needed for fp16, where
    // we decode into it) and the destination canonical `MinMaxElement<8>` row.
    let mut row_f32 = vec![0.0f32; dim];
    let mut dst_row = vec![MinMaxElement::<NBITS>::default(); canonical];
    for i in 0..num_points {
        let row: &[f32] = match (&raw_f32, &raw_u16) {
            (Some(m), _) => m.row(i),
            (_, Some(m)) => {
                for (dst, &bits) in row_f32.iter_mut().zip(m.row(i)) {
                    *dst = Half::from_bits(bits).to_f32();
                }
                &row_f32
            }
            _ => unreachable!(),
        };
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut dst_row);
        let mut compressed = DataMutRef::<NBITS>::from_canonical_front_mut(bytes, dim)
            .map_err(|e| format!("canonical destination: {e:?}"))?;
        quantizer
            .compress_into(row, compressed.reborrow_mut())
            .map_err(|e| format!("compress: {e:?}"))?;
        writer.write_all(bytemuck::cast_slice(&dst_row))?;
    }
    writer.flush()?;
    println!("output: {num_points} x {canonical} MinMax{NBITS} elements  {output}");
    Ok(())
}
