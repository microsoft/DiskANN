/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! 1-bit scalar quantization for PiPNN.
//!
//! Reuses diskann-quantization's ScalarQuantizer for training (shift/scale),
//! then packs vectors into compact bit arrays for fast Hamming distance.

use rayon::prelude::*;

/// Result of 1-bit quantization.
pub struct QuantizedData {
    /// Packed bit vectors: each vector is `bytes_per_vec` bytes.
    /// Layout: npoints * bytes_per_vec, row-major.
    pub bits: Vec<u8>,
    /// Number of bytes per vector (ceil(ndims / 8)).
    pub bytes_per_vec: usize,
    /// Original dimensionality.
    pub ndims: usize,
    /// Number of points.
    pub npoints: usize,
}

/// Train a 1-bit scalar quantizer and compress all data.
///
/// Uses the existing diskann-quantization ScalarQuantizer to compute
/// per-dimension shift and scale, then packs each dimension to 1 bit:
///   bit = 1 if (value - shift[d]) * inverse_scale >= 0.5, else 0
pub fn quantize_1bit(data: &[f32], npoints: usize, ndims: usize) -> QuantizedData {
    // Train: compute per-dimension mean and scale.
    let (shift, inverse_scale) = train_1bit(data, npoints, ndims);

    let bytes_per_vec = (ndims + 7) / 8;
    let mut bits = vec![0u8; npoints * bytes_per_vec];

    // Parallel quantization.
    bits.par_chunks_mut(bytes_per_vec)
        .enumerate()
        .for_each(|(i, out)| {
            let vec = &data[i * ndims..(i + 1) * ndims];
            for d in 0..ndims {
                let code = ((vec[d] - shift[d]) * inverse_scale).clamp(0.0, 1.0).round() as u8;
                if code > 0 {
                    out[d / 8] |= 1 << (d % 8);
                }
            }
        });

    QuantizedData {
        bits,
        bytes_per_vec,
        ndims,
        npoints,
    }
}

/// Train 1-bit quantizer: compute per-dimension shift and inverse_scale.
fn train_1bit(data: &[f32], npoints: usize, ndims: usize) -> (Vec<f32>, f32) {
    let standard_deviations = 2.0f64;

    // Compute per-dimension mean.
    let mut mean = vec![0.0f64; ndims];
    for i in 0..npoints {
        let vec = &data[i * ndims..(i + 1) * ndims];
        for d in 0..ndims {
            mean[d] += vec[d] as f64;
        }
    }
    let inv_n = 1.0 / npoints as f64;
    for d in 0..ndims {
        mean[d] *= inv_n;
    }

    // Compute per-dimension standard deviation.
    let mut var = vec![0.0f64; ndims];
    for i in 0..npoints {
        let vec = &data[i * ndims..(i + 1) * ndims];
        for d in 0..ndims {
            let diff = vec[d] as f64 - mean[d];
            var[d] += diff * diff;
        }
    }
    for d in 0..ndims {
        var[d] = (var[d] * inv_n).sqrt(); // stddev
    }

    // Scale = 2 * stdev * max_stddev (same as diskann-quantization)
    let max_stddev = var.iter().cloned().fold(0.0f64, f64::max);
    let scale = 2.0 * standard_deviations * max_stddev;
    let inverse_scale = 1.0 / scale as f32; // For 1-bit: bit_scale(1) = 1

    // Shift = mean - stdev * max_stddev
    let shift: Vec<f32> = mean
        .iter()
        .map(|&m| (m - standard_deviations * max_stddev) as f32)
        .collect();

    (shift, inverse_scale)
}

impl QuantizedData {
    /// Get the packed bit vector for point i.
    #[inline]
    pub fn get(&self, i: usize) -> &[u8] {
        let start = i * self.bytes_per_vec;
        &self.bits[start..start + self.bytes_per_vec]
    }

    /// Compute Hamming distance between two quantized vectors.
    #[inline]
    pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
        let mut dist = 0u32;
        // Process 8 bytes at a time for efficiency.
        let chunks = a.len() / 8;
        let a64 = a.as_ptr() as *const u64;
        let b64 = b.as_ptr() as *const u64;
        for i in 0..chunks {
            unsafe {
                let xor = *a64.add(i) ^ *b64.add(i);
                dist += xor.count_ones();
            }
        }
        // Handle remaining bytes.
        for i in (chunks * 8)..a.len() {
            dist += (a[i] ^ b[i]).count_ones();
        }
        dist
    }
}
