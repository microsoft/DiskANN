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
    #[inline(always)]
    pub fn get(&self, i: usize) -> &[u8] {
        let start = i * self.bytes_per_vec;
        unsafe { self.bits.get_unchecked(start..start + self.bytes_per_vec) }
    }

    /// Get the packed bit vector as u64 slice for point i (fast path).
    #[inline(always)]
    pub fn get_u64(&self, i: usize) -> &[u64] {
        let start = i * self.bytes_per_vec;
        let u64s = self.bytes_per_vec / 8;
        unsafe {
            let ptr = self.bits.as_ptr().add(start) as *const u64;
            std::slice::from_raw_parts(ptr, u64s)
        }
    }

    /// Number of u64s per vector.
    #[inline]
    pub fn u64s_per_vec(&self) -> usize {
        self.bytes_per_vec / 8
    }

    /// Compute Hamming distance between two quantized vectors (u64 fast path).
    #[inline(always)]
    pub fn hamming_u64(a: &[u64], b: &[u64]) -> u32 {
        let mut dist = 0u32;
        for i in 0..a.len() {
            unsafe {
                dist += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones();
            }
        }
        dist
    }

    /// Compute Hamming distance between two byte slices.
    #[inline]
    pub fn hamming(a: &[u8], b: &[u8]) -> u32 {
        let chunks = a.len() / 8;
        let a64 = a.as_ptr() as *const u64;
        let b64 = b.as_ptr() as *const u64;
        let mut dist = 0u32;
        for i in 0..chunks {
            unsafe {
                dist += (*a64.add(i) ^ *b64.add(i)).count_ones();
            }
        }
        for i in (chunks * 8)..a.len() {
            dist += (a[i] ^ b[i]).count_ones();
        }
        dist
    }

    /// Compute all-pairs Hamming distance matrix for a set of points.
    /// Returns flat n x n matrix (row-major) with f32::MAX on diagonal.
    /// Inlines the Hamming computation and uses unchecked indexing for speed.
    pub fn compute_distance_matrix(&self, indices: &[usize]) -> Vec<f32> {
        let n = indices.len();
        let u64s = self.u64s_per_vec();

        // Extract contiguous u64 data for cache locality.
        let mut local: Vec<u64> = vec![0u64; n * u64s];
        for (i, &idx) in indices.iter().enumerate() {
            let src = self.get_u64(idx);
            local[i * u64s..(i + 1) * u64s].copy_from_slice(src);
        }

        let mut dist = vec![f32::MAX; n * n];
        let local_ptr = local.as_ptr();
        let dist_ptr = dist.as_mut_ptr();

        // Flat loop with inlined Hamming — avoids function call + slice bounds overhead.
        for i in 0..n {
            let a_base = unsafe { local_ptr.add(i * u64s) };
            for j in (i + 1)..n {
                let b_base = unsafe { local_ptr.add(j * u64s) };

                // Inline Hamming: XOR + popcount over u64s.
                let mut h = 0u32;
                for k in 0..u64s {
                    unsafe {
                        h += (*a_base.add(k) ^ *b_base.add(k)).count_ones();
                    }
                }

                let d = h as f32;
                unsafe {
                    *dist_ptr.add(i * n + j) = d;
                    *dist_ptr.add(j * n + i) = d;
                }
            }
        }
        dist
    }

    /// Compute Hamming distances from one point to many leaders.
    /// Returns distances as f32 slice.
    pub fn distances_to_leaders(
        &self,
        point_idx: usize,
        leader_indices: &[usize],
        out: &mut [f32],
    ) {
        let u64s = self.u64s_per_vec();
        let pt = self.get_u64(point_idx);
        for (j, &leader_idx) in leader_indices.iter().enumerate() {
            let ld = self.get_u64(leader_idx);
            out[j] = Self::hamming_u64(pt, ld) as f32;
        }
    }
}
