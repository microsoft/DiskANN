/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! 1-bit scalar quantization for PiPNN.
//!
//! Reuses diskann-quantization's ScalarQuantizer for training (shift/scale),
//! then packs vectors into compact bit arrays for fast Hamming distance.

use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    /// Reusable f32 buffer for T→f32 conversion during parallel quantization.
    static QUANT_F32_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// Result of 1-bit quantization.
pub struct QuantizedData {
    /// Packed bit vectors: each vector is `bytes_per_vec` bytes.
    /// Layout: npoints * bytes_per_vec, row-major.
    /// Packed bit vectors: u64-aligned, each vector is `bytes_per_vec` bytes.
    pub(crate) bits: Vec<u8>,
    /// Number of bytes per vector (ceil(ndims / 8), rounded up to 8-byte alignment).
    pub(crate) bytes_per_vec: usize,
    /// Original dimensionality.
    pub(crate) ndims: usize,
    /// Number of points.
    pub(crate) npoints: usize,
}

/// Quantize data to 1-bit using pre-trained shift and inverse_scale parameters.
///
/// Uses parameters from a `ScalarQuantizer` trained by DiskANN's build pipeline,
/// ensuring identical quantization regardless of build algorithm (Vamana vs PiPNN).
///
/// Each dimension is packed to 1 bit:
///   bit = 1 if (value - shift[d]) * inverse_scale >= 0.5, else 0
///
/// # Arguments
/// * `shift` - Per-dimension shift from ScalarQuantizer (length = ndims)
/// * `inverse_scale` - 1.0 / scale from ScalarQuantizer
pub fn quantize_1bit<T: diskann::utils::VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    shift: &[f32],
    inverse_scale: f32,
) -> QuantizedData {
    // Round up to a multiple of 8 bytes (64 bits) so that get_u64() is always aligned.
    let bytes_per_vec = ((ndims + 63) / 64) * 8;
    let total_bytes = npoints * bytes_per_vec;
    // Allocate as Vec<u64> for guaranteed u64 alignment, then reinterpret as Vec<u8>.
    let u64s_total = total_bytes / 8;
    let mut bits_u64 = vec![0u64; u64s_total];
    // SAFETY: Vec<u64> has stricter alignment than Vec<u8>. We reinterpret the allocation
    // in-place, preserving the original alignment. Length and capacity are scaled by 8.
    let mut bits = unsafe {
        let ptr = bits_u64.as_mut_ptr() as *mut u8;
        let len = bits_u64.len().checked_mul(8).expect("u64→u8 len overflow");
        let cap = bits_u64.capacity().checked_mul(8).expect("u64→u8 cap overflow");
        std::mem::forget(bits_u64);
        Vec::from_raw_parts(ptr, len, cap)
    };

    // Parallel quantization: convert T→f32 per-vector streaming (no full f32 copy).
    bits.par_chunks_mut(bytes_per_vec)
        .enumerate()
        .for_each(|(i, out)| {
            let src = &data[i * ndims..(i + 1) * ndims];
            QUANT_F32_BUF.with(|cell| {
                let mut buf = cell.borrow_mut();
                if buf.len() < ndims { buf.resize(ndims, 0.0); }
                let f32_vec = &mut buf[..ndims];
                T::as_f32_into(src, f32_vec).expect("f32 conversion");
                for d in 0..ndims {
                    let code = ((f32_vec[d] - shift[d]) * inverse_scale).clamp(0.0, 1.0).round() as u8;
                    if code > 0 {
                        out[d / 8] |= 1 << (d % 8);
                    }
                }
            });
        });

    QuantizedData {
        bits,
        bytes_per_vec,
        ndims,
        npoints,
    }
}

impl QuantizedData {
    /// Construct from pre-built raw bits buffer. The buffer must be u64-aligned
    /// and sized exactly npoints × bytes_per_vec.
    pub fn from_raw(bits: Vec<u8>, bytes_per_vec: usize, ndims: usize, npoints: usize) -> Self {
        debug_assert_eq!(bits.len(), npoints * bytes_per_vec);
        debug_assert_eq!(bytes_per_vec % 8, 0, "bytes_per_vec must be 8-byte aligned");
        Self { bits, bytes_per_vec, ndims, npoints }
    }

    /// Number of points.
    pub fn npoints(&self) -> usize { self.npoints }
    /// Number of bytes per quantized vector.
    pub fn bytes_per_vec(&self) -> usize { self.bytes_per_vec }
    /// Original dimensionality.
    pub fn ndims(&self) -> usize { self.ndims }

    /// Get the packed bit vector for point i.
    #[inline(always)]
    pub fn get(&self, i: usize) -> &[u8] {
        debug_assert!(i < self.npoints, "QuantizedData::get index {} out of range (npoints={})", i, self.npoints);
        let start = i * self.bytes_per_vec;
        unsafe { self.bits.get_unchecked(start..start + self.bytes_per_vec) }
    }

    /// Get the packed bit vector as u64 slice for point i (fast path).
    #[inline(always)]
    pub fn get_u64(&self, i: usize) -> &[u64] {
        debug_assert!(i < self.npoints, "QuantizedData::get index {} out of range (npoints={})", i, self.npoints);
        let start = i * self.bytes_per_vec;
        let u64s = self.bytes_per_vec / 8;
        // SAFETY: bits buffer was allocated as Vec<u64>, guaranteeing u64 alignment. bytes_per_vec is always a multiple of 8.
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
                let va = std::ptr::read_unaligned(a64.add(i));
                let vb = std::ptr::read_unaligned(b64.add(i));
                dist += (va ^ vb).count_ones();
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
        let pt = self.get_u64(point_idx);
        for (j, &leader_idx) in leader_indices.iter().enumerate() {
            let ld = self.get_u64(leader_idx);
            out[j] = Self::hamming_u64(pt, ld) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create data with known bit patterns for predictable quantization.
    /// All values are either -1.0 or 1.0 so 1-bit quantization is unambiguous.
    fn make_binary_data(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..npoints * ndims)
            .map(|_| if rng.random_bool(0.5) { 1.0f32 } else { -1.0f32 })
            .collect()
    }

    /// Train SQ parameters and quantize to 1-bit. Test-only convenience wrapper
    /// that uses DiskANN's ScalarQuantizer to compute shift/scale, then calls
    /// `quantize_1bit()`.
    fn train_and_quantize(data: &[f32], npoints: usize, ndims: usize) -> QuantizedData {
        let (shift, inverse_scale) = train_sq_params(data, npoints, ndims);
        quantize_1bit(data, npoints, ndims, &shift, inverse_scale)
    }

    /// Train SQ parameters (shift, inverse_scale) from data. Test-only helper.
    fn train_sq_params(data: &[f32], npoints: usize, ndims: usize) -> (Vec<f32>, f32) {
        use diskann_quantization::scalar::train::ScalarQuantizationParameters;
        use diskann_utils::views::MatrixView;

        let data_matrix = MatrixView::try_from(data, npoints, ndims)
            .expect("data length must equal npoints * ndims");
        let quantizer = ScalarQuantizationParameters::default().train(data_matrix);
        let shift = quantizer.shift().to_vec();
        let scale = quantizer.scale();
        let inverse_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
        (shift, inverse_scale)
    }

    #[test]
    fn test_quantize_1bit_basic() {
        // 4 points, 16 dims -- check packing correctness.
        // bytes_per_vec is rounded up to a multiple of 8 for u64 alignment.
        let ndims = 16;
        let npoints = 4;
        // All dimensions positive -> all bits should be 1.
        let data: Vec<f32> = vec![1.0; npoints * ndims];
        let qd = train_and_quantize(&data, npoints, ndims);

        assert_eq!(qd.npoints, npoints);
        assert_eq!(qd.ndims, ndims);
        assert_eq!(qd.bytes_per_vec, 8); // ((16 + 63) / 64) * 8 = 8

        // With all-identical positive values, after training shift/scale the
        // quantization is deterministic. All bits for every point should be
        // the same since all values are identical.
        for i in 0..npoints {
            let v = qd.get(i);
            // All points should be identical.
            assert_eq!(v, qd.get(0), "point {} differs from point 0", i);
        }
    }

    #[test]
    fn test_quantize_1bit_roundtrip() {
        // Verify that get() and get_u64() return consistent data for the same point.
        let ndims = 64; // 8 bytes per vec -> exactly 1 u64
        let npoints = 10;
        let data = make_binary_data(npoints, ndims, 42);
        let qd = train_and_quantize(&data, npoints, ndims);

        assert_eq!(qd.bytes_per_vec, 8);
        assert_eq!(qd.u64s_per_vec(), 1);

        for i in 0..npoints {
            let bytes = qd.get(i);
            let u64s = qd.get_u64(i);

            // Convert the byte slice to a u64 (little-endian) and compare.
            let from_bytes = u64::from_le_bytes(bytes.try_into().unwrap());
            assert_eq!(
                from_bytes, u64s[0],
                "get() and get_u64() disagree for point {}",
                i
            );
        }
    }

    #[test]
    fn test_hamming_u64_identity() {
        // Hamming distance of a vector with itself is always 0.
        let a: Vec<u64> = vec![0xDEAD_BEEF_CAFE_BABE, 0x0123_4567_89AB_CDEF];
        assert_eq!(QuantizedData::hamming_u64(&a, &a), 0);

        let zeros: Vec<u64> = vec![0, 0, 0, 0];
        assert_eq!(QuantizedData::hamming_u64(&zeros, &zeros), 0);

        let ones: Vec<u64> = vec![u64::MAX, u64::MAX];
        assert_eq!(QuantizedData::hamming_u64(&ones, &ones), 0);
    }

    #[test]
    fn test_hamming_u64_all_different() {
        // XOR of all-zeros and all-ones gives all-ones, popcount = 64 per word.
        let a: Vec<u64> = vec![0u64; 3];
        let b: Vec<u64> = vec![u64::MAX; 3];
        assert_eq!(QuantizedData::hamming_u64(&a, &b), 64 * 3);
    }

    #[test]
    fn test_hamming_byte_matches_u64() {
        // The byte-based and u64-based Hamming distance should agree.
        let ndims = 128; // 16 bytes = 2 u64s
        let npoints = 5;
        let data = make_binary_data(npoints, ndims, 77);
        let qd = train_and_quantize(&data, npoints, ndims);

        for i in 0..npoints {
            for j in 0..npoints {
                let d_byte = QuantizedData::hamming(qd.get(i), qd.get(j));
                let d_u64 = QuantizedData::hamming_u64(qd.get_u64(i), qd.get_u64(j));
                assert_eq!(
                    d_byte, d_u64,
                    "hamming mismatch for ({}, {}): byte={} u64={}",
                    i, j, d_byte, d_u64
                );
            }
        }
    }

    #[test]
    fn test_compute_distance_matrix() {
        // Verify symmetry and diagonal (f32::MAX) of the distance matrix.
        let ndims = 64;
        let npoints = 8;
        let data = make_binary_data(npoints, ndims, 55);
        let qd = train_and_quantize(&data, npoints, ndims);

        let indices: Vec<usize> = (0..npoints).collect();
        let dist = qd.compute_distance_matrix(&indices);

        let n = npoints;
        // Diagonal must be f32::MAX.
        for i in 0..n {
            assert_eq!(
                dist[i * n + i],
                f32::MAX,
                "diagonal at ({},{}) is not f32::MAX",
                i,
                i
            );
        }
        // Symmetry: dist[i][j] == dist[j][i]
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(
                    dist[i * n + j], dist[j * n + i],
                    "asymmetry at ({}, {})",
                    i, j
                );
            }
        }
        // Non-negative off-diagonal.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        dist[i * n + j] >= 0.0,
                        "negative distance at ({}, {}): {}",
                        i,
                        j,
                        dist[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_quantize_1bit_single_point() {
        let ndims = 8;
        let data = vec![1.0f32; ndims];
        let qd = train_and_quantize(&data, 1, ndims);
        assert_eq!(qd.npoints, 1, "should have 1 point");
        assert_eq!(qd.ndims, ndims, "should preserve ndims");
        assert_eq!(qd.bytes_per_vec, 8, "bytes_per_vec should be 8 (rounded up to multiple of 8)");
    }

    #[test]
    fn test_quantize_1bit_single_dim() {
        // ndims=1, bytes_per_vec should round up to 8 (multiple of 8 for u64 alignment).
        let npoints = 5;
        let data: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 0.0];
        let qd = train_and_quantize(&data, npoints, 1);
        assert_eq!(qd.ndims, 1, "should preserve ndims=1");
        assert_eq!(qd.bytes_per_vec, 8, "bytes_per_vec for ndims=1 should be 8");
        assert_eq!(qd.npoints, npoints, "should have correct npoints");
    }

    #[test]
    fn test_quantize_1bit_large_ndims() {
        // ndims=1024, bytes_per_vec = ceil(1024/64) * 8 = 16 * 8 = 128.
        let ndims = 1024;
        let npoints = 3;
        let data = make_binary_data(npoints, ndims, 42);
        let qd = train_and_quantize(&data, npoints, ndims);
        assert_eq!(qd.bytes_per_vec, 128, "bytes_per_vec for ndims=1024 should be 128");
        assert_eq!(qd.u64s_per_vec(), 16, "u64s_per_vec for ndims=1024 should be 16");
    }

    #[test]
    fn test_quantize_zero_variance() {
        // All identical data -- should not crash due to zero-variance guard.
        let npoints = 10;
        let ndims = 8;
        let data = vec![42.0f32; npoints * ndims];
        let qd = train_and_quantize(&data, npoints, ndims);
        assert_eq!(qd.npoints, npoints, "should succeed with zero-variance data");
        // All points should have identical bit patterns.
        for i in 1..npoints {
            assert_eq!(
                qd.get(i), qd.get(0),
                "zero-variance data should produce identical quantized vectors"
            );
        }
    }

    #[test]
    fn test_quantize_negative_data() {
        // All negative values should produce valid quantized data.
        let npoints = 5;
        let ndims = 16;
        let data = vec![-5.0f32; npoints * ndims];
        let qd = train_and_quantize(&data, npoints, ndims);
        assert_eq!(qd.npoints, npoints, "should succeed with all-negative data");
    }

    #[test]
    fn test_hamming_single_bit_diff() {
        // XOR with exactly 1 bit different should give distance 1.
        let a: Vec<u64> = vec![0b0000_0000];
        let b: Vec<u64> = vec![0b0000_0001];
        assert_eq!(
            QuantizedData::hamming_u64(&a, &b), 1,
            "single bit difference should yield Hamming distance 1"
        );
    }

    #[test]
    fn test_compute_distance_matrix_single_point() {
        let ndims = 64;
        let data = make_binary_data(1, ndims, 42);
        let qd = train_and_quantize(&data, 1, ndims);
        let indices = vec![0];
        let dist = qd.compute_distance_matrix(&indices);
        assert_eq!(dist.len(), 1, "1x1 matrix should have 1 element");
        assert_eq!(dist[0], f32::MAX, "diagonal for single point should be f32::MAX");
    }

    #[test]
    fn test_compute_distance_matrix_two_identical() {
        // Two identical points should have distance 0.
        let ndims = 64;
        let npoints = 2;
        let data = vec![1.0f32; npoints * ndims]; // identical
        let qd = train_and_quantize(&data, npoints, ndims);
        let indices = vec![0, 1];
        let dist = qd.compute_distance_matrix(&indices);
        assert_eq!(
            dist[0 * 2 + 1], 0.0,
            "two identical quantized vectors should have Hamming distance 0"
        );
        assert_eq!(
            dist[1 * 2 + 0], 0.0,
            "symmetric: two identical quantized vectors should have Hamming distance 0"
        );
    }

    #[test]
    fn test_distances_to_leaders_empty() {
        let ndims = 64;
        let data = make_binary_data(3, ndims, 42);
        let qd = train_and_quantize(&data, 3, ndims);
        let leader_indices: Vec<usize> = vec![];
        let mut out: Vec<f32> = vec![];
        // Should not crash with empty leader list.
        qd.distances_to_leaders(0, &leader_indices, &mut out);
        assert!(out.is_empty(), "empty leader list should produce empty output");
    }

    #[test]
    fn test_bytes_per_vec_alignment() {
        // Verify bytes_per_vec is always a multiple of 8 for various ndims.
        for ndims in [1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 512, 1024] {
            let data = vec![0.0f32; ndims];
            let qd = train_and_quantize(&data, 1, ndims);
            assert_eq!(
                qd.bytes_per_vec % 8, 0,
                "bytes_per_vec ({}) should be a multiple of 8 for ndims={}",
                qd.bytes_per_vec, ndims
            );
        }
    }

    #[test]
    fn test_distances_to_leaders() {
        // Verify distances_to_leaders matches manual pairwise computation.
        let ndims = 64;
        let npoints = 6;
        let data = make_binary_data(npoints, ndims, 33);
        let qd = train_and_quantize(&data, npoints, ndims);

        let point_idx = 0;
        let leader_indices: Vec<usize> = vec![1, 3, 5];
        let mut out = vec![0.0f32; leader_indices.len()];
        qd.distances_to_leaders(point_idx, &leader_indices, &mut out);

        // Compare with direct hamming_u64 computation.
        let pt = qd.get_u64(point_idx);
        for (j, &leader_idx) in leader_indices.iter().enumerate() {
            let ld = qd.get_u64(leader_idx);
            let expected = QuantizedData::hamming_u64(pt, ld) as f32;
            assert_eq!(
                out[j], expected,
                "distance to leader {} mismatch: got {}, expected {}",
                leader_idx, out[j], expected
            );
        }
    }
}
