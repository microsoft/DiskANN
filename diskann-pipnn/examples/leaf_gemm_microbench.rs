//! Microbench for leaf-GEMM variants. Synthetic random f32 data.
//!
//! Compares three pipelines on the same input:
//!   A: full GEMM via `sgemm_aat` + per-row SIMD top-k (current v1)
//!   B: triangular GEMM via `sgemm_aat_lower` + symmetrize + per-row SIMD top-k (v2)
//!   C: triangular GEMM via `sgemm_aat_lower` + NO symmetrize +
//!      two-segment top-k (row prefix + column suffix, both scalar)
//!
//! Usage:
//!   cargo run --release --example leaf_gemm_microbench -- \
//!     [n=1024,1536,2048] [d=128] [k=3] [reps=20]

use std::env;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn gen_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    // Cheap deterministic generator (linear congruential). No real Gaussian.
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = vec![0.0f32; n * d];
    for v in out.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (state >> 32) as u32;
        // [0, 1) then center
        *v = (bits as f32) / (u32::MAX as f32) - 0.5;
    }
    out
}

fn pre_norms_sq(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        let mut s = 0.0;
        for v in row { s += v * v; }
        out[i] = s;
    }
    out
}

// ----- Top-k helpers (simple but identical across variants) -----

/// In-place top-3 tracker over a contiguous row of length `n`, skipping self_idx.
/// Heap kept sorted ASCENDING by dist (out[0] is closest).
fn topk3_row(row: &[f32], n: usize, self_idx: usize, out: &mut [(u32, f32)]) {
    debug_assert!(out.len() == 3);
    let inf = f32::MAX;
    let mut h: [(u32, f32); 3] = [(u32::MAX, inf); 3];
    for j in 0..n {
        if j == self_idx { continue; }
        let d = row[j];
        if d < h[2].1 {
            h[2] = (j as u32, d);
            if h[2].1 < h[1].1 { h.swap(1, 2); }
            if h[1].1 < h[0].1 { h.swap(0, 1); }
        }
    }
    out.copy_from_slice(&h);
}

/// Top-3 over two segments: a contiguous prefix `row[0..mid]` and a strided
/// column `dot[mid*n + col_off + s*n]` for s in 0..suffix_len.
/// Used by variant C (no symmetrize) to read row i's distances from the lower
/// triangle: prefix = row i columns 0..=i, strided = column i rows i+1..n.
fn topk3_split(
    prefix: &[f32],
    strided: &[f32],
    stride: usize,
    suffix_len: usize,
    self_idx: usize,
    out: &mut [(u32, f32)],
) {
    debug_assert!(out.len() == 3);
    let prefix_len = prefix.len();
    let mut h: [(u32, f32); 3] = [(u32::MAX, f32::MAX); 3];
    for j in 0..prefix_len {
        if j == self_idx { continue; }
        let d = prefix[j];
        if d < h[2].1 {
            h[2] = (j as u32, d);
            if h[2].1 < h[1].1 { h.swap(1, 2); }
            if h[1].1 < h[0].1 { h.swap(0, 1); }
        }
    }
    // strided[s] is at index 0, stride, 2*stride, ...
    let base = prefix_len; // column index for these entries is prefix_len, prefix_len+1, ...
    for s in 0..suffix_len {
        let abs_j = base + s;
        if abs_j == self_idx { continue; }
        let d = strided[s * stride];
        if d < h[2].1 {
            h[2] = (abs_j as u32, d);
            if h[2].1 < h[1].1 { h.swap(1, 2); }
            if h[1].1 < h[0].1 { h.swap(0, 1); }
        }
    }
    out.copy_from_slice(&h);
}

// ----- Variants -----

fn convert_l2_row(row: &mut [f32], n: usize, ni: f32, norms: &[f32]) {
    for j in 0..n {
        // d² = ni + nj - 2 dot
        row[j] = (ni + norms[j] - 2.0 * row[j]).max(0.0);
    }
}

/// A: full GEMM via `sgemm_aat` + per-row SIMD top-k.
fn variant_a(data: &[f32], n: usize, d: usize, dot: &mut [f32], norms: &[f32], knn: &mut [(u32, f32)]) {
    diskann_linalg::sgemm_aat(data, n, d, dot);
    for i in 0..n {
        let row = &mut dot[i * n..(i + 1) * n];
        let ni = norms[i];
        convert_l2_row(row, n, ni, norms);
        let out = &mut knn[i * 3..(i + 1) * 3];
        topk3_row(row, n, i, out);
    }
}

fn symmetrize_lower_to_upper(dot: &mut [f32], n: usize) {
    for i in 0..n {
        for j in (i + 1)..n {
            unsafe {
                let src = *dot.get_unchecked(j * n + i);
                *dot.get_unchecked_mut(i * n + j) = src;
            }
        }
    }
}

/// B: triangular GEMM + symmetrize + per-row SIMD top-k.
fn variant_b(data: &[f32], n: usize, d: usize, dot: &mut [f32], norms: &[f32], knn: &mut [(u32, f32)]) {
    diskann_linalg::sgemm_aat_lower(data, n, d, dot);
    symmetrize_lower_to_upper(dot, n);
    for i in 0..n {
        let row = &mut dot[i * n..(i + 1) * n];
        let ni = norms[i];
        convert_l2_row(row, n, ni, norms);
        let out = &mut knn[i * 3..(i + 1) * 3];
        topk3_row(row, n, i, out);
    }
}

/// C: triangular GEMM + NO symmetrize + two-segment top-k per row.
/// For row i, distances come from:
///   - row i columns 0..=i (lower-triangle entries in row i)
///   - column i rows i+1..n (lower-triangle entries in column i below diagonal)
/// We convert the row prefix in-place and gather + convert the column suffix
/// into a temp buffer to share the same convert kernel.
fn variant_c(data: &[f32], n: usize, d: usize, dot: &mut [f32], norms: &[f32], knn: &mut [(u32, f32)], scratch: &mut [f32]) {
    diskann_linalg::sgemm_aat_lower(data, n, d, dot);
    debug_assert!(scratch.len() >= n);
    for i in 0..n {
        let prefix_len = i + 1; // columns 0..=i are valid in row i
        let suffix_len = n - prefix_len; // columns i+1..n come from column i below diag
        let ni = norms[i];

        // Convert row prefix in-place.
        {
            let row = &mut dot[i * n..i * n + prefix_len];
            for j in 0..prefix_len {
                row[j] = (ni + norms[j] - 2.0 * row[j]).max(0.0);
            }
        }

        // Gather + convert column i below diagonal into scratch[0..suffix_len].
        // dot[(i+1)*n + i], dot[(i+2)*n + i], ...
        let strided_base = (i + 1) * n + i;
        for s in 0..suffix_len {
            let abs_j = prefix_len + s;
            let raw = dot[strided_base + s * n];
            scratch[s] = (ni + norms[abs_j] - 2.0 * raw).max(0.0);
        }

        let prefix = &dot[i * n..i * n + prefix_len];
        let strided = &scratch[..suffix_len];
        let out = &mut knn[i * 3..(i + 1) * 3];
        topk3_split(prefix, strided, 1, suffix_len, i, out);
    }
}

fn time_repeated<F: FnMut()>(mut f: F, reps: usize) -> Vec<f64> {
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts
}

fn run_for_n(n: usize, d: usize, reps: usize) {
    println!("\n=== n={n} d={d} reps={reps} ===");
    let data = gen_data(n, d, 0xC0DEC0DE);
    let norms = pre_norms_sq(&data, n, d);
    let mut dot = vec![0.0f32; n * n];
    let mut knn_a = vec![(u32::MAX, f32::MAX); n * 3];
    let mut knn_b = vec![(u32::MAX, f32::MAX); n * 3];
    let mut knn_c = vec![(u32::MAX, f32::MAX); n * 3];
    let mut scratch = vec![0.0f32; n];

    // Warmup
    variant_a(&data, n, d, &mut dot, &norms, &mut knn_a);
    variant_b(&data, n, d, &mut dot, &norms, &mut knn_b);
    variant_c(&data, n, d, &mut dot, &norms, &mut knn_c, &mut scratch);

    let ta = time_repeated(|| { variant_a(&data, n, d, &mut dot, &norms, &mut knn_a); }, reps);
    let tb = time_repeated(|| { variant_b(&data, n, d, &mut dot, &norms, &mut knn_b); }, reps);
    let tc = time_repeated(|| { variant_c(&data, n, d, &mut dot, &norms, &mut knn_c, &mut scratch); }, reps);

    let median = |xs: &[f64]| xs[xs.len() / 2];
    let min = |xs: &[f64]| xs[0];
    let max = |xs: &[f64]| xs[xs.len() - 1];

    println!("{:30}  min        median    max", "");
    println!("A full+SIMD topk    : {:8.3}ms  {:8.3}ms  {:8.3}ms", min(&ta), median(&ta), max(&ta));
    println!("B tri+sym+SIMD topk : {:8.3}ms  {:8.3}ms  {:8.3}ms   ({:+.1}% vs A)",
        min(&tb), median(&tb), max(&tb), 100.0 * (median(&tb) - median(&ta)) / median(&ta));
    println!("C tri+gather topk   : {:8.3}ms  {:8.3}ms  {:8.3}ms   ({:+.1}% vs A)",
        min(&tc), median(&tc), max(&tc), 100.0 * (median(&tc) - median(&ta)) / median(&ta));

    // Quality check: do A and B produce the same knn (modulo numeric noise)?
    let mut diffs = 0usize;
    let mut max_dist_diff = 0.0f32;
    for i in 0..n {
        for k in 0..3 {
            let a = knn_a[i * 3 + k];
            let b = knn_b[i * 3 + k];
            if a.0 != b.0 {
                diffs += 1;
            }
            let dd = (a.1 - b.1).abs();
            if dd > max_dist_diff { max_dist_diff = dd; }
        }
    }
    let mut diffs_ac = 0usize;
    let mut max_dist_diff_ac = 0.0f32;
    for i in 0..n {
        for k in 0..3 {
            let a = knn_a[i * 3 + k];
            let c = knn_c[i * 3 + k];
            if a.0 != c.0 { diffs_ac += 1; }
            let dd = (a.1 - c.1).abs();
            if dd > max_dist_diff_ac { max_dist_diff_ac = dd; }
        }
    }
    println!("Quality A vs B  :  id_diffs={}  max_dist_diff={:.3e}", diffs, max_dist_diff);
    println!("Quality A vs C  :  id_diffs={}  max_dist_diff={:.3e}", diffs_ac, max_dist_diff_ac);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let ns_arg = args.get(1).cloned().unwrap_or_else(|| "1024,1536,2048".to_string());
    let d: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let _k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let reps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);

    for ns in ns_arg.split(',') {
        let n: usize = ns.trim().parse().expect("n must be int");
        run_for_n(n, d, reps);
    }
}
