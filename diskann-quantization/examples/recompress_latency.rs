//! Microbenchmark: legacy vs new (grid_scale-aware) recompress kernel.
//!
//! Loads a vector file in DiskANN `.bin` format, quantizes it to 8-bit,
//! and runs the recompression pipeline (8 -> 4 and 8 -> 2) many times
//! through both kernels to measure per-vector latency.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example recompress_latency -- <vectors.bin> [iters=20]
//! ```
//!
//! `iters` is the number of full-corpus passes over which the average is taken.

use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufReader, Read};
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::{Duration, Instant};

use diskann_quantization::CompressInto;
use diskann_quantization::algorithms::{Transform, transforms::NullTransform};
use diskann_quantization::bits::{Representation, Unsigned};
use diskann_quantization::minmax::{
    Data, DataMutRef, DataRef, MinMaxCompensation, MinMaxQuantizer,
};
use diskann_quantization::num::Positive;
use diskann_quantization::scalar::bit_scale;
use diskann_utils::{Reborrow, ReborrowMut};
use half::f16;

fn read_bin_f32(path: &Path) -> std::io::Result<(usize, usize, Vec<f32>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let num = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[4..].try_into().unwrap()) as usize;
    let total_bytes = num * dim * std::mem::size_of::<f32>();
    let mut buf = vec![0u8; total_bytes];
    f.read_exact(&mut buf)?;
    let mut out = vec![0f32; num * dim];
    bytemuck::cast_slice_mut::<f32, u8>(&mut out).copy_from_slice(&buf);
    Ok((num, dim, out))
}

fn read_bin_fp16_as_f32(path: &Path) -> std::io::Result<(usize, usize, Vec<f32>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let num = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[4..].try_into().unwrap()) as usize;
    let total_bytes = num * dim * 2;
    let mut buf = vec![0u8; total_bytes];
    f.read_exact(&mut buf)?;
    let mut raw = vec![0u16; num * dim];
    bytemuck::cast_slice_mut::<u16, u8>(&mut raw).copy_from_slice(&buf);
    let out: Vec<f32> = raw
        .into_iter()
        .map(|b| f16::from_bits(b).to_f32())
        .collect();
    Ok((num, dim, out))
}

/// Verbatim copy of the *legacy* (pre-grid_scale) kernel for fair comparison.
/// This is mathematically equivalent to the new kernel at `grid_scale = 1.0`.
#[inline(always)]
fn legacy_recompress_kernel<const N: usize, const M: usize>(
    from: DataRef<'_, N>,
    mut to: DataMutRef<'_, M>,
) where
    Unsigned: Representation<N> + Representation<M>,
{
    let dim = from.len();
    assert_eq!(dim, to.vector().len());

    let src_meta = from.meta();
    let src_a = src_meta.a;
    let src_b = src_meta.b;

    let scale_n = bit_scale::<N>();
    let scale_m = bit_scale::<M>();
    let code_scale = scale_m / scale_n;

    let new_a = src_a / code_scale;
    let new_b = src_b;

    let from_vec = from.vector();
    let mut to_vec = to.vector_mut();

    let mut code_sum: f32 = 0.0;
    let mut norm_squared: f32 = 0.0;

    for i in 0..dim {
        // SAFETY: bounds checked above
        let old_code = unsafe { from_vec.get_unchecked(i) };
        let old_code_f = old_code as f32;

        let new_code_pre = (old_code_f * code_scale).round_ties_even();
        let new_code = new_code_pre as u8;

        // SAFETY: bounds checked above
        unsafe { to_vec.set_unchecked(i, new_code) };

        let new_code_f = new_code as f32;
        code_sum += new_code_f;

        let v_m = new_code_f * new_a + new_b;
        norm_squared += v_m * v_m;
    }

    to.set_meta(MinMaxCompensation {
        dim: dim as u32,
        b: new_b,
        a: new_a,
        n: new_a * code_sum,
        norm_squared,
    });
}

fn quantize_corpus_8(
    quantizer: &MinMaxQuantizer,
    corpus: &[f32],
    num: usize,
    dim: usize,
) -> Vec<Data<8>> {
    (0..num)
        .map(|i| {
            let mut d = Data::<8>::new_boxed(dim);
            quantizer
                .compress_into(&corpus[i * dim..(i + 1) * dim], d.reborrow_mut())
                .expect("compress 8");
            d
        })
        .collect()
}

fn bench<const M: usize, F>(
    label: &str,
    iters: usize,
    src: &[Data<8>],
    dim: usize,
    mut run: F,
) -> Duration
where
    Unsigned: Representation<M>,
    F: FnMut(DataRef<'_, 8>, DataMutRef<'_, M>),
{
    // Pre-allocate the destination buffers (allocation is *not* counted).
    let mut dst: Vec<Data<M>> = (0..src.len()).map(|_| Data::<M>::new_boxed(dim)).collect();

    // Warm-up.
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        run(s.reborrow(), d.reborrow_mut());
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            run(black_box(s.reborrow()), d.reborrow_mut());
        }
        // Prevent the loop from being optimized away.
        black_box(&dst);
    }
    let total = t0.elapsed();

    let total_vectors = iters * src.len();
    let per_vec_ns = total.as_nanos() as f64 / total_vectors as f64;
    let per_vec_per_dim_ns = per_vec_ns / dim as f64;
    println!(
        "  {:<32} total={:?}  ({} vectors)  per-vector={:.2} ns  per-dim={:.3} ns",
        label, total, total_vectors, per_vec_ns, per_vec_per_dim_ns
    );
    total
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut fp16 = false;
    let args: Vec<String> = args
        .into_iter()
        .filter(|a| {
            if a == "--fp16" {
                fp16 = true;
                false
            } else {
                true
            }
        })
        .collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} [--fp16] <vectors.bin> [iters=20]",
            args.first()
                .map(String::as_str)
                .unwrap_or("recompress_latency")
        );
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    println!("Loading vectors from {:?} (fp16={}) ...", path, fp16);
    let (n, dim, vectors) = if fp16 {
        read_bin_fp16_as_f32(path)?
    } else {
        read_bin_f32(path)?
    };
    println!("  -> {} vectors of dim {}", n, dim);

    let quantizer = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
        Positive::new(1.0).unwrap(),
    );
    let corpus_8 = quantize_corpus_8(&quantizer, &vectors, n, dim);

    println!(
        "\nRunning {} iters over {} vectors (total {} vector-ops per kernel)",
        iters,
        n,
        iters * n
    );

    println!("\n--- 8 -> 4 ---");
    let t_legacy_4 = bench::<4, _>("legacy (no grid_scale)", iters, &corpus_8, dim, |s, d| {
        legacy_recompress_kernel::<8, 4>(s, d)
    });
    // New path at g=1.0
    let rc_1 = diskann_quantization::minmax::Recompressor::new(Positive::new(1.0).unwrap());
    let t_new_4_g1 = bench::<4, _>("new kernel, g=1.0", iters, &corpus_8, dim, |s, d| {
        rc_1.compress_into(s, d).expect("recompress");
    });
    let rc_06 = diskann_quantization::minmax::Recompressor::new(Positive::new(0.6).unwrap());
    let t_new_4_g06 = bench::<4, _>("new kernel, g=0.6", iters, &corpus_8, dim, |s, d| {
        rc_06.compress_into(s, d).expect("recompress");
    });

    println!("\n--- 8 -> 2 ---");
    let t_legacy_2 = bench::<2, _>("legacy (no grid_scale)", iters, &corpus_8, dim, |s, d| {
        legacy_recompress_kernel::<8, 2>(s, d)
    });
    let t_new_2_g1 = bench::<2, _>("new kernel, g=1.0", iters, &corpus_8, dim, |s, d| {
        rc_1.compress_into(s, d).expect("recompress");
    });
    let t_new_2_g06 = bench::<2, _>("new kernel, g=0.6", iters, &corpus_8, dim, |s, d| {
        rc_06.compress_into(s, d).expect("recompress");
    });

    // Average per-vector latency in nanoseconds.
    let per_vec = |d: Duration| d.as_nanos() as f64 / (iters * n) as f64;

    println!("\n--- Summary (avg per-vector latency) ---");
    for (label, l, n1, n06) in [
        ("8 -> 4", t_legacy_4, t_new_4_g1, t_new_4_g06),
        ("8 -> 2", t_legacy_2, t_new_2_g1, t_new_2_g06),
    ] {
        let pl = per_vec(l);
        let p1 = per_vec(n1);
        let p06 = per_vec(n06);
        let delta_g1 = p1 - pl;
        let delta_g06 = p06 - pl;
        let pct_g1 = 100.0 * delta_g1 / pl;
        let pct_g06 = 100.0 * delta_g06 / pl;
        println!(
            "  {}:  legacy={:>7.2}ns  new(g=1.0)={:>7.2}ns ({:+.2}ns / {:+.2}%)  new(g=0.6)={:>7.2}ns ({:+.2}ns / {:+.2}%)",
            label, pl, p1, delta_g1, pct_g1, p06, delta_g06, pct_g06
        );
    }

    Ok(())
}
