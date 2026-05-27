//! Recall@10 benchmark for the MinMax recompress API.
//!
//! Loads a `(vectors, queries, groundtruth)` triple in DiskANN `.bin` format
//! and measures recall@10 of brute-force IP search over quantized vectors
//! produced via two pipelines:
//!
//!   1. Direct: f32 -> 8-bit -> brute force IP search.
//!   2. Recompressed: f32 -> 8-bit -> recompress -> {2,4}-bit -> brute force IP search.
//!
//! For (2), each `grid_scale` from the command line is evaluated.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example recompress_recall -- \
//!     <vectors.bin> <queries.bin> <groundtruth_u32.bin>
//! ```
//!
//! Optional 4th argument: comma-separated list of grid scales (default: "1.0,0.6").
//!
//! Files use the standard DiskANN binary layout: a header of two little-endian
//! `u32`s `(num_points, dim)`, followed by row-major data
//! (`f32` for vectors/queries, `u32` for groundtruth ids).

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Instant;

use diskann_quantization::CompressInto;
use diskann_quantization::algorithms::{Transform, transforms::NullTransform};
use diskann_quantization::distances;
use diskann_quantization::minmax::{Data, FullQuery, MinMaxIP, MinMaxQuantizer, Recompressor};
use diskann_quantization::num::Positive;
use diskann_utils::{Reborrow, ReborrowMut};
use diskann_vector::PureDistanceFunction;
use half::f16;
use rayon::prelude::*;

fn read_bin<T: bytemuck::Pod>(path: &Path) -> std::io::Result<(usize, usize, Vec<T>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let num = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(hdr[4..].try_into().unwrap()) as usize;
    let total_bytes = num * dim * std::mem::size_of::<T>();
    let mut buf = vec![0u8; total_bytes];
    f.read_exact(&mut buf)?;
    let mut out = vec![T::zeroed(); num * dim];
    bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&buf);
    Ok((num, dim, out))
}

/// Read a vector file that may be stored as either f32 or f16, returning f32.
fn read_vectors_as_f32(path: &Path, fp16: bool) -> std::io::Result<(usize, usize, Vec<f32>)> {
    if fp16 {
        let (n, d, raw) = read_bin::<u16>(path)?;
        let v: Vec<f32> = raw
            .into_iter()
            .map(|b| f16::from_bits(b).to_f32())
            .collect();
        Ok((n, d, v))
    } else {
        read_bin::<f32>(path)
    }
}

/// Read a groundtruth file. If `with_distances` is true, the file is laid out
/// as `num_queries * k * (u32 id + f32 distance)`. Otherwise just `u32` ids.
/// Returns `(num, k, ids)`.
fn read_groundtruth(
    path: &Path,
    with_distances: bool,
) -> std::io::Result<(usize, usize, Vec<u32>)> {
    if with_distances {
        let mut f = BufReader::new(File::open(path)?);
        let mut hdr = [0u8; 8];
        f.read_exact(&mut hdr)?;
        let num = u32::from_le_bytes(hdr[..4].try_into().unwrap()) as usize;
        let k = u32::from_le_bytes(hdr[4..].try_into().unwrap()) as usize;
        // First all ids (num*k * u32), then all distances (num*k * f32).
        let id_bytes = num * k * 4;
        let mut ids = vec![0u32; num * k];
        f.read_exact(bytemuck::cast_slice_mut::<u32, u8>(&mut ids))?;
        // Skip distances.
        let mut sink = vec![0u8; id_bytes];
        f.read_exact(&mut sink)?;
        Ok((num, k, ids))
    } else {
        read_bin::<u32>(path)
    }
}

/// Return the indices of the top-`k` largest IP scores (i.e. smallest `-IP`).
fn top_k_indices(scores: &[f32], k: usize) -> Vec<u32> {
    // partial sort by ascending score (since scores are -IP).
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    let pivot = k.min(idx.len() - 1);
    idx.select_nth_unstable_by(pivot, |&a, &b| {
        scores[a as usize]
            .partial_cmp(&scores[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<u32> = idx.into_iter().take(k).collect();
    top.sort_by(|&a, &b| {
        scores[a as usize]
            .partial_cmp(&scores[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top
}

fn recall_at_k(result: &[u32], truth: &[u32], k: usize) -> f32 {
    let truth_set: std::collections::HashSet<u32> = truth.iter().take(k).copied().collect();
    let hits = result
        .iter()
        .take(k)
        .filter(|i| truth_set.contains(i))
        .count();
    hits as f32 / k as f32
}

fn quantize_corpus<const NBITS: usize>(
    quantizer: &MinMaxQuantizer,
    corpus: &[f32],
    num: usize,
    dim: usize,
) -> Vec<Data<NBITS>>
where
    diskann_quantization::bits::Unsigned: diskann_quantization::bits::Representation<NBITS>,
    MinMaxQuantizer:
        for<'a, 'b> CompressInto<&'a [f32], diskann_quantization::minmax::DataMutRef<'b, NBITS>>,
{
    (0..num)
        .into_par_iter()
        .map(|i| {
            let mut d = Data::<NBITS>::new_boxed(dim);
            let slice = &corpus[i * dim..(i + 1) * dim];
            quantizer
                .compress_into(slice, d.reborrow_mut())
                .expect("compress");
            d
        })
        .collect()
}

fn recompress<const M: usize>(rc: Recompressor, src: &[Data<8>], dim: usize) -> Vec<Data<M>>
where
    diskann_quantization::bits::Unsigned: diskann_quantization::bits::Representation<M>,
    Recompressor: for<'a, 'b> CompressInto<
            diskann_quantization::minmax::DataRef<'a, 8>,
            diskann_quantization::minmax::DataMutRef<'b, M>,
            Output = (),
        >,
{
    src.par_iter()
        .map(|s| {
            let mut d = Data::<M>::new_boxed(dim);
            rc.compress_into(s.reborrow(), d.reborrow_mut())
                .expect("recompress");
            d
        })
        .collect()
}

fn build_queries(quantizer: &MinMaxQuantizer, q: &[f32], num: usize, dim: usize) -> Vec<FullQuery> {
    (0..num)
        .into_par_iter()
        .map(|i| {
            let mut fq = FullQuery::new_in(dim, diskann_quantization::alloc::GlobalAllocator)
                .expect("alloc full query");
            quantizer
                .compress_into(&q[i * dim..(i + 1) * dim], fq.reborrow_mut())
                .expect("compress query");
            fq
        })
        .collect()
}

fn evaluate_recall<const M: usize>(
    label: &str,
    queries: &[FullQuery],
    corpus: &[Data<M>],
    gt: &[u32],
    gt_k: usize,
    k: usize,
) where
    diskann_quantization::bits::Unsigned: diskann_quantization::bits::Representation<M>,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
            diskann_quantization::minmax::FullQueryRef<'a>,
            diskann_quantization::minmax::DataRef<'b, M>,
            distances::Result<f32>,
        >,
{
    let t0 = Instant::now();

    let recalls: Vec<f32> = queries
        .par_iter()
        .enumerate()
        .map(|(qi, q)| {
            let scores: Vec<f32> = corpus
                .iter()
                .map(|c| {
                    let s: distances::Result<f32> = MinMaxIP::evaluate(q.reborrow(), c.reborrow());
                    s.expect("evaluate")
                })
                .collect();
            let top = top_k_indices(&scores, k);
            recall_at_k(&top, &gt[qi * gt_k..(qi + 1) * gt_k], k)
        })
        .collect();

    let elapsed = t0.elapsed();
    let mean: f32 = recalls.iter().copied().sum::<f32>() / recalls.len() as f32;
    let min_r = recalls.iter().copied().fold(f32::INFINITY, f32::min);
    let max_r = recalls.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!(
        "{:<46} recall@{}: mean={:.4}  min={:.4}  max={:.4}  ({} queries, {:?})",
        label,
        k,
        mean,
        min_r,
        max_r,
        recalls.len(),
        elapsed
    );
}

fn main() -> std::io::Result<()> {
    let raw_args: Vec<String> = env::args().collect();
    // Split out boolean flags.
    let mut fp16 = false;
    let mut gt_with_distances = false;
    let args: Vec<String> = raw_args
        .into_iter()
        .filter(|a| match a.as_str() {
            "--fp16" => {
                fp16 = true;
                false
            }
            "--gt-with-distances" => {
                gt_with_distances = true;
                false
            }
            _ => true,
        })
        .collect();
    if args.len() < 4 {
        eprintln!(
            "usage: {} [--fp16] [--gt-with-distances] <vectors.bin> <queries.bin> <groundtruth.bin> [grid_scales=1.0,0.6]",
            args.first()
                .map(String::as_str)
                .unwrap_or("recompress_recall")
        );
        std::process::exit(2);
    }
    let vectors_path = Path::new(&args[1]);
    let queries_path = Path::new(&args[2]);
    let gt_path = Path::new(&args[3]);
    let grid_scales: Vec<f32> = if args.len() > 4 {
        args[4]
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect()
    } else {
        vec![1.0, 0.6]
    };
    let k = 10usize;

    println!(
        "Loading vectors from {:?} (fp16={}) ...",
        vectors_path, fp16
    );
    let (n_v, dim_v, vectors) = read_vectors_as_f32(vectors_path, fp16)?;
    println!("  -> {} vectors of dim {}", n_v, dim_v);

    println!(
        "Loading queries from {:?} (fp16={}) ...",
        queries_path, fp16
    );
    let (n_q, dim_q, queries) = read_vectors_as_f32(queries_path, fp16)?;
    println!("  -> {} queries of dim {}", n_q, dim_q);
    assert_eq!(dim_v, dim_q, "vector / query dim mismatch");

    println!(
        "Loading groundtruth from {:?} (with_distances={}) ...",
        gt_path, gt_with_distances
    );
    let (n_gt, gt_k, gt) = read_groundtruth(gt_path, gt_with_distances)?;
    println!("  -> {} groundtruth rows of {} neighbors each", n_gt, gt_k);
    assert_eq!(n_gt, n_q, "groundtruth row count must match query count");
    assert!(gt_k >= k, "groundtruth k must be >= {k}");

    let dim = dim_v;
    let quantizer_1 = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
        Positive::new(1.0).unwrap(),
    );

    println!("\nQuantizing corpus to 8-bit (grid_scale=1.0) ...");
    let t = Instant::now();
    let corpus_8 = quantize_corpus::<8>(&quantizer_1, &vectors, n_v, dim);
    println!("  done in {:?}", t.elapsed());

    println!("Building FullQuery (uncompressed) queries ...");
    let t = Instant::now();
    let full_queries = build_queries(&quantizer_1, &queries, n_q, dim);
    println!("  done in {:?}", t.elapsed());

    println!("\n--- Recall@{k} (brute-force IP) ---");

    // Baseline: 8-bit direct.
    evaluate_recall::<8>(
        "f32 -> 8-bit (direct, g=1.0)",
        &full_queries,
        &corpus_8,
        &gt,
        gt_k,
        k,
    );

    for &g in &grid_scales {
        let pg = Positive::new(g).unwrap_or_else(|_| panic!("grid_scale must be > 0; got {g}"));

        // Direct compression baselines at the same grid_scale.
        let direct_q = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            pg,
        );

        // f32 -> 4 (direct)
        let t = Instant::now();
        let direct_4 = quantize_corpus::<4>(&direct_q, &vectors, n_v, dim);
        let d_time = t.elapsed();
        evaluate_recall::<4>(
            &format!("f32 -> 4 (direct)            g={g:>4.2}  (compress   {d_time:?})"),
            &full_queries,
            &direct_4,
            &gt,
            gt_k,
            k,
        );

        // f32 -> 2 (direct)
        let t = Instant::now();
        let direct_2 = quantize_corpus::<2>(&direct_q, &vectors, n_v, dim);
        let d_time = t.elapsed();
        evaluate_recall::<2>(
            &format!("f32 -> 2 (direct)            g={g:>4.2}  (compress   {d_time:?})"),
            &full_queries,
            &direct_2,
            &gt,
            gt_k,
            k,
        );

        // Recompressed pipelines reusing the 8-bit (g=1.0) corpus.
        let rc = Recompressor::new(pg);

        let t = Instant::now();
        let corpus_4 = recompress::<4>(rc, &corpus_8, dim);
        let rc_time = t.elapsed();
        evaluate_recall::<4>(
            &format!("8 -> 4   (recompress)        g={g:>4.2}  (recompress {rc_time:?})"),
            &full_queries,
            &corpus_4,
            &gt,
            gt_k,
            k,
        );

        let t = Instant::now();
        let corpus_2 = recompress::<2>(rc, &corpus_8, dim);
        let rc_time = t.elapsed();
        evaluate_recall::<2>(
            &format!("8 -> 2   (recompress)        g={g:>4.2}  (recompress {rc_time:?})"),
            &full_queries,
            &corpus_2,
            &gt,
            gt_k,
            k,
        );
    }

    Ok(())
}
