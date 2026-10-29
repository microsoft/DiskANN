/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Translated point tiles scored against shared prepared centers.

use crate::ivf::metrics::ComputeStats;

pub(super) const GEMM_ROWS: usize = 128;
pub(super) const GEMM_MIN_CENTERS: usize = 8;

/// Replace labels by scoring all centers for each point tile.
/// Center-zero translation reduces cancellation, but does not ensure exact labels.
/// Count is a work estimate and capacity hint; actual rows determine output length.
/// Preparation, allocation, scheduling and SGEMM belong to assignment timing.
pub(super) fn assign_gemm<'a>(
    mut points: impl Iterator<Item = &'a [f32]>,
    centers: &[f32],
    dim: usize,
    count: usize,
    labels: &mut Vec<usize>,
    stats: &mut ComputeStats,
) -> crate::ANNResult<()> {
    debug_assert!(dim > 0 && centers.len() >= 2 * dim);
    debug_assert_eq!(centers.len() % dim, 0);
    labels.clear();
    let Some(prepared) = PreparedCenters::new(centers, dim) else {
        labels.extend(points.map(|x| {
            stats.numeric_fallbacks += 1;
            super::nearest(x, centers, dim)
        }));
        return Ok(());
    };

    #[cfg(feature = "ivf-parallel")]
    if count > GEMM_ROWS
        && super::super::use_parallel(
            count
                .saturating_mul(dim)
                .saturating_mul(prepared.norms.len()),
        )
    {
        return assign_parallel(points, &prepared, count, labels, stats);
    }
    let mut scratch = GemmScratch::new(dim, prepared.norms.len())?;
    let mut rows = Vec::with_capacity(GEMM_ROWS.min(count));
    loop {
        rows.clear();
        rows.extend(points.by_ref().take(GEMM_ROWS));
        if rows.is_empty() {
            break;
        }
        let start = labels.len();
        labels.resize(start + rows.len(), 0);
        stats.merge(assign_tile(
            &prepared,
            &rows,
            &mut labels[start..],
            &mut scratch,
        )?);
    }
    Ok(())
}

struct PreparedCenters<'a> {
    original: &'a [f32],
    dim: usize,
    packed: Vec<f32>,
    norms: Vec<f64>,
}

impl<'a> PreparedCenters<'a> {
    fn new(centers: &'a [f32], dim: usize) -> Option<Self> {
        let origin = &centers[..dim];
        let mut packed = Vec::with_capacity(centers.len());
        let mut norms = Vec::with_capacity(centers.len() / dim);
        let mut valid = true;
        for center in centers.chunks_exact(dim) {
            let mut norm = 0.0f64;
            for (&c, &o) in center.iter().zip(origin) {
                let shifted = c - o;
                valid &= shifted.is_finite();
                packed.push(shifted);
                norm += f64::from(shifted) * f64::from(shifted);
            }
            norms.push(norm);
            valid &= norm.is_finite();
        }
        valid.then_some(Self {
            original: centers,
            dim,
            packed,
            norms,
        })
    }
}

struct GemmScratch {
    packed: Vec<f32>,
    scores: Vec<f32>,
    valid_rows: Vec<bool>,
}

impl GemmScratch {
    fn new(dim: usize, centers: usize) -> crate::ANNResult<Self> {
        use crate::ANNError;

        let packed_len = GEMM_ROWS
            .checked_mul(dim)
            .ok_or_else(|| ANNError::message("IVF GEMM point tile overflow"))?;
        let score_len = GEMM_ROWS
            .checked_mul(centers)
            .ok_or_else(|| ANNError::message("IVF GEMM score tile overflow"))?;
        Ok(Self {
            packed: Vec::with_capacity(packed_len),
            scores: vec![0.0; score_len],
            valid_rows: Vec::with_capacity(GEMM_ROWS),
        })
    }
}

fn assign_tile(
    centers: &PreparedCenters<'_>,
    rows: &[&[f32]],
    labels: &mut [usize],
    scratch: &mut GemmScratch,
) -> crate::ANNResult<ComputeStats> {
    use diskann_linalg::{Transpose, sgemm};

    use super::nearest;
    use crate::ANNError;

    debug_assert!(rows.len() <= GEMM_ROWS);
    debug_assert_eq!(rows.len(), labels.len());
    let origin = &centers.original[..centers.dim];
    let GemmScratch {
        packed,
        scores,
        valid_rows,
    } = scratch;
    packed.clear();
    valid_rows.clear();
    for &point in rows {
        debug_assert_eq!(point.len(), centers.dim);
        let start = packed.len();
        let mut valid = true;
        for (&value, &offset) in point.iter().zip(origin) {
            let shifted = value - offset;
            valid &= shifted.is_finite();
            packed.push(shifted);
        }
        // Keep overflowed rows out of SGEMM and recompute their labels below.
        if !valid {
            packed[start..].fill(0.0);
        }
        valid_rows.push(valid);
    }
    let m = rows.len();
    let k = centers.norms.len();
    sgemm(
        Transpose::None,
        Transpose::Ordinary,
        m,
        k,
        centers.dim,
        1.0,
        packed,
        &centers.packed,
        None,
        &mut scores[..m * k],
    )
    .map_err(ANNError::new)?;
    let mut stats = ComputeStats {
        gemm_calls: 1,
        ..ComputeStats::default()
    };
    for (r, ((&point, &valid), label)) in rows.iter().zip(valid_rows.iter()).zip(labels).enumerate()
    {
        let mut best = 0;
        let mut best_score = f64::INFINITY;
        let mut finite = valid;
        for (j, &dot) in scores[r * k..(r + 1) * k].iter().enumerate() {
            // The point norm is common to all centers. Negative scores are valid.
            let score = centers.norms[j] - 2.0 * f64::from(dot);
            finite &= score.is_finite();
            if score < best_score {
                best = j;
                best_score = score;
            }
        }
        if finite {
            *label = best;
            stats.gemm_points += 1;
        } else {
            *label = nearest(point, centers.original, centers.dim);
            stats.numeric_fallbacks += 1;
        }
    }
    Ok(stats)
}

#[cfg(feature = "ivf-parallel")]
fn assign_parallel<'a>(
    points: impl Iterator<Item = &'a [f32]>,
    centers: &PreparedCenters<'_>,
    count: usize,
    labels: &mut Vec<usize>,
    stats: &mut ComputeStats,
) -> crate::ANNResult<()> {
    use rayon::prelude::*;

    // Copy references only; point coordinates remain borrowed from the provider.
    let mut rows = Vec::with_capacity(count);
    rows.extend(points);
    labels.resize(rows.len(), 0);
    let tiles = rows.len().div_ceil(GEMM_ROWS);
    let jobs = tiles.min(rayon::current_num_threads());
    if jobs <= 1 {
        stats.merge(assign_job(centers, &rows, labels)?);
        return Ok(());
    }
    let tiles_per_job = tiles / jobs;
    let extra_tiles = tiles % jobs;
    let mut remaining_rows = rows.as_slice();
    let mut remaining_labels = labels.as_mut_slice();
    let mut tasks = Vec::with_capacity(jobs);
    for job in 0..jobs {
        // Boundaries remain global multiples of 128, independent of pool size.
        let count = (tiles_per_job + usize::from(job < extra_tiles))
            .saturating_mul(GEMM_ROWS)
            .min(remaining_rows.len());
        let (points, next_rows) = remaining_rows.split_at(count);
        let (output, next_labels) = remaining_labels.split_at_mut(count);
        tasks.push((points, output));
        remaining_rows = next_rows;
        remaining_labels = next_labels;
    }
    let results: Vec<crate::ANNResult<ComputeStats>> = tasks
        .into_par_iter()
        .map(|(rows, labels)| {
            let mut result = assign_job(centers, rows, labels)?;
            result.gemm_parallel_jobs = 1;
            Ok(result)
        })
        .collect();
    for result in results {
        stats.merge(result?);
    }
    Ok(())
}

#[cfg(feature = "ivf-parallel")]
fn assign_job(
    centers: &PreparedCenters<'_>,
    rows: &[&[f32]],
    labels: &mut [usize],
) -> crate::ANNResult<ComputeStats> {
    let mut scratch = GemmScratch::new(centers.dim, centers.norms.len())?;
    let mut stats = ComputeStats::default();
    for (points, output) in rows.chunks(GEMM_ROWS).zip(labels.chunks_mut(GEMM_ROWS)) {
        stats.merge(assign_tile(centers, points, output, &mut scratch)?);
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_layout_translation_tail_and_fallback() -> crate::ANNResult<()> {
        // D=3, K=8, 131 rows exercise row-major layout and a short final tile.
        let centers: Vec<f32> = (0..8)
            .flat_map(|j| [10_000.0 + 4.0 * j as f32, 2.0, -3.0])
            .collect();
        let points: Vec<Vec<f32>> = (0..131)
            .map(|i| vec![10_000.0 + 4.0 * (i % 8) as f32 + 0.25, 2.0, -3.0])
            .collect();
        let mut labels = Vec::with_capacity(points.len());
        let mut stats = ComputeStats::default();
        assign_gemm(
            points.iter().map(Vec::as_slice),
            &centers,
            3,
            points.len(),
            &mut labels,
            &mut stats,
        )?;
        assert_eq!(labels, (0..131).map(|i| i % 8).collect::<Vec<_>>());
        assert_eq!(stats.gemm_calls, 2);
        assert_eq!(stats.gemm_points, 131);
        assert_eq!(stats.numeric_fallbacks, 0);

        let mut stats = ComputeStats::default();
        assign_gemm(
            [&[2.0][..]].into_iter(),
            &[1.0; 8],
            1,
            1,
            &mut labels,
            &mut stats,
        )?;
        assert_eq!(labels, [0]); // Reusing the label buffer must discard prior rows.

        let mut stats = ComputeStats::default();
        assign_gemm(
            [&[0.0][..]].into_iter(),
            &[-f32::MAX, f32::MAX],
            1,
            1,
            &mut labels,
            &mut stats,
        )?;
        assert_eq!(labels, [0]); // Both direct L2 distances overflow; first tie wins.
        assert_eq!(stats.numeric_fallbacks, 1);
        assert_eq!(stats.gemm_calls, 0);

        // A translated point overflows while centers and another row remain finite.
        let mut stats = ComputeStats::default();
        assign_gemm(
            [&[-f32::MAX][..], &[f32::MAX][..]].into_iter(),
            &[f32::MAX, f32::MAX / 2.0],
            1,
            2,
            &mut labels,
            &mut stats,
        )?;
        assert_eq!(labels, [0, 0]);
        assert_eq!(stats.gemm_calls, 1);
        assert_eq!(stats.gemm_points, 1);
        assert_eq!(stats.numeric_fallbacks, 1);

        // Finite packing can still overflow inside the f32 matrix product.
        let mut stats = ComputeStats::default();
        assign_gemm(
            [&[1.0e20][..]].into_iter(),
            &[0.0, 1.0e20],
            1,
            1,
            &mut labels,
            &mut stats,
        )?;
        assert_eq!(labels, [1]);
        assert_eq!(stats.gemm_calls, 1);
        assert_eq!(stats.gemm_points, 0);
        assert_eq!(stats.numeric_fallbacks, 1);

        #[cfg(feature = "ivf-parallel")]
        {
            // Five global tiles and D=33 cross the work threshold, with a short tail.
            let dim = 33;
            let count = 4 * GEMM_ROWS + 3;
            let centers: Vec<f32> = (0..16)
                .flat_map(|center| std::iter::repeat_n(4.0 * center as f32, dim))
                .collect();
            let mut points: Vec<Vec<f32>> = (0..count)
                .map(|point| vec![4.0 * (point % 16) as f32 + 0.25; dim])
                .collect();
            let mut expected: Vec<_> = (0..count).map(|point| point % 16).collect();
            for point in [2 * GEMM_ROWS + 1, count - 1] {
                points[point].fill(f32::MAX);
                expected[point] = 0; // GEMM overflows; direct L2 preserves its first tie.
            }
            let run = |threads| -> crate::ANNResult<(Vec<usize>, ComputeStats)> {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(crate::ANNError::new)?;
                pool.install(|| {
                    let mut labels = Vec::with_capacity(count);
                    let mut stats = ComputeStats::default();
                    assign_gemm(
                        points.iter().map(Vec::as_slice),
                        &centers,
                        dim,
                        count,
                        &mut labels,
                        &mut stats,
                    )?;
                    Ok((labels, stats))
                })
            };
            let (serial, serial_stats) = run(1)?;
            let (parallel, parallel_stats) = run(4)?;
            assert_eq!(serial, expected);
            assert_eq!(parallel, serial);
            for stats in [serial_stats, parallel_stats] {
                assert_eq!(stats.gemm_calls, count.div_ceil(GEMM_ROWS));
                assert_eq!(stats.gemm_points, count - 2);
                assert_eq!(stats.numeric_fallbacks, 2);
                assert_eq!(stats.projection_points, 0);
            }
            assert_eq!(serial_stats.gemm_parallel_jobs, 0);
            assert_eq!(parallel_stats.gemm_parallel_jobs, 4);
        }
        Ok(())
    }
}
