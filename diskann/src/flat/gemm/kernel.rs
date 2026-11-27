/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, ops::Range, sync::OnceLock};

use diskann_linalg::{Transpose, sgemm};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{
    Norm, PreprocessedDistanceFunction, distance::Metric, norm::FastL2NormSquared,
};
use rayon::ThreadPool;

use super::{
    BatchOptions, BatchStats, CandidateFilter, Dataset, Engine, GemmElement, Partial, check_len,
    partition_start, scratch_len,
};
use crate::{ANNError, ANNResult, flat::topk::BatchCollector, utils::VectorId};

#[expect(
    clippy::too_many_arguments,
    reason = "shared synchronous and streaming execution core"
)]
#[track_caller]
pub(super) fn search<T, I, F>(
    data: Dataset<'_, T, I>,
    queries: rowmajor::Ref<'_, T>,
    k: NonZeroUsize,
    metric: Metric,
    filter: &F,
    options: &BatchOptions,
    pool: &ThreadPool,
    ordinal_base: usize,
) -> ANNResult<Vec<Partial<I>>>
where
    T: GemmElement,
    I: VectorId,
    F: CandidateFilter<I> + ?Sized,
{
    options.validate()?;
    if data.data.ncols() != queries.ncols() {
        return Err(ANNError::message("database and query dimensions differ"));
    }
    check_len::<Partial<I>>(queries.nrows())?;
    ordinal_base
        .checked_add(data.len())
        .ok_or_else(|| ANNError::message("flat scan ordinal overflow"))?;
    if queries.nrows() == 0 {
        return Ok(Vec::new());
    }
    let Some(rows) = NonZeroUsize::new(data.len()) else {
        return Ok((0..queries.nrows()).map(|_| Partial::empty()).collect());
    };
    let k = k.min(rows);
    let workers = options.gemm.worker_count(pool, rows);
    scratch_len(
        options.gemm.query_block_size.get().min(queries.nrows()),
        options
            .gemm
            .database_block_size
            .get()
            .min(data.len().div_ceil(workers)),
    )?;
    let effective = T::effective_metric(metric);
    if options.engine != Engine::Direct && matches!(effective, Metric::L2 | Metric::Cosine) {
        check_len::<f32>(data.len())?;
    }
    // Initialize only if a tile actually selects GEMM. Sparse filtered/direct
    // searches must not read the entire database just to prepare unused norms.
    let norms = OnceLock::new();
    let context = Context {
        data,
        queries,
        k,
        metric,
        effective,
        filter,
        options,
        norms: &norms,
        ordinal_base,
    };
    let mut output = Vec::with_capacity(queries.nrows());
    for q0 in (0..queries.nrows()).step_by(options.gemm.query_block_size.get()) {
        let nq = options
            .gemm
            .query_block_size
            .get()
            .min(queries.nrows() - q0);
        if workers == 1 {
            output.extend(pool.install(|| context.scan(q0, nq, 0..rows.get()))?);
        } else {
            let mut partials: Vec<ANNResult<Vec<Partial<I>>>> =
                (0..workers).map(|_| Ok(Vec::new())).collect();
            pool.scope(|scope| {
                for (worker, result) in partials.iter_mut().enumerate() {
                    let range = partition_start(rows.get(), workers, worker)
                        ..partition_start(rows.get(), workers, worker + 1);
                    let context = &context;
                    scope.spawn(move |_| {
                        *result = context.scan(q0, nq, range);
                    });
                }
            });
            let mut merged: Vec<_> = (0..nq).map(|_| Partial::empty()).collect();
            for result in partials {
                for (dst, src) in merged.iter_mut().zip(result?) {
                    dst.merge(src, k.get())?;
                }
            }
            output.extend(merged);
        }
    }
    Ok(output)
}

fn prepare_norms<T: GemmElement, I: Copy>(data: &Dataset<'_, T, I>, block: usize) -> Vec<f32> {
    if let Some(norms) = data.norms {
        return (0..data.len()).map(|i| norms[data.row_index(i)]).collect();
    }
    let dim = data.data.ncols();
    let mut norms = Vec::with_capacity(data.len());
    let mut selected = Vec::new();
    let mut packed = Vec::new();
    for start in (0..data.len()).step_by(block) {
        let len = block.min(data.len() - start);
        let values = T::pack(data.tile(start, len, &mut selected), &mut packed);
        for row in 0..len {
            let v = &values[row * dim..(row + 1) * dim];
            norms.push(FastL2NormSquared.evaluate(v));
        }
    }
    norms
}

struct Context<'a, 'b, T, I, F: ?Sized> {
    data: Dataset<'a, T, I>,
    queries: rowmajor::Ref<'a, T>,
    k: NonZeroUsize,
    metric: Metric,
    effective: Metric,
    filter: &'b F,
    options: &'b BatchOptions,
    norms: &'b OnceLock<Vec<f32>>,
    ordinal_base: usize,
}

impl<T, I, F> Context<'_, '_, T, I, F>
where
    T: GemmElement,
    I: VectorId,
    F: CandidateFilter<I> + ?Sized,
{
    fn scan(&self, q0: usize, nq: usize, range: Range<usize>) -> ANNResult<Vec<Partial<I>>> {
        let dim = self.data.data.ncols();
        let block = self.options.gemm.database_block_size.get().min(range.len());
        let local_k = self.k.get().min(range.len());
        let k = NonZeroUsize::new(local_k)
            .ok_or_else(|| ANNError::message("empty database partition"))?;
        let mut collectors: Vec<_> = (0..nq)
            .map(|_| BatchCollector::new(k, self.options.gemm.collection.reservoir_min_k))
            .collect();
        let mut stats = vec![BatchStats::default(); nq];
        let original_queries = &self.queries.as_slice()[q0 * dim..(q0 + nq) * dim];
        let computers: Vec<_> = (0..nq)
            .map(|q| T::query_distance(&original_queries[q * dim..(q + 1) * dim], self.metric))
            .collect();
        let mut query_scratch = Vec::new();
        let can_use_gemm = self.options.use_gemm::<T>(nq, block, dim, nq * block);
        let queries = if can_use_gemm {
            T::pack(original_queries, &mut query_scratch)
        } else {
            &[]
        };
        let mut query_norms = Vec::new();
        let mut scores = vec![0.0; scratch_len(nq, block)?];
        let mut mask = if F::ALL {
            Vec::new()
        } else {
            vec![false; scores.len()]
        };
        let mut eligible = vec![0usize; nq];
        let mut selected = Vec::new();
        let mut packed = Vec::new();
        let mut ids = Vec::with_capacity(block);
        let mut kept_ids = Vec::with_capacity(block);
        let mut kept_scores = Vec::with_capacity(block);
        for start in range.clone().step_by(block) {
            let len = block.min(range.end - start);
            ids.clear();
            ids.extend((start..start + len).map(|i| (self.ordinal_base + i, self.data.id(i))));
            for (q, count) in eligible.iter_mut().enumerate() {
                *count = if F::ALL {
                    len
                } else {
                    let line = &mut mask[q * len..(q + 1) * len];
                    for (allowed, &(_, id)) in line.iter_mut().zip(&ids) {
                        *allowed = self.filter.is_member(q0 + q, id);
                    }
                    line.iter().filter(|&&b| b).count()
                };
                stats[q].eligible_candidates += *count as u64;
            }
            let count = eligible.iter().sum();
            if count == 0 {
                continue;
            }
            let use_gemm = self.options.use_gemm::<T>(nq, len, dim, count);
            let norms = if use_gemm && matches!(self.effective, Metric::L2 | Metric::Cosine) {
                if query_norms.is_empty() {
                    query_norms.extend((0..nq).map(|q| {
                        let row = &queries[q * dim..(q + 1) * dim];
                        FastL2NormSquared.evaluate(row)
                    }));
                }
                Some(self.norms.get_or_init(|| {
                    prepare_norms(&self.data, self.options.gemm.database_block_size.get())
                }))
            } else {
                None
            };
            let scores = &mut scores[..nq * len];
            if use_gemm {
                let values = T::pack(self.data.tile(start, len, &mut selected), &mut packed);
                if dim == 0 {
                    scores.fill(0.0);
                } else {
                    sgemm(
                        Transpose::None,
                        Transpose::Ordinary,
                        nq,
                        len,
                        dim,
                        1.0,
                        queries,
                        values,
                        None,
                        scores,
                    )
                    .map_err(|error| ANNError::new(error).context("flat distance GEMM"))?;
                }
            }
            for q in 0..nq {
                if use_gemm {
                    stats[q].cmps += len as u64;
                    stats[q].gemm_tiles += 1;
                } else if eligible[q] != 0 {
                    stats[q].direct_tiles += 1;
                }
                kept_ids.clear();
                kept_scores.clear();
                for j in 0..len {
                    if !F::ALL && !mask[q * len + j] {
                        continue;
                    }
                    let score = if use_gemm {
                        let qnorm = query_norms.get(q).copied().unwrap_or(0.0);
                        let xnorm = norms.map_or(0.0, |norms| norms[start + j]);
                        transform(self.effective, scores[q * len + j], qnorm, xnorm, dim)
                    } else {
                        None
                    };
                    let score = match score {
                        Some(value) => value,
                        None => {
                            stats[q].cmps += 1;
                            if use_gemm {
                                stats[q].recomputations += 1;
                            }
                            computers[q].evaluate_similarity(
                                self.data.data.row(self.data.row_index(start + j)),
                            )
                        }
                    };
                    if F::ALL {
                        scores[q * len + j] = score;
                    } else {
                        kept_ids.push(ids[j]);
                        kept_scores.push(score);
                    }
                }
                if F::ALL {
                    collectors[q].add_batch(&ids, &scores[q * len..(q + 1) * len]);
                } else {
                    collectors[q].add_batch(&kept_ids, &kept_scores);
                }
            }
        }
        Ok(collectors
            .into_iter()
            .zip(stats)
            .map(|(collector, stats)| Partial {
                neighbors: collector.finish(),
                stats,
            })
            .collect())
    }
}

// None requests the original distance kernel. In particular, do not turn a NaN
// into a zero distance by clamping, or conceal cancellation in norm-expanded L2.
fn transform(metric: Metric, dot: f32, qnorm: f32, xnorm: f32, dim: usize) -> Option<f32> {
    if !dot.is_finite() {
        return None;
    }
    match metric {
        Metric::InnerProduct => Some(-dot),
        Metric::CosineNormalized => Some(1.0 - dot),
        Metric::L2 | Metric::Cosine => {
            if !qnorm.is_finite() || !xnorm.is_finite() || qnorm < 0.0 || xnorm < 0.0 {
                return None;
            }
            if metric == Metric::Cosine {
                // Match FullCosineAccumulator's treatment of subnormal squared norms.
                if qnorm < f32::MIN_POSITIVE || xnorm < f32::MIN_POSITIVE {
                    return Some(1.0);
                }
                let value = dot / (qnorm.sqrt() * xnorm.sqrt());
                value.is_finite().then(|| 1.0 - value.clamp(-1.0, 1.0))
            } else {
                let value = qnorm + xnorm - 2.0 * dot;
                let roundoff = 8.0 * f32::EPSILON * dim as f32 * (qnorm + xnorm + 2.0 * dot.abs());
                (value.is_finite() && value > roundoff).then_some(value)
            }
        }
    }
}
