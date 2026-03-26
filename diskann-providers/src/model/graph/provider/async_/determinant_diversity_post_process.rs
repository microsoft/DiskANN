/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Determinant-diversity search post-processing.

use std::future::Future;

use diskann::{
    ANNError,
    graph::{SearchOutputBuffer, glue},
    neighbor::Neighbor,
    provider::BuildQueryComputer,
    utils::{IntoUsize, VectorRepr},
};
use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};

use super::{
    inmem::GetFullPrecision,
    postprocess::{AsDeletionCheck, DeletionCheck},
};

#[derive(Debug)]
pub enum DeterminantDiversityError {
    InvalidTopK { top_k: usize },
    InvalidEta { eta: f64 },
    InvalidPower { power: f64 },
}

impl std::fmt::Display for DeterminantDiversityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTopK { top_k } => write!(f, "top_k must be > 0, got {top_k}"),
            Self::InvalidEta { eta } => write!(f, "eta must be >= 0.0, got {eta}"),
            Self::InvalidPower { power } => write!(f, "power must be > 0.0, got {power}"),
        }
    }
}

impl std::error::Error for DeterminantDiversityError {}

#[derive(Debug, Clone, Copy)]
pub struct DeterminantDiversitySearchParams {
    pub top_k: usize,
    pub determinant_diversity_eta: f64,
    pub determinant_diversity_power: f64,
}

impl DeterminantDiversitySearchParams {
    pub fn new(
        top_k: usize,
        determinant_diversity_eta: f64,
        determinant_diversity_power: f64,
    ) -> Result<Self, DeterminantDiversityError> {
        if top_k == 0 {
            return Err(DeterminantDiversityError::InvalidTopK { top_k });
        }

        if determinant_diversity_eta < 0.0 || !determinant_diversity_eta.is_finite() {
            return Err(DeterminantDiversityError::InvalidEta {
                eta: determinant_diversity_eta,
            });
        }

        if determinant_diversity_power <= 0.0 || !determinant_diversity_power.is_finite() {
            return Err(DeterminantDiversityError::InvalidPower {
                power: determinant_diversity_power,
            });
        }

        Ok(Self {
            top_k,
            determinant_diversity_eta,
            determinant_diversity_power,
        })
    }
}

impl<A, T> glue::SearchPostProcess<A, [T]> for DeterminantDiversitySearchParams
where
    T: VectorRepr,
    A: BuildQueryComputer<[T], Id = u32> + GetFullPrecision<Repr = T> + AsDeletionCheck,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &[T],
        _computer: &A::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let result = (|| {
            let query_f32 = T::as_f32(query).map_err(Into::into)?;
            let full = accessor.as_full_precision();
            let checker = accessor.as_deletion_check();

            let mut candidates_with_vectors = Vec::new();
            for candidate in candidates {
                if checker.deletion_check(candidate.id) {
                    continue;
                }

                let vector = unsafe { full.get_vector_sync(candidate.id.into_usize()) };
                let vector_f32 = T::as_f32(vector).map_err(Into::into)?;
                candidates_with_vectors.push((
                    candidate.id,
                    candidate.distance,
                    vector_f32.to_vec(),
                ));
            }

            let reranked = determinant_diversity_post_process(
                candidates_with_vectors,
                &query_f32[..],
                self.top_k,
                self.determinant_diversity_eta,
                self.determinant_diversity_power,
            );

            Ok(output.extend(reranked))
        })();

        std::future::ready(result)
    }
}

pub fn determinant_diversity_post_process<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_eta: f64,
    determinant_diversity_power: f64,
) -> Vec<(Id, f32)> {
    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let k = k.min(candidates.len());
    if k == 0 {
        return Vec::new();
    }

    if candidates[0].2.is_empty() {
        return Vec::new();
    }

    if determinant_diversity_eta > 0.0 {
        post_process_with_eta_f32(
            candidates,
            query,
            k,
            determinant_diversity_eta,
            determinant_diversity_power,
        )
    } else {
        post_process_greedy_orthogonalization_f32(candidates, query, k, determinant_diversity_power)
    }
}

fn post_process_with_eta_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_eta: f64,
    determinant_diversity_power: f64,
) -> Vec<(Id, f32)> {
    let eta = determinant_diversity_eta as f32;
    let power = determinant_diversity_power;

    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    if candidates[0].2.is_empty() {
        return Vec::new();
    }

    let inv_sqrt_eta = 1.0 / eta.sqrt();
    let mut residuals = Vec::with_capacity(n);
    let mut norms_sq = Vec::with_capacity(n);

    for (_, _, v) in &candidates {
        let similarity = dot_product(v, query);
        let scale = similarity.max(0.0).powf(power as f32) * inv_sqrt_eta;
        let residual: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        let norm_sq = dot_product(&residual, &residual);
        residuals.push(residual);
        norms_sq.push(norm_sq);
    }

    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);

    for _ in 0..k {
        let best_idx = available
            .iter()
            .enumerate()
            .filter(|&(_, &avail)| avail)
            .max_by(|(i, _), (j, _)| {
                norms_sq[*i]
                    .partial_cmp(&norms_sq[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let Some(selected_index) = best_idx else {
            break;
        };

        selected.push(selected_index);
        available[selected_index] = false;

        if selected.len() == k {
            break;
        }

        let best_norm_sq = norms_sq[selected_index];
        if best_norm_sq <= 0.0 {
            continue;
        }

        let inv_norm_sq = 1.0 / best_norm_sq;
        let r_star_copy = residuals[selected_index].clone();

        let mut projections = Vec::with_capacity(n);
        for i in 0..n {
            if !available[i] {
                projections.push(0.0);
            } else {
                let projection = dot_product(&residuals[i], &r_star_copy) * inv_norm_sq;
                projections.push(projection);
            }
        }

        for i in 0..n {
            if !available[i] {
                continue;
            }

            let projection = projections[i];
            for (residual, &star) in residuals[i].iter_mut().zip(r_star_copy.iter()) {
                *residual -= projection * star;
            }

            norms_sq[i] = (norms_sq[i] - projection * projection * best_norm_sq).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = &candidates[idx];
            (*id, *dist)
        })
        .collect()
}

fn post_process_greedy_orthogonalization_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_power: f64,
) -> Vec<(Id, f32)> {
    let power = determinant_diversity_power;

    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let n = candidates.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let mut residuals = Vec::with_capacity(n);
    let mut norms_sq = Vec::with_capacity(n);

    for (_, _, v) in &candidates {
        let similarity = dot_product(v, query);
        let scale = similarity.max(0.0).powf(power as f32);
        let residual: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        let norm_sq = dot_product(&residual, &residual);
        residuals.push(residual);
        norms_sq.push(norm_sq);
    }

    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);

    for _ in 0..k {
        let best = available
            .iter()
            .enumerate()
            .filter(|&(_, &avail)| avail)
            .max_by(|(i, _), (j, _)| {
                norms_sq[*i]
                    .partial_cmp(&norms_sq[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let Some((best_index, _)) = best else {
            break;
        };

        let best_norm_sq = norms_sq[best_index];
        selected.push(best_index);
        available[best_index] = false;

        if selected.len() == k {
            break;
        }

        if best_norm_sq <= 0.0 {
            continue;
        }

        let inv_norm_sq_star = 1.0 / best_norm_sq;
        let r_star_copy = residuals[best_index].clone();

        let mut projections = Vec::with_capacity(n);
        for j in 0..n {
            if !available[j] {
                projections.push(0.0);
            } else {
                let projection = dot_product(&residuals[j], &r_star_copy) * inv_norm_sq_star;
                projections.push(projection);
            }
        }

        for j in 0..n {
            if !available[j] {
                continue;
            }

            let projection = projections[j];
            for (residual, &star) in residuals[j].iter_mut().zip(r_star_copy.iter()) {
                *residual -= projection * star;
            }

            norms_sq[j] = (norms_sq[j] - projection * projection * best_norm_sq).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = &candidates[idx];
            (*id, *dist)
        })
        .collect()
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    <InnerProduct as PureDistanceFunction<&[f32], &[f32], MathematicalValue<f32>>>::evaluate(a, b)
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_valid_params() {
        let result = DeterminantDiversitySearchParams::new(10, 0.01, 2.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_invalid_params() {
        let test_cases = [
            (
                DeterminantDiversitySearchParams::new(0, 0.01, 2.0),
                DeterminantDiversityError::InvalidTopK { top_k: 0 },
            ),
            (
                DeterminantDiversitySearchParams::new(10, -0.01, 2.0),
                DeterminantDiversityError::InvalidEta { eta: -0.01 },
            ),
            (
                DeterminantDiversitySearchParams::new(10, f64::NAN, 2.0),
                DeterminantDiversityError::InvalidEta { eta: f64::NAN },
            ),
            (
                DeterminantDiversitySearchParams::new(10, 0.01, 0.0),
                DeterminantDiversityError::InvalidPower { power: 0.0 },
            ),
            (
                DeterminantDiversitySearchParams::new(10, 0.01, -1.0),
                DeterminantDiversityError::InvalidPower { power: -1.0 },
            ),
            (
                DeterminantDiversitySearchParams::new(10, 0.01, f64::INFINITY),
                DeterminantDiversityError::InvalidPower {
                    power: f64::INFINITY,
                },
            ),
        ];

        for (result, expected) in test_cases {
            match (result, expected) {
                (
                    Err(DeterminantDiversityError::InvalidTopK { top_k: actual }),
                    DeterminantDiversityError::InvalidTopK { top_k: expected },
                ) => assert_eq!(actual, expected),
                (
                    Err(DeterminantDiversityError::InvalidEta { eta: actual }),
                    DeterminantDiversityError::InvalidEta { eta: expected },
                ) => {
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(actual, expected);
                    }
                }
                (
                    Err(DeterminantDiversityError::InvalidPower { power: actual }),
                    DeterminantDiversityError::InvalidPower { power: expected },
                ) => {
                    if expected.is_infinite() {
                        assert!(actual.is_infinite());
                    } else {
                        assert_eq!(actual, expected);
                    }
                }
                (other, expected) => {
                    panic!("Unexpected result {:?} for expected {:?}", other, expected)
                }
            }
        }
    }

    #[test]
    fn test_determinant_diversity_post_process_with_eta() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        let v3 = vec![0.0f32, 0.0, 1.0];
        let candidates = vec![(1u32, 0.5f32, v1), (2u32, 0.3f32, v2), (3u32, 0.7f32, v3)];
        let query = vec![1.0, 1.0, 1.0];

        let result = determinant_diversity_post_process(candidates, &query, 3, 0.01, 2.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_determinant_diversity_post_process_enabled_greedy() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.99f32, 0.1, 0.0];
        let v3 = vec![0.0f32, 1.0, 0.0];
        let candidates = vec![(1u32, 0.5f32, v1), (2u32, 0.3f32, v2), (3u32, 0.4f32, v3)];
        let query = vec![1.0, 1.0, 0.0];

        let result = determinant_diversity_post_process(candidates, &query, 2, 0.0, 1.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_determinant_diversity_post_process_empty() {
        let candidates: Vec<(u32, f32, Vec<f32>)> = vec![];
        let query = vec![1.0, 1.0, 1.0];

        let result = determinant_diversity_post_process(candidates, &query, 3, 0.01, 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_determinant_diversity_post_process_k_larger_than_candidates() {
        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        let candidates = vec![(1u32, 0.5f32, v1), (2u32, 0.3f32, v2)];
        let query = vec![1.0, 1.0, 1.0];

        let result = determinant_diversity_post_process(candidates, &query, 10, 0.01, 2.0);
        assert_eq!(result.len(), 2);
    }
}
