/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    error::ANNError,
    graph::glue,
    neighbor::Neighbor,
    provider::Accessor,
};
use diskann::graph::search_output_buffer::SearchOutputBuffer;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeterminantDiversity {
    power: f32,
    eta: f32,
}

impl DeterminantDiversity {
    pub(crate) const fn new(power: f32, eta: f32) -> Self {
        Self { power, eta }
    }
}

pub(crate) fn rank_and_limit_by_distance(
    distances: &[f32],
    power: f32,
    eta: f32,
) -> (Vec<usize>, usize) {
    let mut ranked: Vec<(usize, f32)> = distances
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, distance)| {
            let transformed = distance.abs().powf(power) + (rank as f32) * eta;
            (rank, -transformed)
        })
        .collect();

    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let ranked_indices: Vec<usize> = ranked.into_iter().map(|(rank, _)| rank).collect();
    if ranked_indices.is_empty() {
        return (ranked_indices, 0);
    }

    let keep_ratio = (1.0 / (1.0 + power * eta * 10.0)).clamp(0.1, 1.0);
    let max_emit = ((ranked_indices.len() as f32) * keep_ratio)
        .round()
        .max(1.0) as usize;

    (ranked_indices, max_emit)
}

impl<A, T> glue::SearchPostProcess<A, T, A::Id> for DeterminantDiversity
where
    A: Accessor + diskann::provider::BuildQueryComputer<T> + Send,
    T: Send + Sync,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: T,
        _computer: &<A as diskann::provider::BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<A::Id> + Send + ?Sized,
    {
        let candidates: Vec<Neighbor<A::Id>> = candidates.collect();
        let distances: Vec<f32> = candidates.iter().map(|c| c.distance).collect();
        let (ranked_indices, max_emit) =
            rank_and_limit_by_distance(&distances, self.power, self.eta);

        let mut count = 0;
        for rank in ranked_indices.into_iter().take(max_emit) {
            let candidate = &candidates[rank];
            let state = output.push(candidate.id, candidate.distance);
            count += 1;
            if !state.is_available() {
                break;
            }
        }

        Ok(count)
    }
}
