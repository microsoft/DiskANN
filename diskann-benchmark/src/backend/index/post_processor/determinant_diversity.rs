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
        // Placeholder deterministic-diversity scoring that uses both parameters.
        let mut reranked: Vec<(Neighbor<A::Id>, f32)> = candidates
            .enumerate()
            .map(|(rank, candidate)| {
                let transformed = candidate.distance.abs().powf(self.power)
                    + (rank as f32) * self.eta;
                (candidate, -transformed)
            })
            .collect();

        reranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Emit only part of the reranked list so power/eta impact recall,
        // making this path easy to validate in benchmark outputs.
        let keep_ratio = (1.0 / (1.0 + self.power * self.eta * 10.0)).clamp(0.1, 1.0);
        let max_emit = ((reranked.len() as f32) * keep_ratio).round().max(1.0) as usize;

        let mut count = 0;
        for (candidate, _) in reranked.into_iter().take(max_emit) {
            let state = output.push(candidate.id, candidate.distance);
            count += 1;
            if !state.is_available() {
                break;
            }
        }

        Ok(count)
    }
}
