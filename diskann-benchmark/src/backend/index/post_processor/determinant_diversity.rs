/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::search_output_buffer::SearchOutputBuffer;
use diskann::{
    error::ANNError,
    graph::glue,
    neighbor::Neighbor,
    provider::Accessor,
};
use diskann_providers::model::graph::provider::async_::determinant_diversity_post_process;

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
        let candidates: Vec<Neighbor<A::Id>> = candidates.collect();
        let embedded: Vec<_> = candidates
            .iter()
            .map(|c| (c.id, c.distance, vec![c.distance]))
            .collect();

        let reranked = determinant_diversity_post_process(
            embedded,
            &[0.0],
            candidates.len(),
            self.eta,
            self.power,
        );

        Ok(output.extend(reranked))
    }
}
