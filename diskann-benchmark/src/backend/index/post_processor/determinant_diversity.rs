/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::future::Future;

use diskann::graph::search_output_buffer::SearchOutputBuffer;
use diskann::{
    error::ANNError,
    graph::glue,
    neighbor::Neighbor,
    provider::Accessor,
};
use diskann_providers::model::graph::provider::async_::{
    determinant_diversity_post_process,
    inmem,
};
use diskann_utils::future::AsyncFriendly;

pub(crate) trait FullPrecisionVectorAccessor: Accessor + Send {
    fn get_full_precision_vector(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Vec<f32>, ANNError>> + Send;
}

impl<Q, D, Ctx> FullPrecisionVectorAccessor for inmem::FullAccessor<'_, f32, Q, D, Ctx>
where
    Q: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: diskann::provider::ExecutionContext,
{
    fn get_full_precision_vector(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Vec<f32>, ANNError>> + Send {
        async move {
            self.get_element(id)
                .await
                .map(|vector| vector.to_vec())
                .map_err(Into::into)
        }
    }
}

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
    A: FullPrecisionVectorAccessor + diskann::provider::BuildQueryComputer<T> + Send,
    T: AsRef<[f32]> + Send + Sync,
{
    type Error = ANNError;

    async fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: T,
        _computer: &<A as diskann::provider::BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<A::Id> + Send + ?Sized,
    {
        let candidates: Vec<Neighbor<A::Id>> = candidates.collect();
        let mut embedded = Vec::with_capacity(candidates.len());

        for candidate in &candidates {
            embedded.push((
                candidate.id,
                candidate.distance,
                accessor.get_full_precision_vector(candidate.id).await?,
            ));
        }

        let reranked = determinant_diversity_post_process(
            embedded,
            query.as_ref(),
            candidates.len(),
            self.eta,
            self.power,
        );

        Ok(output.extend(reranked))
    }
}
