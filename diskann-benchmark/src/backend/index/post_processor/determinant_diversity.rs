/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::future::Future;

use diskann::graph::search_output_buffer::SearchOutputBuffer;
use diskann::utils::IntoUsize;
use diskann::{error::ANNError, graph::glue, neighbor::Neighbor, provider::HasId};
use diskann_providers::model::graph::provider::{
    async_::inmem::{self, GetFullPrecision},
    determinant_diversity_post_process,
};
use diskann_quantization::num::Positive;

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

impl<'a, A> glue::SearchPostProcess<A, &'a [f32], A::Id> for DeterminantDiversity
where
    A: HasId<Id = u32> + GetFullPrecision<Repr = f32> + Send + Sync,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: &'a [f32],
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<A::Id> + Send + ?Sized,
    {
        let candidates: Vec<Neighbor<A::Id>> = candidates.collect();
        let store: &inmem::FullPrecisionStore<f32> = accessor.as_full_precision();
        let mut embedded = Vec::with_capacity(candidates.len());

        for candidate in &candidates {
            // SAFETY: We accept potential unsynchronized concurrent mutation, matching the
            // pattern used by `Rerank` in `inmem::full_precision`.
            let vector = unsafe { store.get_vector_sync(candidate.id.into_usize()) };
            embedded.push((candidate.id, candidate.distance, vector.to_vec()));
        }

        let reranked = determinant_diversity_post_process(
            embedded,
            query,
            candidates.len(),
            self.eta,
            Positive::new(self.power).expect("power must be > 0"),
        );

        std::future::ready(Ok(output.extend(reranked)))
    }
}
