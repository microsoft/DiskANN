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
    determinant_diversity, DeterminantDiversityParams,
};
use diskann_utils::views::Matrix;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DeterminantDiversity {
    params: DeterminantDiversityParams,
}

impl DeterminantDiversity {
    pub(crate) const fn new(params: DeterminantDiversityParams) -> Self {
        Self { params }
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
        let mut vectors = Matrix::new(0.0f32, candidates.len(), query.len());
        let mut ids = Vec::with_capacity(candidates.len());
        let mut distances = Vec::with_capacity(candidates.len());

        for (i, candidate) in candidates.iter().enumerate() {
            // SAFETY: We accept potential unsynchronized concurrent mutation, matching the
            // pattern used by `Rerank` in `inmem::full_precision`.
            let vector = unsafe { store.get_vector_sync(candidate.id.into_usize()) };
            ids.push(candidate.id);
            distances.push(candidate.distance);
            vectors.row_mut(i).copy_from_slice(vector);
        }

        let reranked = determinant_diversity(
            vectors.as_mut_view(),
            &distances,
            query,
            candidates.len(),
            &self.params,
        )
        .into_iter()
        .map(|idx| (ids[idx], distances[idx]));

        std::future::ready(Ok(output.extend(reranked)))
    }
}
