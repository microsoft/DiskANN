/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, num::NonZeroUsize, ops::Range, sync::Arc};

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix},
};
use diskann_vector::distance::Metric;
use rayon::ThreadPool;

use super::{
    BatchOptions, BatchStats, CandidateFilter, Dataset, GemmElement, Partial, Selection, check_len,
    kernel,
};
use crate::{
    ANNError, ANNResult,
    error::{ErrorExt, IntoANNResult, ToRanked},
    graph::{SearchOutputBuffer, glue::SearchPostProcess},
    provider::HasId,
    utils::VectorId,
};

/// A block retains its storage while a worker is running, even if its search
/// future is dropped. A subrange can share a larger immutable in-memory database.
#[derive(Debug, Clone)]
pub struct OwnedBlock<T, I> {
    data: Arc<rowmajor::Owned<T>>,
    ids: Arc<[I]>,
    rows: Range<usize>,
    norms: Option<Arc<[f32]>>,
}

impl<T, I: Copy> OwnedBlock<T, I> {
    /// # Errors
    /// Returns an error unless there is one ID per row.
    #[track_caller]
    pub fn new(data: Arc<rowmajor::Owned<T>>, ids: Arc<[I]>) -> ANNResult<Self> {
        if data.nrows() != ids.len() {
            return Err(ANNError::message("database row and ID counts differ"));
        }
        let rows = 0..data.nrows();
        Ok(Self {
            data,
            ids,
            rows,
            norms: None,
        })
    }

    /// Attach squared norms for the same immutable storage, in original row order.
    ///
    /// # Errors
    /// Returns an error unless there is one norm per stored row.
    #[track_caller]
    pub fn with_squared_norms(mut self, norms: Arc<[f32]>) -> ANNResult<Self> {
        if norms.len() != self.data.nrows() {
            return Err(ANNError::message("database row and norm counts differ"));
        }
        self.norms = Some(norms);
        Ok(self)
    }

    pub fn nrows(&self) -> usize {
        self.rows.len()
    }
    pub fn dimensions(&self) -> usize {
        self.data.ncols()
    }

    /// Access a block-relative row without advancing the source.
    pub fn get_row(&self, row: usize) -> Option<(I, &[T])> {
        if row >= self.nrows() {
            return None;
        }
        let row = self.rows.start + row;
        Some((self.ids[row], self.data.row(row)))
    }

    /// Borrow the block for synchronous dense search or post-processing.
    #[expect(
        clippy::expect_used,
        reason = "private shape invariant validated by the constructor"
    )]
    pub fn as_dataset(&self) -> Dataset<'_, T, I> {
        // All fields are private and every subrange is constructed in bounds.
        Dataset::new(self.data.as_view(), &self.ids)
            .and_then(|d| match &self.norms {
                Some(norms) => d.with_squared_norms(norms),
                None => Ok(d),
            })
            .and_then(|d| {
                d.select(Selection::Range {
                    start: self.rows.start,
                    end: self.rows.end,
                })
            })
            .expect("owned block invariant")
    }
}

/// A one-pass source with stable dimensions and block/row order. All candidates
/// must be delivered before None; a failed read is never a successful partial scan.
/// Sources may filter or await metadata before yielding their owned blocks.
pub trait BatchSource: HasId + Send + Sync {
    type Element: GemmElement;
    type Error: ToRanked + Debug + Send + Sync + 'static;

    fn dimensions(&self) -> usize;

    /// Return a nonempty block of at most max_rows, or None at end of scan.
    fn next_block(
        &mut self,
        max_rows: NonZeroUsize,
    ) -> impl SendFuture<BlockResult<Self::Element, Self::Id, Self::Error>>;
}

/// A block, end of stream, or a source-specific read failure.
pub type BlockResult<T, I, E> = Result<Option<OwnedBlock<T, I>>, E>;

/// Shares immutable database storage with workers; reading a block does not copy vectors.
#[derive(Debug)]
pub struct MemorySource<T, I> {
    block: OwnedBlock<T, I>,
    position: usize,
}

impl<T, I> MemorySource<T, I> {
    pub fn new(block: OwnedBlock<T, I>) -> Self {
        Self { block, position: 0 }
    }

    /// Storage remains available to post-processors after the scan is consumed.
    pub fn storage(&self) -> &OwnedBlock<T, I> {
        &self.block
    }
}

impl<T, I: VectorId> HasId for MemorySource<T, I> {
    type Id = I;
}

impl<T: GemmElement, I: VectorId> BatchSource for MemorySource<T, I> {
    type Element = T;
    type Error = crate::error::Infallible;
    fn dimensions(&self) -> usize {
        self.block.dimensions()
    }
    async fn next_block(
        &mut self,
        max_rows: NonZeroUsize,
    ) -> Result<Option<OwnedBlock<T, I>>, Self::Error> {
        if self.position == self.block.nrows() {
            return Ok(None);
        }
        let len = max_rows.get().min(self.block.nrows() - self.position);
        let start = self.block.rows.start + self.position;
        self.position += len;
        Ok(Some(OwnedBlock {
            data: self.block.data.clone(),
            ids: self.block.ids.clone(),
            rows: start..start + len,
            norms: self.block.norms.clone(),
        }))
    }
}

async fn read<S: BatchSource>(
    source: &mut S,
    max_rows: NonZeroUsize,
    dim: usize,
) -> ANNResult<Option<OwnedBlock<S::Element, S::Id>>> {
    let block = source
        .next_block(max_rows)
        .await
        .escalate("flat scan must complete to produce correct k-NN results")?;
    if let Some(b) = &block
        && (b.nrows() == 0 || b.nrows() > max_rows.get() || b.dimensions() != dim)
    {
        return Err(ANNError::message(
            "batch source returned an invalid block shape",
        ));
    }
    Ok(block)
}

/// Search a source once, overlapping one prefetched block with CPU computation.
/// The supplied pool is the only compute pool; at most one compute job is in
/// flight and each job respects the per-request worker limit.
///
/// Post-processing runs after the complete scan, on the same mutable source,
/// once per query. It may await further reads. Filtering here precedes Top-k;
/// post-processing that removes results does not trigger a refill.
/// Dropping the future stops further reads/submissions; an already running job
/// may finish, retaining only owned data. CPU panics are resumed on the caller.
///
/// # Errors
/// Returns an error for invalid shapes/options, scan errors, computation errors,
/// or post-processing errors. Read/compute errors leave output buffers untouched;
/// a post-processing error can leave earlier queries' outputs populated.
#[expect(
    clippy::too_many_arguments,
    reason = "batch equivalent of search with processor and output"
)]
pub async fn knn_search_batch<S, F, PP, O, OB>(
    source: &mut S,
    queries: Arc<rowmajor::Owned<S::Element>>,
    k: NonZeroUsize,
    metric: Metric,
    filter: Arc<F>,
    options: &BatchOptions,
    pool: Arc<ThreadPool>,
    processor: PP,
    output: &mut [OB],
) -> ANNResult<Vec<BatchStats>>
where
    S: BatchSource,
    F: CandidateFilter<S::Id> + Send + 'static,
    PP: for<'a> SearchPostProcess<S, &'a [S::Element], O> + Send + Sync,
    O: Send,
    OB: SearchOutputBuffer<O> + Send,
{
    let post_queries = queries.clone();
    check_len::<Partial<S::Id>>(queries.nrows())?;
    if output.len() != queries.nrows() {
        return Err(ANNError::message(
            "batch query and output dimensions differ",
        ));
    }
    let contexts: Vec<_> = post_queries.rows().collect();
    knn_search_batch_with_contexts(
        source, queries, k, metric, filter, options, pool, processor, &contexts, output,
    )
    .await
}

/// Batched search with caller-defined per-query post-processing context.
/// Contexts stay on the caller; only vectors and the filter are sent to workers.
///
/// # Errors
/// Has the same failure and output guarantees as knn_search_batch. Also returns
/// an error if there is not exactly one context and output buffer per query.
#[expect(
    clippy::too_many_arguments,
    reason = "batch search with explicit per-query post-processing context"
)]
pub async fn knn_search_batch_with_contexts<S, F, Q, PP, O, OB>(
    source: &mut S,
    queries: Arc<rowmajor::Owned<S::Element>>,
    k: NonZeroUsize,
    metric: Metric,
    filter: Arc<F>,
    options: &BatchOptions,
    pool: Arc<ThreadPool>,
    processor: PP,
    contexts: &[Q],
    output: &mut [OB],
) -> ANNResult<Vec<BatchStats>>
where
    S: BatchSource,
    F: CandidateFilter<S::Id> + Send + 'static,
    Q: Copy + Send + Sync,
    PP: SearchPostProcess<S, Q, O> + Send + Sync,
    O: Send,
    OB: SearchOutputBuffer<O> + Send,
{
    options.validate()?;
    if source.dimensions() != queries.ncols()
        || output.len() != queries.nrows()
        || contexts.len() != queries.nrows()
    {
        return Err(ANNError::message(
            "batch query, source, or output dimensions differ",
        ));
    }
    check_len::<Partial<S::Id>>(queries.nrows())?;
    if queries.nrows() == 0 {
        return Ok(Vec::new());
    }
    let mut results: Vec<_> = (0..queries.nrows()).map(|_| Partial::empty()).collect();
    let block_size = options.read_block_size;
    let mut current = read(source, block_size, queries.ncols()).await?;
    let mut ordinal = 0usize;
    while let Some(block) = current {
        let next_ordinal = ordinal
            .checked_add(block.nrows())
            .ok_or_else(|| ANNError::message("flat scan ordinal overflow"))?;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let task_queries = queries.clone();
        let task_filter = filter.clone();
        let task_pool = pool.clone();
        let task_options = *options;
        pool.spawn(move || {
            if sender.is_closed() {
                return;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                kernel::search(
                    block.as_dataset(),
                    task_queries.as_view(),
                    k,
                    metric,
                    task_filter.as_ref(),
                    &task_options,
                    &task_pool,
                    ordinal,
                )
            }));
            let _ = sender.send(result);
        });
        let compute = async {
            match receiver
                .await
                .map_err(|e| ANNError::new(e).context("flat compute worker disconnected"))?
            {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            }
        };
        let (partials, next) =
            futures_util::future::try_join(compute, read(source, block_size, queries.ncols()))
                .await?;
        for (dst, src) in results.iter_mut().zip(partials) {
            dst.merge(src, k.get())?;
        }
        ordinal = next_ordinal;
        current = next;
    }
    let mut stats = Vec::with_capacity(results.len());
    for (q, (result, buffer)) in results.into_iter().zip(output).enumerate() {
        let mut result = result.finish();
        let count = processor
            .post_process(source, contexts[q], result.neighbors.into_iter(), buffer)
            .await
            .into_ann_result()?;
        result.stats.result_count = count as u64;
        stats.push(result.stats);
    }
    Ok(stats)
}
