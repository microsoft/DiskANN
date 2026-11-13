/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Seek, Write},
    num::NonZeroUsize,
    pin::Pin,
    sync::Arc,
};

use diskann::{
    ANNError, ANNResult,
    graph::DiskANNIndex,
    utils::{IntoUsize, VectorRepr},
};
use diskann_inmem::{
    self as di,
    num::{Capacity, MaxDegree},
    provider::buildonly::Provider,
};

use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix},
};
use tracing::debug;

/// Configuration for the in-memory builder.
#[derive(Debug, Clone)]
pub(super) struct Config {
    pub(super) non_frozen_points: usize,
    pub(super) prefetch_lookahead: Option<NonZeroUsize>,
    pub(super) num_threads: NonZeroUsize,
}

/// The start point for an inmem build.
#[derive(Debug)]
pub(crate) struct StartPoint<T> {
    data: Vec<T>,
    id: u32,
}

impl<T> StartPoint<T> {
    /// Construct a new singular start point containing `data`.
    pub(crate) fn new(data: Vec<T>, id: u32) -> Self {
        Self { data, id }
    }

    /// Return the data for the start point.
    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    /// Return the `id` associated with this start point.
    pub(crate) fn id(&self) -> u32 {
        self.id
    }
}

/// A factory method for creating an [`InmemIndexBuilder`].
///
/// Factories are needed to support sharded builds where many smaller-scale builders are
/// used with all results merged at the end. A factory allows all sub-builds to use similar
/// settings.
pub(super) trait CreateBuilder<T>: std::fmt::Debug + Send + Sync + 'static {
    /// Crate a [`InmemIndexBuilder`] with the graph and provider configurations.
    ///
    /// The [`StartPoint`] will be eagerly inserted into the memory for the builder and
    /// will be used as the graph start point.
    fn create_builder(
        &self,
        graph: diskann::graph::Config,
        provider: Config,
        start: StartPoint<T>,
    ) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>>;
}

pub(super) trait WriteSeek: Write + Seek {}
impl<T> WriteSeek for T where T: Write + Seek {}

/// Builder facade for in memory index construction and persistence.
///
/// Thread safety:
///
/// Implementors must be `Send` and `Sync`. Methods can be called from many tasks.
pub(super) trait InmemIndexBuilder<T: Sized>: Send + Sync {
    /// Return the total capacity of the provider, **including** start points.
    fn capacity(&self) -> usize;

    /// Insert a vector with a `id`.
    ///
    /// The slice must match the underlying vector type.
    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>>;

    /// Prune the built graph over `[range.start, range.end)`.
    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>>;

    /// Write the graph into `io`.
    fn save_graph_mut(&mut self, io: &mut dyn WriteSeek) -> ANNResult<()>;

    /// Count the number of nodes in the graph reachable from the given `start_points`.
    ///
    /// This function has a large memory footprint for large graphs and should not be called
    /// frequently. This is mainly for analysis and sanity tests.
    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&mut self) -> usize;
}

////////////////////
// Full Precision //
////////////////////

/// A [`CreateBuilder`] dispatching to [`di::repr::Full`].
///
/// This is compatible with [`di::repr::FullPrecision`] element types.
#[derive(Debug, Clone, Copy)]
pub(super) struct FullPrecision {
    metric: diskann_vector::distance::Metric,
}

impl FullPrecision {
    /// Construct a new [`FullPrecision`] with `metric`.
    pub(super) fn new(metric: diskann_vector::distance::Metric) -> Self {
        Self { metric }
    }
}

impl<T> CreateBuilder<T> for FullPrecision
where
    T: di::repr::FullPrecision,
{
    fn create_builder(
        &self,
        graph: diskann::graph::Config,
        provider: Config,
        start: StartPoint<T>,
    ) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>> {
        let Self { metric } = *self;

        let Config {
            non_frozen_points,
            num_threads,
            prefetch_lookahead,
        } = provider;

        let start_point_ids = di::num::Range::start_plus(start.id(), 1)
            .ok_or_else(|| ANNError::message("start point ID should not be u32::MAX"))?;

        let full = di::repr::Full::config(
            Capacity::new(non_frozen_points),
            MaxDegree::new(graph.max_degree().get()),
            metric,
            rowmajor::Ref::row_vector(start.data()).to_rowmajor_owned(),
        )?
        .prefetch(prefetch_lookahead)
        .store(store_config(num_threads))
        .locate_start_point_ids_at(start_point_ids)?;

        let provider = Provider::new(full)?;
        let index = Index {
            index: DiskANNIndex::new(graph, provider, None),
            start_id: start.id(),
        };

        debug!("Set start point to medoid ID: {}", start.id());
        Ok(Arc::new(index))
    }
}

impl<T> InmemIndexBuilder<T> for Index<di::repr::Full<T>>
where
    T: di::repr::FullPrecision,
{
    fn capacity(&self) -> usize {
        self.index.provider().id_limit().value().into_usize()
    }

    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        // Skip inserting the start ID since it's already in the graph.
        if id == self.start_id {
            Box::pin(std::future::ready(Ok(())))
        } else {
            Box::pin(async move {
                self.index
                    .insert(&di::Strategy, &di::Context, &id, vector)
                    .await
            })
        }
    }

    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>> {
        Box::pin(async move {
            self.index
                .prune_range(&di::Strategy, &di::Context, range)
                .await
        })
    }

    fn save_graph_mut(&mut self, write: &mut dyn WriteSeek) -> ANNResult<()> {
        save_graph(
            self.index.provider_mut().inspect_neighbors(),
            write,
            self.start_id,
        )?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&mut self) -> usize {
        count_reachable_nodes(self.index.provider_mut().inspect_neighbors(), self.start_id)
    }
}

///////////////
// Quantized //
///////////////

pub(super) trait QuantizedBackend: std::fmt::Debug + Send + Sync + 'static {
    fn quantized_backend(&self) -> di::repr::quantized::Backend;
}

#[derive(Debug)]
pub(super) struct Quantized<Q>(Q);

impl<Q> Quantized<Q> {
    pub(super) fn new(quantizer: Q) -> Self {
        Self(quantizer)
    }
}

impl<T, Q> CreateBuilder<T> for Quantized<Q>
where
    T: VectorRepr,
    Q: QuantizedBackend,
{
    fn create_builder(
        &self,
        graph: diskann::graph::Config,
        provider: Config,
        start: StartPoint<T>,
    ) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>> {
        quantized(self.0.quantized_backend(), graph, provider, start)
    }
}

// This merely wraps `backend` in an `InmemIndexBuilder` that uses [`VectorRepr`] to convert
// incoming data to `f32` before passing it to `backend`.
//
// Technically, `VectorRepr` is way more than is strictly needed, but it's what we have
// to work with.
fn quantized<T>(
    backend: di::repr::quantized::Backend,
    graph: diskann::graph::Config,
    provider: Config,
    start: StartPoint<T>,
) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>>
where
    T: VectorRepr,
{
    let Config {
        non_frozen_points,
        num_threads,
        prefetch_lookahead,
    } = provider;

    let start_point_ids = di::num::Range::start_plus(start.id(), 1)
        .ok_or_else(|| ANNError::message("start point ID should not be u32::MAX"))?;

    let start_point_data = T::as_f32(start.data()).map_err(ANNError::new)?;
    let quantized = di::repr::Quantized::config(
        backend,
        Capacity::new(non_frozen_points),
        MaxDegree::new(graph.max_degree().get()),
        rowmajor::Ref::row_vector(&start_point_data).to_rowmajor_owned(),
        di::repr::quantized::Rerank::None,
    )?
    .prefetch(prefetch_lookahead)
    .store(store_config(num_threads))
    .locate_start_point_ids_at(start_point_ids)?;

    let provider = Provider::new(quantized)?;
    let index = Index {
        index: DiskANNIndex::new(graph, provider, None),
        start_id: start.id(),
    };

    debug!("Set start point to medoid ID: {}", start.id());
    Ok(Arc::new(index))
}

impl<T> InmemIndexBuilder<T> for Index<di::repr::Quantized>
where
    T: VectorRepr,
{
    fn capacity(&self) -> usize {
        self.index.provider().id_limit().value().into_usize()
    }

    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        // Skip inserting the start ID since it's already in the graph.
        if id == self.start_id {
            Box::pin(std::future::ready(Ok(())))
        } else {
            Box::pin(async move {
                let vector: Box<[f32]> = (*T::as_f32(vector).map_err(ANNError::new)?).into();
                self.index
                    .insert(&di::Strategy, &di::Context, &id, &vector)
                    .await
            })
        }
    }

    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>> {
        Box::pin(async move {
            self.index
                .prune_range(&di::Strategy, &di::Context, range)
                .await
        })
    }

    fn save_graph_mut(&mut self, write: &mut dyn WriteSeek) -> ANNResult<()> {
        save_graph(
            self.index.provider_mut().inspect_neighbors(),
            write,
            self.start_id,
        )?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&mut self) -> usize {
        count_reachable_nodes(self.index.provider_mut().inspect_neighbors(), self.start_id)
    }
}

//----------//
// Internal //
//----------//

fn store_config(num_threads: NonZeroUsize) -> di::store::Config {
    // Use 4x the number of threads.
    //
    // If the user is really in a situation where multiplying the number of threads by 4
    // causes the count to overflow `usize::MAX` - they are going to have much larger
    // problems than simply running out of guard slots.
    //
    // FIXME: Decrease headroom once `diskann-inmem` doesn't need to acquire 2 guard slots
    // per insert.
    const MULTIPLIER: NonZeroUsize = NonZeroUsize::new(4).unwrap();

    let mut config = di::store::Config::new();
    if let Some(slots) = num_threads.checked_mul(MULTIPLIER) {
        config.epoch_guard_slots(slots);
    }

    config
}

/// A small wrapper for `DiskANNIndex` that keeps track of the construction start point.
///
/// The build algorithms in this crate embed the start point within data being built. Keeping
/// the start ID bundled with the index lets us:
///
/// 1. Natrually skip the call to `insert_vector` for the start ID since that will cause
///    the upstream insertion algorithm to fail.
///
/// 2. Save the start ID during graph serialization without requiring the caller to provide
///    it.
#[derive(Debug)]
struct Index<R>
where
    R: di::repr::Representation,
{
    index: DiskANNIndex<Provider<R>>,
    start_id: u32,
}

#[cfg(debug_assertions)]
fn count_reachable_nodes(
    snapshot: diskann_inmem::repr::NeighborSnapshot<'_>,
    start_point: u32,
) -> usize {
    let mut seen = vec![false; snapshot.entries().into_usize()];
    seen[start_point.into_usize()] = true;

    let mut count = 1;
    let mut queue = vec![start_point];
    while let Some(current) = queue.pop() {
        let Some(neighbors) = snapshot.get(current) else {
            // Since this is compiled under debug-assertions, we can use this to assert
            // that all elements explored are in-bounds.
            panic!("graph construction yielded an out-of-bounds neighbor {current}");
        };

        for n in neighbors.iter() {
            // Panic on out-of-bounds.
            let seen = &mut seen[n.into_usize()];
            if !*seen {
                *seen = true;
                count += 1;
                queue.push(*n);
            }
        }
    }

    count
}

/// Write packed adjacency lists in the disk builder's intermediate graph format.
///
/// Preserves all rows, including empty lists, without validating graph completeness.
/// `start_point` is a dataset ID; no phantom-point rewriting is performed.
/// The writer must be positioned at the beginning of the output.
fn save_graph<W: Write + Seek>(
    snapshot: diskann_inmem::repr::NeighborSnapshot<'_>,
    writer: W,
    start_point: u32,
) -> std::io::Result<u64> {
    let mut out = std::io::BufWriter::new(writer);
    let mut index_size = 24u64;
    let mut max_degree = 0u32;

    out.write_all(&index_size.to_le_bytes())?;
    out.write_all(&max_degree.to_le_bytes())?;
    out.write_all(&start_point.to_le_bytes())?;
    out.write_all(&0u64.to_le_bytes())?;

    for id in 0..snapshot.entries() {
        let neighbors = snapshot.get(id).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("snapshot is missing adjacency list {id}"),
            )
        })?;
        let degree = u32::try_from(neighbors.len())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        out.write_all(&degree.to_le_bytes())?;
        for neighbor in neighbors {
            out.write_all(&neighbor.to_le_bytes())?;
        }

        max_degree = max_degree.max(degree);
        index_size += (1 + u64::from(degree)) * size_of::<u32>() as u64;
    }

    out.seek(std::io::SeekFrom::Start(0))?;
    out.write_all(&index_size.to_le_bytes())?;
    out.write_all(&max_degree.to_le_bytes())?;
    out.flush()?;
    Ok(index_size)
}
