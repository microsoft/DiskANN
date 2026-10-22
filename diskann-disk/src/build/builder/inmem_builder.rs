/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::{Seek, Write},
    pin::Pin,
    sync::Arc,
};

use diskann::{
    graph::{Config, DiskANNIndex},
    utils::{IntoUsize, VectorRepr},
    ANNError, ANNResult,
};
use diskann_inmem::{self as di, repr::quantized::Backend};
use diskann_providers::model::graph::provider::async_::inmem::DefaultProviderParameters;
use diskann_utils::{
    future::SendFuture,
    views::rowmajor::{self, Matrix},
};
use tracing::debug;

use super::quantizer::BuildQuantizer;

pub(super) trait WriteSeek: Write + Seek {}
impl<T> WriteSeek for T where T: Write + Seek {}

/// Builder facade for in memory index construction and persistence.
///
/// Thread safety:
/// Implementors must be `Send` and `Sync`. Methods can be called from many tasks.
pub(super) trait InmemIndexBuilder<T: Sized>: Send + Sync {
    /// Return the total capacity of the provider, **excluding** start points.
    fn capacity(&self) -> usize;

    /// Return the total capacity of the provider, **including** start points.
    fn total_points(&self) -> usize;

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

    fn save_graph_mut(&mut self, io: &mut dyn WriteSeek, start_id: u32) -> ANNResult<()>;

    /// Return the number of vector reads for full_precision and quantized stores respectively.
    #[cfg(debug_assertions)]
    fn counts_for_get_vector(&self) -> (usize, usize);

    /// Count the number of nodes in the graph reachable from the given `start_points`.
    ///
    /// This function has a large memory footprint for large graphs and should not be called
    /// frequently. This is mainly for analysis and sanity tests.
    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&self) -> Pin<Box<dyn SendFuture<ANNResult<usize>> + '_>>;
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

    pub(crate) fn data(&self) -> &[T] {
        &self.data
    }

    /// Return the `id` associated with this start point.
    pub(crate) fn id(&self) -> u32 {
        self.id
    }
}

//////////////////////////////////
// FullPrecision Implementation //
//////////////////////////////////

impl<T> InmemIndexBuilder<T> for DiskANNIndex<di::provider::buildonly::Provider<di::repr::Full<T>>>
where
    T: di::repr::FullPrecision,
{
    fn capacity(&self) -> usize {
        self.provider().id_limit().value().into_usize()
    }

    fn total_points(&self) -> usize {
        self.provider().id_limit().value().into_usize()
    }

    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move { self.insert(&di::Strategy, &di::Context, &id, vector).await })
    }

    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>> {
        Box::pin(async move { self.prune_range(&di::Strategy, &di::Context, range).await })
    }

    fn save_graph_mut(&mut self, write: &mut dyn WriteSeek, start_id: u32) -> ANNResult<()> {
        save_graph(self.provider_mut().neighbor_snapshot(), write, start_id)?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn counts_for_get_vector(&self) -> (usize, usize) {
        (0, 0)
    }

    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&self) -> Pin<Box<dyn SendFuture<ANNResult<usize>> + '_>> {
        Box::pin(async move { Ok(0) })
    }
}

//////////////////////////
// Quant Implementation //
//////////////////////////

impl<T> InmemIndexBuilder<T>
    for DiskANNIndex<di::provider::buildonly::Provider<di::repr::Quantized>>
where
    T: VectorRepr,
{
    fn capacity(&self) -> usize {
        self.provider().id_limit().value().into_usize()
    }

    fn total_points(&self) -> usize {
        self.provider().id_limit().value().into_usize()
    }

    fn insert_vector<'a>(
        &'a self,
        id: u32,
        vector: &'a [T],
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + 'a>> {
        Box::pin(async move {
            let vector: Box<[f32]> = (*T::as_f32(vector).map_err(ANNError::new)?).into();
            self.insert(&di::Strategy, &di::Context, &id, &vector).await
        })
    }

    fn final_prune(
        &self,
        range: core::ops::Range<u32>,
    ) -> Pin<Box<dyn SendFuture<ANNResult<()>> + '_>> {
        Box::pin(async move { self.prune_range(&di::Strategy, &di::Context, range).await })
    }

    fn save_graph_mut(&mut self, write: &mut dyn WriteSeek, start_id: u32) -> ANNResult<()> {
        save_graph(self.provider_mut().neighbor_snapshot(), write, start_id)?;
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn counts_for_get_vector(&self) -> (usize, usize) {
        (0, 0)
    }

    #[cfg(debug_assertions)]
    fn count_reachable_nodes(&self) -> Pin<Box<dyn SendFuture<ANNResult<usize>> + '_>> {
        Box::pin(async move { Ok(0) })
    }
}

/// Create a new in-memory index builder for vectors of type `T`.
///
/// Chooses the builder implementation based on the given `BuildQuantizer`.
/// - `NoQuant` uses a plain index with no quantization.
/// - `Scalar1Bit`, `Spherical1Bit`, and `PQ` create quantized only indexes backed by
///   `QuantInMemBuilder`.
///
/// # Parameters
/// * `config` – Index configuration.
/// * `build_quantizer` – Quantization strategy to apply.
/// * `start_point` - The value of the data start point.
///
/// # Returns
/// An `Arc<dyn InmemIndexBuilder>` wrapped in `ANNResult`.
///
/// # Errors
/// Returns an error if the underlying index creation fails.
pub(super) fn new_inmem_index_builder<T>(
    config: Config,
    params: DefaultProviderParameters,
    build_quantizer: &BuildQuantizer,
    start_point: StartPoint<T>,
) -> ANNResult<Arc<dyn InmemIndexBuilder<T>>>
where
    T: VectorRepr + di::repr::FullPrecision,
{
    let start_point_ids = di::num::Range::new(start_point.id(), start_point.id() + 1)?;

    let make_quantized = |backend: Backend| -> ANNResult<Arc<_>> {
        let start_point_data = T::as_f32(start_point.data()).map_err(ANNError::new)?;
        let start_point = rowmajor::Ref::row_vector(&*start_point_data).to_rowmajor_owned();

        let quantized = di::repr::Quantized::config(
            backend,
            di::num::Capacity::new(params.max_points),
            di::num::MaxDegree::new(params.max_degree.into_usize()),
            start_point,
            di::repr::quantized::Rerank::None,
        )?
        .embed_start_points(start_point_ids);

        let provider = di::provider::buildonly::Provider::new(quantized)?;
        Ok(Arc::new(DiskANNIndex::new(config.clone(), provider, None)))
    };

    let result: Arc<dyn InmemIndexBuilder<T>> = match &build_quantizer {
        BuildQuantizer::NoQuant(_) => {
            let full = di::repr::Full::config(
                di::num::Capacity::new(params.max_points),
                di::num::MaxDegree::new(params.max_degree.into_usize()),
                params.metric.into(),
                rowmajor::Ref::row_vector(start_point.data()).to_rowmajor_owned(),
            )?
            .embed_start_points(start_point_ids);

            let provider = di::provider::buildonly::Provider::new(full)?;
            let index = diskann::graph::DiskANNIndex::new(config, provider, None);
            Arc::new(index)
        }
        BuildQuantizer::Scalar1Bit(q) => {
            make_quantized(Backend::new(di::repr::quantized::scalar::Scalar::<1>::new(
                q.quantizer().clone(),
                params.metric.try_into()?,
            )))?
        }
        BuildQuantizer::Spherical1Bit(q) => {
            let quantizer = q.try_clone().map_err(ANNError::new)?;
            let with_bits = quantizer.as_quantizer_impl::<1>().map_err(ANNError::new)?;
            make_quantized(Backend::new(
                di::repr::quantized::spherical::Spherical::new(with_bits),
            ))?
        }
        BuildQuantizer::PQ(table) => {
            make_quantized(Backend::new(di::repr::quantized::product::Product::new(
                table.table().clone(),
                params.metric.into(),
                None,
            )?))?
        }
    };

    debug!("Set start point to medoid ID: {}", start_point.id());

    Ok(result)
}

//----------//
// Adaptors //
//----------//

/// Write packed adjacency lists in the disk builder's intermediate graph format.
///
/// Preserves all rows, including empty lists, without validating graph completeness.
/// `start_point` is a dataset ID; no phantom-point rewriting is performed.
/// The writer must be positioned at the beginning of the output.
pub fn save_graph<W: Write + Seek>(
    snapshot: diskann_inmem::neighbors::Snapshot<'_>,
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
