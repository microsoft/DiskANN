/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph step of the disk index build.
//!
//! PiPNN replaces the Vamana graph step and writes the same intermediate
//! in-memory index file, so the PQ and disk layout steps of the disk build do
//! not change.

use diskann::{
    ANNResult,
    graph::{AdjacencyList, pipnn},
    utils::VectorRepr,
};
use diskann_providers::{
    common::MinMaxElement,
    storage::{StorageReadProvider, StorageWriteProvider},
    utils::RayonThreadPoolRef,
};
use diskann_quantization::{
    bits::{BitSlice, Representation, Unsigned},
    distances::InnerProduct,
    minmax::{
        DataRef, MinMaxCompensation, MinMaxCosine, MinMaxCosineNormalized, MinMaxIP,
        MinMaxL2Squared,
    },
};
use diskann_utils::{
    io::read_bin,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::Metric};

use super::inmem_builder;
use crate::error::{ErrorKind, diskann_error};

/// Element types supported by the disk PiPNN builder.
///
/// Plain scalar types (`f32`, [`half::f16`], `u8`, `i8`) build through
/// [`pipnn::Rows`]; [`MinMaxElement`] builds through a pre-validated
/// [`pipnn::Dataset`].
pub trait PipnnBuildElement: VectorRepr {
    #[doc(hidden)]
    fn build_graph(
        data: rowmajor::Ref<'_, Self>,
        metric: Metric,
        config: &pipnn::Config,
    ) -> ANNResult<Vec<AdjacencyList<u32>>>;
}

fn build_rows<T: pipnn::Element>(
    data: rowmajor::Ref<'_, T>,
    metric: Metric,
    config: &pipnn::Config,
) -> ANNResult<Vec<AdjacencyList<u32>>> {
    pipnn::build_graph(&pipnn::Rows::new(data, metric), config)
}

impl PipnnBuildElement for f32 {
    fn build_graph(
        data: rowmajor::Ref<'_, Self>,
        metric: Metric,
        config: &pipnn::Config,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        build_rows(data, metric, config)
    }
}

impl PipnnBuildElement for half::f16 {
    fn build_graph(
        data: rowmajor::Ref<'_, Self>,
        metric: Metric,
        config: &pipnn::Config,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        build_rows(data, metric, config)
    }
}

impl PipnnBuildElement for u8 {
    fn build_graph(
        data: rowmajor::Ref<'_, Self>,
        metric: Metric,
        config: &pipnn::Config,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        build_rows(data, metric, config)
    }
}

impl PipnnBuildElement for i8 {
    fn build_graph(
        data: rowmajor::Ref<'_, Self>,
        metric: Metric,
        config: &pipnn::Config,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        build_rows(data, metric, config)
    }
}

impl<const NBITS: usize> PipnnBuildElement for MinMaxElement<NBITS>
where
    Self: VectorRepr,
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'a, 'b> PureDistanceFunction<
            BitSlice<'a, NBITS, Unsigned>,
            BitSlice<'b, NBITS, Unsigned>,
            diskann_quantization::distances::MathematicalResult<u32>,
        >,
{
    fn build_graph(
        data: rowmajor::Ref<'_, Self>,
        metric: Metric,
        config: &pipnn::Config,
    ) -> ANNResult<Vec<AdjacencyList<u32>>> {
        let dataset = MinMaxDataset::new(data, metric)?;
        pipnn::build_graph(&dataset, config)
    }
}

/// Pre-validated view of MinMax quantized rows implementing [`pipnn::Dataset`].
///
/// Construction validates every row's embedded [`MinMaxCompensation`] header
/// and checks that all rows share one full dimension and canonical byte length,
/// keeping fallible MinMax metadata decoding out of [`pipnn::Element`].
#[derive(Debug)]
struct MinMaxDataset<'a, const NBITS: usize> {
    data: rowmajor::Ref<'a, MinMaxElement<NBITS>>,
    dimensions: usize,
    canonical_bytes: usize,
    metric: Metric,
}

impl<'a, const NBITS: usize> MinMaxDataset<'a, NBITS>
where
    Unsigned: Representation<NBITS>,
{
    fn new(data: rowmajor::Ref<'a, MinMaxElement<NBITS>>, metric: Metric) -> ANNResult<Self> {
        let (dimensions, canonical_bytes) = match data.get_row(0) {
            None => (0, 0),
            Some(first_row) => {
                let first_bytes: &[u8] = bytemuck::must_cast_slice(first_row);
                let dimensions =
                    MinMaxCompensation::read_dimension(first_bytes).map_err(|err| {
                        diskann_error!(
                            ErrorKind::IndexError,
                            "invalid MinMax metadata at row 0: {err}"
                        )
                    })?;
                let canonical_bytes = dimensions
                    .checked_mul(NBITS)
                    .map(|bits| bits.div_ceil(8))
                    .and_then(|bytes| bytes.checked_add(size_of::<MinMaxCompensation>()))
                    .ok_or_else(|| {
                        diskann_error!(
                            ErrorKind::IndexError,
                            "MinMax dimension {dimensions} overflows canonical byte size"
                        )
                    })?;
                if data.ncols() < canonical_bytes {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "MinMax row width {} is smaller than canonical size {} for dimension {}",
                        data.ncols(),
                        canonical_bytes,
                        dimensions
                    ));
                }
                DataRef::<NBITS>::from_canonical_front(&first_bytes[..canonical_bytes], dimensions)
                    .map_err(|err| {
                        diskann_error!(ErrorKind::IndexError, "non-canonical MinMax row 0: {err}")
                    })?;

                for (row_idx, row) in data.rows().enumerate().skip(1) {
                    let row_bytes: &[u8] = bytemuck::must_cast_slice(row);
                    let row_dim = MinMaxCompensation::read_dimension(row_bytes).map_err(|err| {
                        diskann_error!(
                            ErrorKind::IndexError,
                            "invalid MinMax metadata at row {row_idx}: {err}"
                        )
                    })?;
                    if row_dim != dimensions {
                        return Err(diskann_error!(
                            ErrorKind::IndexError,
                            "MinMax row {row_idx} declares dimension {row_dim}, expected {dimensions}"
                        ));
                    }
                }
                (dimensions, canonical_bytes)
            }
        };

        Ok(Self {
            data,
            dimensions,
            canonical_bytes,
            metric,
        })
    }

    #[inline]
    fn row(&self, id: u32) -> DataRef<'_, NBITS> {
        let bytes: &[u8] = bytemuck::must_cast_slice(self.data.row(id as usize));
        DataRef::<'_, NBITS>::from_canonical_front(&bytes[..self.canonical_bytes], self.dimensions)
            .expect("MinMaxDataset::new validated every row's canonical length")
    }
}

impl<const NBITS: usize> pipnn::Dataset for MinMaxDataset<'_, NBITS>
where
    Unsigned: Representation<NBITS>,
    InnerProduct: for<'x, 'y> PureDistanceFunction<
            BitSlice<'x, NBITS, Unsigned>,
            BitSlice<'y, NBITS, Unsigned>,
            diskann_quantization::distances::MathematicalResult<u32>,
        >,
{
    fn nrows(&self) -> usize {
        self.data.nrows()
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn gather(&self, ids: &[u32], mut output: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
        assert_eq!(ids.len(), output.nrows());
        assert_eq!(output.ncols(), self.dimensions);
        for (&id, destination) in ids.iter().zip(output.rows_mut()) {
            self.row(id).decompress_into(destination).map_err(|err| {
                diskann_error!(ErrorKind::IndexError, "MinMax decompression failed: {err}")
            })?;
        }
        Ok(())
    }

    fn distance(&self, source: u32, target: u32) -> f32 {
        let (x, y) = (self.row(source), self.row(target));
        match self.metric {
            Metric::Cosine => MinMaxCosine::evaluate(x, y),
            Metric::InnerProduct => MinMaxIP::evaluate(x, y),
            Metric::L2 => MinMaxL2Squared::evaluate(x, y),
            Metric::CosineNormalized => MinMaxCosineNormalized::evaluate(x, y),
        }
        .expect("MinMaxDataset::new validated uniform row dimensions")
    }
}

/// Build the PiPNN graph of the dataset at `data_path` and write it to
/// `save_path` as an in-memory index file.
///
/// The caller's `start_id` is recorded in the graph header.
pub(super) fn build_inmem_index<T, StorageProvider>(
    metric: Metric,
    start_id: u32,
    config: &pipnn::Config,
    data_path: &str,
    save_path: &str,
    storage_provider: &StorageProvider,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<()>
where
    T: PipnnBuildElement,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let graph = build_graph::<T, _>(metric, config, data_path, storage_provider, pool)?;
    let writer = storage_provider.create_for_write(save_path)?;
    inmem_builder::write_graph(graph.iter().map(|row| Ok(&**row)), writer, start_id)?;
    Ok(())
}

/// Read the dataset at `data_path` and build its PiPNN graph on `pool`.
///
/// The dataset is dropped on return, so only the graph stays in memory while
/// it is written.
fn build_graph<T, StorageProvider>(
    metric: Metric,
    config: &pipnn::Config,
    data_path: &str,
    storage_provider: &StorageProvider,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: PipnnBuildElement,
    StorageProvider: StorageReadProvider,
{
    let data = read_bin::<T>(&mut storage_provider.open_reader(data_path)?)?;
    pool.install(|| T::build_graph(data.as_view(), metric, config))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use diskann::graph::pipnn::Dataset;
    use diskann_providers::{storage::VirtualStorageProvider, utils::create_thread_pool};
    use diskann_quantization::{
        CompressInto,
        algorithms::{Transform, transforms::NullTransform},
        minmax::{DataMutRef, MinMaxQuantizer},
        num::Positive,
    };
    use diskann_utils::{ReborrowMut, io::write_bin, views::rowmajor};
    use diskann_vector::{DistanceFunction, distance::Metric};
    use vfs::MemoryFS;

    use super::*;

    const DATA: &str = "/data.fbin";
    const GRAPH: &str = "/graph";
    const DEGREE: usize = 32;
    const SEED: u64 = 42;

    fn pipnn_config() -> pipnn::Config {
        pipnn::Config::builder(NonZeroUsize::new(DEGREE).unwrap())
            .seed(SEED)
            .build()
            .unwrap()
    }

    fn storage_with_data(points: usize, dimensions: usize) -> VirtualStorageProvider<MemoryFS> {
        let storage = VirtualStorageProvider::new_memory();
        let data: Vec<f32> = (0..points * dimensions)
            .map(|index| ((index * 17) % 251) as f32)
            .collect();
        write_bin(
            rowmajor::Ref::try_from_data(data.as_slice(), points, dimensions).unwrap(),
            &mut storage.create_for_write(DATA).unwrap(),
        )
        .unwrap();
        storage
    }

    fn build(storage: &VirtualStorageProvider<MemoryFS>, start_id: u32) -> ANNResult<()> {
        let pool = create_thread_pool(1)?;
        build_inmem_index::<f32, _>(
            Metric::L2,
            start_id,
            &pipnn_config(),
            DATA,
            GRAPH,
            storage,
            pool.as_ref(),
        )
    }

    fn compress_minmax_rows<const NBITS: usize>(
        rows: &[Vec<f32>],
    ) -> (Vec<MinMaxElement<NBITS>>, usize)
    where
        Unsigned: Representation<NBITS>,
    {
        let dim = rows[0].len();
        let transform = Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap()));
        let quantizer = MinMaxQuantizer::new(transform, Positive::new(1.0).unwrap());
        let row_bytes = DataRef::<NBITS>::canonical_bytes(dim);
        let mut flat = Vec::with_capacity(rows.len() * row_bytes);
        for row in rows {
            let mut bytes = vec![0_u8; row_bytes];
            let mut dst = DataMutRef::<NBITS>::from_canonical_front_mut(&mut bytes, dim).unwrap();
            quantizer
                .compress_into(row.as_slice(), dst.reborrow_mut())
                .unwrap();
            flat.extend_from_slice(MinMaxElement::<NBITS>::from_bytes(&bytes));
        }
        (flat, row_bytes)
    }

    #[test]
    fn build_writes_the_graph_in_the_inmem_index_format() {
        let (points, dimensions) = (256, 8);
        let storage = storage_with_data(points, dimensions);

        let start_id = 17;
        build(&storage, start_id).unwrap();

        let mut file = Vec::new();
        std::io::Read::read_to_end(&mut storage.open_reader(GRAPH).unwrap(), &mut file).unwrap();
        let word = |index: usize| u32::from_le_bytes(file[4 * index..][..4].try_into().unwrap());

        assert_eq!(
            u64::from_le_bytes(file[..8].try_into().unwrap()),
            file.len() as u64
        );
        let header_max_degree = word(2) as usize;
        assert_eq!(word(3), start_id);
        assert_eq!(u64::from_le_bytes(file[16..24].try_into().unwrap()), 0);
        // Each row is its length and then its neighbor IDs.
        let mut index = 6;
        let mut observed_max_degree = 0;
        for _ in 0..points {
            let len = word(index) as usize;
            assert!(len <= DEGREE);
            observed_max_degree = observed_max_degree.max(len);
            assert!(
                (index + 1..index + 1 + len).all(|neighbor| (word(neighbor) as usize) < points)
            );
            index += 1 + len;
        }
        assert!(observed_max_degree > 0);
        assert_eq!(header_max_degree, observed_max_degree);
        assert_eq!(4 * index, file.len());
    }

    fn verify_minmax_dataset_matches_vector_repr<const NBITS: usize>()
    where
        MinMaxElement<NBITS>: VectorRepr,
        Unsigned: Representation<NBITS>,
        InnerProduct: for<'a, 'b> PureDistanceFunction<
                BitSlice<'a, NBITS, Unsigned>,
                BitSlice<'b, NBITS, Unsigned>,
                diskann_quantization::distances::MathematicalResult<u32>,
            >,
    {
        let dim = 17;
        let vectors: Vec<Vec<f32>> = (0..6)
            .map(|r| {
                (0..dim)
                    .map(|c| ((r * 31 + c * 13) % 97) as f32 * 0.1 - 4.0)
                    .collect()
            })
            .collect();
        let (flat, ncols) = compress_minmax_rows::<NBITS>(&vectors);
        let view = rowmajor::Ref::try_from_data(&flat, vectors.len(), ncols).unwrap();

        for metric in [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ] {
            let dataset = MinMaxDataset::new(view, metric).unwrap();
            assert_eq!(dataset.nrows(), vectors.len());
            assert_eq!(dataset.dimensions(), dim);
            assert_eq!(dataset.metric(), metric);

            let ids = [4_u32, 1, 5, 0];
            let mut gathered = rowmajor::Owned::from_fn(ids.len(), dim, |_| 0.0_f32);
            dataset.gather(&ids, gathered.as_view_mut()).unwrap();
            for (slot, &id) in ids.iter().enumerate() {
                let mut expected = vec![0.0_f32; dim];
                MinMaxElement::<NBITS>::as_f32_into(view.row(id as usize), &mut expected).unwrap();
                assert_eq!(gathered.row(slot), expected.as_slice());
            }

            let reference = MinMaxElement::<NBITS>::distance(metric, None);
            for i in 0..vectors.len() {
                for j in 0..vectors.len() {
                    let actual = dataset.distance(i as u32, j as u32);
                    let expected = reference.evaluate_similarity(view.row(i), view.row(j));
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            }
        }
    }

    #[test]
    fn minmax_dataset_matches_vector_repr_across_bitrates() {
        verify_minmax_dataset_matches_vector_repr::<1>();
        verify_minmax_dataset_matches_vector_repr::<2>();
        verify_minmax_dataset_matches_vector_repr::<4>();
        verify_minmax_dataset_matches_vector_repr::<8>();
    }

    #[test]
    fn minmax_dataset_accepts_empty_matrix() {
        let empty: [MinMaxElement<8>; 0] = [];
        let view = rowmajor::Ref::try_from_data(&empty, 0, 0).unwrap();
        let dataset = MinMaxDataset::new(view, Metric::L2).unwrap();
        assert_eq!(dataset.nrows(), 0);
        assert_eq!(dataset.dimensions(), 0);
        assert_eq!(dataset.metric(), Metric::L2);
    }

    #[test]
    fn minmax_dataset_rejects_short_metadata() {
        let raw = vec![0_u8; size_of::<MinMaxCompensation>() - 1];
        let elems = MinMaxElement::<8>::from_bytes(&raw);
        let view = rowmajor::Ref::try_from_data(elems, 1, elems.len()).unwrap();
        let err = MinMaxDataset::new(view, Metric::L2).unwrap_err();
        assert!(err.to_string().contains("invalid MinMax metadata"), "{err}");
    }

    #[test]
    fn minmax_dataset_rejects_truncated_canonical_row() {
        // Metadata declares dim = 32 (requires 20 + 32 = 52 bytes), but row only has 24 bytes.
        let mut raw = vec![0_u8; 24];
        raw[..4].copy_from_slice(&32_u32.to_le_bytes());
        let elems = MinMaxElement::<8>::from_bytes(&raw);
        let view = rowmajor::Ref::try_from_data(elems, 1, elems.len()).unwrap();
        let err = MinMaxDataset::new(view, Metric::L2).unwrap_err();
        assert!(
            err.to_string().contains("smaller than canonical size"),
            "{err}"
        );
    }

    #[test]
    fn minmax_dataset_rejects_mismatched_row_dimensions() {
        let rows = vec![vec![1.0_f32; 16], vec![2.0_f32; 16]];
        let (mut flat, ncols) = compress_minmax_rows::<8>(&rows);
        // Corrupt row 1's embedded dimension header from 16 to 8.
        let raw: &mut [u8] = bytemuck::must_cast_slice_mut(&mut flat);
        raw[ncols..ncols + 4].copy_from_slice(&8_u32.to_le_bytes());
        let view = rowmajor::Ref::try_from_data(&flat, 2, ncols).unwrap();
        let err = MinMaxDataset::new(view, Metric::L2).unwrap_err();
        assert!(
            err.to_string()
                .contains("row 1 declares dimension 8, expected 16"),
            "{err}"
        );
    }
}
