/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Parquet persistence for in-memory Vamana adjacency lists.

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, UInt32Builder};
use arrow_array::types::UInt32Type;
use arrow_array::{Array, ArrayRef, ListArray, PrimitiveArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use diskann::{ANNError, graph::AdjacencyList};
use futures_util::StreamExt;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

use super::simple_neighbor_provider::SimpleNeighborProviderAsync;

/// The Parquet artifact type for an in-memory Vamana graph.
pub const GRAPH_ARTIFACT_TYPE: &str = "vector.search-structure";
/// The first version of the Vamana graph Parquet schema.
pub const GRAPH_ENCODING_VERSION: u16 = 1;

const BATCH_SIZE: usize = 8192;
const ARTIFACT_TYPE_KEY: &str = "diskann.artifact-type";
const ENCODING_VERSION_KEY: &str = "diskann.encoding-version";
const TOTAL_POINTS_KEY: &str = "diskann.total-points";
const MUTABLE_CAPACITY_KEY: &str = "diskann.mutable-capacity";
const NUM_START_POINTS_KEY: &str = "diskann.num-start-points";
const START_POINT_IDS_KEY: &str = "diskann.start-point-ids";
const MAX_DEGREE_KEY: &str = "diskann.max-degree";
const CHECKSUM_KEY: &str = "diskann.logical-checksum-crc32";

/// Layout metadata required to interpret an in-memory Vamana graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphParquetLayout {
    /// Number of mutable point slots before frozen start points.
    pub mutable_capacity: u32,
    /// Frozen start-point IDs in canonical order.
    pub start_point_ids: Vec<u32>,
    /// Allocated maximum adjacency-list degree.
    pub max_degree: u32,
}

impl GraphParquetLayout {
    /// Returns the total point count, including frozen start points.
    pub fn total_points(&self) -> Result<u32, GraphParquetError> {
        self.mutable_capacity
            .checked_add(u32::try_from(self.start_point_ids.len()).map_err(|_| {
                GraphParquetError::Invalid("start-point count cannot be represented by u32")
            })?)
            .ok_or(GraphParquetError::Invalid(
                "mutable and frozen point counts overflow u32",
            ))
    }
}

/// Information about a completed graph Parquet write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphParquetWriteSummary {
    /// Number of adjacency rows written.
    pub rows: u64,
    /// CRC32 of canonical `(node_id, neighbors)` rows.
    pub checksum: u32,
}

/// A failure while encoding or decoding Vamana graph Parquet data.
#[derive(Debug, Error)]
pub enum GraphParquetError {
    /// Arrow rejected an array or record batch.
    #[error("invalid graph Arrow data")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Parquet I/O or decoding failed.
    #[error("graph Parquet I/O failed")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// The in-memory neighbor provider rejected a graph operation.
    #[error("in-memory graph operation failed")]
    Provider(#[from] ANNError),
    /// The logical schema, metadata, or rows are incompatible.
    #[error("invalid graph Parquet artifact: {0}")]
    Invalid(&'static str),
    /// A required metadata value is missing or malformed.
    #[error("invalid graph Parquet metadata {key:?}: {value:?}")]
    InvalidMetadata {
        /// Metadata key.
        key: &'static str,
        /// Supplied value, if present.
        value: Option<String>,
    },
}

impl SimpleNeighborProviderAsync {
    /// Streams every allocated adjacency list to Parquet in ascending node-ID order.
    ///
    /// # Errors
    ///
    /// Returns an error when `layout` does not describe this provider, an adjacency cannot be
    /// read, or Arrow/Parquet writing fails.
    pub async fn write_graph_parquet<W>(
        &self,
        writer: W,
        layout: &GraphParquetLayout,
    ) -> Result<GraphParquetWriteSummary, GraphParquetError>
    where
        W: AsyncFileWriter + Send + 'static,
    {
        validate_provider_layout(self, layout)?;
        let checksum = self.graph_checksum()?;
        let properties = WriterProperties::builder()
            .set_key_value_metadata(Some(graph_metadata(layout, checksum)?))
            .build();
        let schema = graph_schema();
        let mut writer = AsyncArrowWriter::try_new(writer, schema.clone(), Some(properties))?;
        for start in (0..self.total_points()).step_by(BATCH_SIZE) {
            let end = (start + BATCH_SIZE).min(self.total_points());
            let start_id = u32::try_from(start)
                .map_err(|_| GraphParquetError::Invalid("node ID exceeds u32"))?;
            let end_id = u32::try_from(end)
                .map_err(|_| GraphParquetError::Invalid("node ID exceeds u32"))?;
            let ids = UInt32Array::from_iter_values(start_id..end_id);
            let mut neighbors = ListBuilder::new(UInt32Builder::new())
                .with_field(Arc::new(Field::new("item", DataType::UInt32, false)));
            for node_id in start..end {
                let mut list = AdjacencyList::new();
                self.get_neighbors_sync(node_id, &mut list)?;
                neighbors.values().append_slice(&list);
                neighbors.append(true);
            }
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(ids) as ArrayRef, Arc::new(neighbors.finish())],
            )?;
            writer.write(&batch).await?;
        }
        writer.finish().await?;
        Ok(GraphParquetWriteSummary {
            rows: u64::from(layout.total_points()?),
            checksum,
        })
    }

    /// Imports a complete Parquet adjacency table into this preallocated provider.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible metadata or schema, noncanonical node IDs,
    /// out-of-range neighbors, excessive degree, start-point mismatch, checksum failure, or
    /// Parquet I/O failure.
    pub async fn read_graph_parquet<R>(
        &self,
        reader: R,
        expected: &GraphParquetLayout,
    ) -> Result<(), GraphParquetError>
    where
        R: AsyncFileReader + Unpin + Send + 'static,
    {
        validate_provider_layout(self, expected)?;
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
        if builder.schema().fields() != graph_schema().fields() {
            return Err(GraphParquetError::Invalid(
                "schema does not match vector.search-structure v1",
            ));
        }
        let metadata = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .cloned();
        validate_graph_metadata(metadata.as_ref(), expected)?;
        let expected_checksum = parse_checksum(metadata.as_ref())?;
        let mut stream = builder.build()?;
        let mut next_node_id = 0_u32;
        let total_points = expected.total_points()?;
        let mut checksum = crc32fast::Hasher::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(GraphParquetError::Invalid("node_id column is not UInt32"))?;
            let neighbors = batch.column(1).as_any().downcast_ref::<ListArray>().ok_or(
                GraphParquetError::Invalid("neighbors column is not List<UInt32>"),
            )?;
            if ids.null_count() != 0 || neighbors.null_count() != 0 {
                return Err(GraphParquetError::Invalid(
                    "graph rows contain null IDs or adjacency lists",
                ));
            }
            for row in 0..batch.num_rows() {
                let node_id = ids.value(row);
                if node_id != next_node_id {
                    return Err(GraphParquetError::Invalid(
                        "node IDs are not complete and strictly ordered",
                    ));
                }
                let values = neighbors.value(row);
                let values = values
                    .as_any()
                    .downcast_ref::<PrimitiveArray<UInt32Type>>()
                    .ok_or(GraphParquetError::Invalid("neighbor values are not UInt32"))?;
                if values.null_count() != 0 {
                    return Err(GraphParquetError::Invalid(
                        "adjacency list contains null neighbors",
                    ));
                }
                let list = values.values();
                if list.len() > expected.max_degree as usize {
                    return Err(GraphParquetError::Invalid(
                        "adjacency list exceeds configured maximum degree",
                    ));
                }
                if list.iter().any(|neighbor| *neighbor >= total_points) {
                    return Err(GraphParquetError::Invalid(
                        "adjacency list contains an out-of-range neighbor",
                    ));
                }
                checksum.update(&node_id.to_le_bytes());
                checksum.update(&(list.len() as u32).to_le_bytes());
                for neighbor in list {
                    checksum.update(&neighbor.to_le_bytes());
                }
                self.set_neighbors_sync(node_id as usize, list)?;
                next_node_id = next_node_id
                    .checked_add(1)
                    .ok_or(GraphParquetError::Invalid("node ID overflow"))?;
            }
        }
        if next_node_id != total_points {
            return Err(GraphParquetError::Invalid(
                "graph row count does not match total point count",
            ));
        }
        if checksum.finalize() != expected_checksum {
            return Err(GraphParquetError::Invalid(
                "logical checksum does not match graph rows",
            ));
        }
        Ok(())
    }

    fn graph_checksum(&self) -> Result<u32, GraphParquetError> {
        let mut checksum = crc32fast::Hasher::new();
        for node_id in 0..self.total_points() {
            let mut neighbors = AdjacencyList::new();
            self.get_neighbors_sync(node_id, &mut neighbors)?;
            checksum.update(&(node_id as u32).to_le_bytes());
            checksum.update(&(neighbors.len() as u32).to_le_bytes());
            for neighbor in neighbors.iter() {
                checksum.update(&neighbor.to_le_bytes());
            }
        }
        Ok(checksum.finalize())
    }
}

fn validate_provider_layout(
    provider: &SimpleNeighborProviderAsync,
    layout: &GraphParquetLayout,
) -> Result<(), GraphParquetError> {
    if usize::try_from(layout.total_points()?).ok() != Some(provider.total_points()) {
        return Err(GraphParquetError::Invalid(
            "layout point count does not match graph provider",
        ));
    }
    if layout.start_point_ids.len() != provider.num_start_points() {
        return Err(GraphParquetError::Invalid(
            "layout start-point count does not match graph provider",
        ));
    }
    if usize::try_from(layout.max_degree).ok() != Some(provider.max_degree()) {
        return Err(GraphParquetError::Invalid(
            "layout maximum degree does not match graph provider",
        ));
    }
    let total_points = layout.total_points()?;
    if layout
        .start_point_ids
        .iter()
        .any(|start| *start >= total_points)
    {
        return Err(GraphParquetError::Invalid(
            "layout contains an out-of-range start point",
        ));
    }
    Ok(())
}

fn graph_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::UInt32, false),
        Field::new(
            "neighbors",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
    ]))
}

fn graph_metadata(
    layout: &GraphParquetLayout,
    checksum: u32,
) -> Result<Vec<KeyValue>, GraphParquetError> {
    let total_points = layout.total_points()?;
    Ok([
        (ARTIFACT_TYPE_KEY, GRAPH_ARTIFACT_TYPE.to_owned()),
        (ENCODING_VERSION_KEY, GRAPH_ENCODING_VERSION.to_string()),
        (TOTAL_POINTS_KEY, total_points.to_string()),
        (MUTABLE_CAPACITY_KEY, layout.mutable_capacity.to_string()),
        (
            NUM_START_POINTS_KEY,
            layout.start_point_ids.len().to_string(),
        ),
        (
            START_POINT_IDS_KEY,
            format_start_points(&layout.start_point_ids),
        ),
        (MAX_DEGREE_KEY, layout.max_degree.to_string()),
        (CHECKSUM_KEY, format!("{checksum:08x}")),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue {
        key: key.to_owned(),
        value: Some(value),
    })
    .collect())
}

fn validate_graph_metadata(
    metadata: Option<&Vec<KeyValue>>,
    expected: &GraphParquetLayout,
) -> Result<(), GraphParquetError> {
    require_metadata(metadata, ARTIFACT_TYPE_KEY, GRAPH_ARTIFACT_TYPE)?;
    require_metadata(
        metadata,
        ENCODING_VERSION_KEY,
        &GRAPH_ENCODING_VERSION.to_string(),
    )?;
    require_metadata(
        metadata,
        TOTAL_POINTS_KEY,
        &expected.total_points()?.to_string(),
    )?;
    require_metadata(
        metadata,
        MUTABLE_CAPACITY_KEY,
        &expected.mutable_capacity.to_string(),
    )?;
    require_metadata(
        metadata,
        NUM_START_POINTS_KEY,
        &expected.start_point_ids.len().to_string(),
    )?;
    require_metadata(
        metadata,
        START_POINT_IDS_KEY,
        &format_start_points(&expected.start_point_ids),
    )?;
    require_metadata(metadata, MAX_DEGREE_KEY, &expected.max_degree.to_string())
}

fn format_start_points(start_points: &[u32]) -> String {
    start_points
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_checksum(metadata: Option<&Vec<KeyValue>>) -> Result<u32, GraphParquetError> {
    let value = metadata_value(metadata, CHECKSUM_KEY)?;
    value
        .and_then(|value| u32::from_str_radix(value, 16).ok())
        .ok_or_else(|| GraphParquetError::InvalidMetadata {
            key: CHECKSUM_KEY,
            value: value.map(str::to_owned),
        })
}

fn require_metadata(
    metadata: Option<&Vec<KeyValue>>,
    key: &'static str,
    expected: &str,
) -> Result<(), GraphParquetError> {
    let value = metadata_value(metadata, key)?;
    if value != Some(expected) {
        return Err(GraphParquetError::InvalidMetadata {
            key,
            value: value.map(str::to_owned),
        });
    }
    Ok(())
}

fn metadata_value<'a>(
    metadata: Option<&'a Vec<KeyValue>>,
    key: &'static str,
) -> Result<Option<&'a str>, GraphParquetError> {
    let mut values = metadata
        .into_iter()
        .flatten()
        .filter(|entry| entry.key == key);
    let value = values.next().and_then(|entry| entry.value.as_deref());
    if values.next().is_some() {
        return Err(GraphParquetError::InvalidMetadata {
            key,
            value: value.map(str::to_owned),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> SimpleNeighborProviderAsync {
        let provider = SimpleNeighborProviderAsync::new(4, 1, 3, 1.0);
        provider.set_neighbors_sync(0, &[1, 4]).expect("set node 0");
        provider.set_neighbors_sync(1, &[0, 2]).expect("set node 1");
        provider.set_neighbors_sync(2, &[1]).expect("set node 2");
        provider.set_neighbors_sync(3, &[]).expect("set node 3");
        provider
            .set_neighbors_sync(4, &[0, 2])
            .expect("set start node");
        provider
    }

    fn layout() -> GraphParquetLayout {
        GraphParquetLayout {
            mutable_capacity: 4,
            start_point_ids: vec![4],
            max_degree: 3,
        }
    }

    #[tokio::test]
    async fn graph_parquet_round_trip_preserves_adjacency_and_start_points() {
        let original = provider();
        let directory = tempfile::tempdir().expect("create temporary directory");
        let path = directory.path().join("graph.parquet");
        let output = tokio::fs::File::create(&path).await.expect("create output");
        let summary = original
            .write_graph_parquet(output, &layout())
            .await
            .expect("write graph artifact");
        assert_eq!(summary.rows, 5);

        let restored = SimpleNeighborProviderAsync::new(4, 1, 3, 1.0);
        let input = tokio::fs::File::open(&path).await.expect("open input");
        restored
            .read_graph_parquet(input, &layout())
            .await
            .expect("read graph artifact");
        for node_id in 0..5 {
            let mut expected = AdjacencyList::new();
            let mut actual = AdjacencyList::new();
            original
                .get_neighbors_sync(node_id, &mut expected)
                .expect("read original adjacency");
            restored
                .get_neighbors_sync(node_id, &mut actual)
                .expect("read restored adjacency");
            assert_eq!(&*actual, &*expected);
        }
    }

    #[tokio::test]
    async fn graph_parquet_rejects_start_point_mismatch() {
        let original = provider();
        let directory = tempfile::tempdir().expect("create temporary directory");
        let path = directory.path().join("graph.parquet");
        let output = tokio::fs::File::create(&path).await.expect("create output");
        original
            .write_graph_parquet(output, &layout())
            .await
            .expect("write graph artifact");

        let restored = SimpleNeighborProviderAsync::new(4, 1, 3, 1.0);
        let input = tokio::fs::File::open(&path).await.expect("open input");
        assert!(
            restored
                .read_graph_parquet(
                    input,
                    &GraphParquetLayout {
                        start_point_ids: vec![3],
                        ..layout()
                    }
                )
                .await
                .is_err()
        );
    }
}
