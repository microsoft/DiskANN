/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Parquet persistence for canonical spherical code slots.

use std::sync::Arc;

use arrow_array::builder::FixedSizeBinaryBuilder;
use arrow_array::{Array, ArrayRef, FixedSizeBinaryArray, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures_util::StreamExt;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

use super::spherical::{CodeImportError, SphericalStore};
use crate::model::graph::provider::async_::common::VectorStore;

/// The Parquet artifact type for canonical spherical code slots.
pub const CODES_ARTIFACT_TYPE: &str = "vector.codes";
/// The first version of the compressed-code Parquet schema.
pub const CODES_ENCODING_VERSION: u16 = 1;

const BATCH_SIZE: usize = 8192;
const ARTIFACT_TYPE_KEY: &str = "diskann.artifact-type";
const ENCODING_VERSION_KEY: &str = "diskann.encoding-version";
const NBITS_KEY: &str = "diskann.nbits";
const BYTES_PER_CODE_KEY: &str = "diskann.bytes-per-code";
const TOTAL_CAPACITY_KEY: &str = "diskann.total-capacity";
const MUTABLE_CAPACITY_KEY: &str = "diskann.mutable-capacity";
const FROZEN_POINTS_KEY: &str = "diskann.frozen-points";
const CHECKSUM_KEY: &str = "diskann.logical-checksum-crc32";

/// Layout metadata required to interpret a complete spherical code store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeParquetLayout {
    /// Number of bits used per transformed dimension.
    pub nbits: u8,
    /// Number of mutable code slots.
    pub mutable_capacity: u32,
    /// Number of frozen code slots following the mutable slots.
    pub frozen_points: u32,
}

impl CodeParquetLayout {
    /// Returns the total number of code slots.
    pub fn total_capacity(self) -> Result<u32, CodeParquetError> {
        self.mutable_capacity
            .checked_add(self.frozen_points)
            .ok_or(CodeParquetError::Invalid(
                "mutable and frozen capacities overflow u32",
            ))
    }
}

/// Information about a completed code Parquet write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeParquetWriteSummary {
    /// Number of code-slot rows written.
    pub rows: u64,
    /// CRC32 of canonical `(node_id, code)` rows.
    pub checksum: u32,
}

/// A failure while encoding or decoding compressed-code Parquet data.
#[derive(Debug, Error)]
pub enum CodeParquetError {
    /// Arrow rejected an array or record batch.
    #[error("invalid compressed-code Arrow data")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Parquet I/O or decoding failed.
    #[error("compressed-code Parquet I/O failed")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// The code-store import rejected a slot payload.
    #[error("unable to import compressed-code slot")]
    Import(#[from] CodeImportError),
    /// The logical schema, metadata, or rows are incompatible.
    #[error("invalid compressed-code Parquet artifact: {0}")]
    Invalid(&'static str),
    /// A required metadata value is missing or malformed.
    #[error("invalid compressed-code Parquet metadata {key:?}: {value:?}")]
    InvalidMetadata {
        /// Metadata key.
        key: &'static str,
        /// Supplied value, if present.
        value: Option<String>,
    },
}

impl SphericalStore {
    /// Streams every allocated spherical code slot to Parquet in ascending node-ID order.
    ///
    /// # Errors
    ///
    /// Returns an error when `layout` does not describe this store, a code width cannot be
    /// represented by Arrow, or Arrow/Parquet writing fails.
    pub async fn write_codes_parquet<W>(
        &self,
        writer: W,
        layout: CodeParquetLayout,
    ) -> Result<CodeParquetWriteSummary, CodeParquetError>
    where
        W: AsyncFileWriter + Send + 'static,
    {
        let total_capacity = layout.total_capacity()?;
        if usize::try_from(total_capacity).ok() != Some(self.total()) {
            return Err(CodeParquetError::Invalid(
                "layout capacity does not match spherical code store",
            ));
        }
        let code_width = i32::try_from(self.bytes()).map_err(|_| {
            CodeParquetError::Invalid("compressed-code width cannot be represented by Arrow")
        })?;
        let checksum = self.code_checksum();
        let properties = WriterProperties::builder()
            .set_key_value_metadata(Some(code_metadata(layout, self.bytes(), checksum)?))
            .build();
        let schema = code_schema(code_width);
        let mut writer = AsyncArrowWriter::try_new(writer, schema.clone(), Some(properties))?;
        for start in (0..self.total()).step_by(BATCH_SIZE) {
            let end = (start + BATCH_SIZE).min(self.total());
            let start_id = u32::try_from(start)
                .map_err(|_| CodeParquetError::Invalid("node ID exceeds u32"))?;
            let end_id =
                u32::try_from(end).map_err(|_| CodeParquetError::Invalid("node ID exceeds u32"))?;
            let mut codes = FixedSizeBinaryBuilder::with_capacity(end - start, code_width);
            for node_id in start..end {
                codes.append_value(self.code(node_id))?;
            }
            let ids = UInt32Array::from_iter_values(start_id..end_id);
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(ids) as ArrayRef, Arc::new(codes.finish())],
            )?;
            writer.write(&batch).await?;
        }
        writer.finish().await?;
        Ok(CodeParquetWriteSummary {
            rows: u64::from(total_capacity),
            checksum,
        })
    }

    /// Imports a complete Parquet code-slot table into this preallocated store.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible metadata or schema, noncanonical node IDs, code-width
    /// mismatch, checksum failure, capacity mismatch, or Parquet I/O failure.
    pub async fn read_codes_parquet<R>(
        &self,
        reader: R,
        expected: CodeParquetLayout,
    ) -> Result<(), CodeParquetError>
    where
        R: AsyncFileReader + Unpin + Send + 'static,
    {
        let total_capacity = expected.total_capacity()?;
        if usize::try_from(total_capacity).ok() != Some(self.total()) {
            return Err(CodeParquetError::Invalid(
                "layout capacity does not match spherical code store",
            ));
        }
        let code_width = i32::try_from(self.bytes()).map_err(|_| {
            CodeParquetError::Invalid("compressed-code width cannot be represented by Arrow")
        })?;
        let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
        if builder.schema().fields() != code_schema(code_width).fields() {
            return Err(CodeParquetError::Invalid(
                "schema does not match vector.codes v1",
            ));
        }
        let metadata = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .cloned();
        validate_code_metadata(metadata.as_ref(), expected, self.bytes())?;
        let expected_checksum = parse_checksum(metadata.as_ref())?;
        let mut stream = builder.build()?;
        let mut next_node_id = 0_u32;
        let mut checksum = crc32fast::Hasher::new();
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(CodeParquetError::Invalid("node_id column is not UInt32"))?;
            let codes = batch
                .column(1)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or(CodeParquetError::Invalid(
                    "code column is not FixedSizeBinary",
                ))?;
            if ids.null_count() != 0 || codes.null_count() != 0 {
                return Err(CodeParquetError::Invalid(
                    "compressed-code rows contain null values",
                ));
            }
            for row in 0..batch.num_rows() {
                let node_id = ids.value(row);
                if node_id != next_node_id {
                    return Err(CodeParquetError::Invalid(
                        "node IDs are not complete and strictly ordered",
                    ));
                }
                let code = codes.value(row);
                checksum.update(&node_id.to_le_bytes());
                checksum.update(code);
                self.import_codes_at(node_id as usize, code)?;
                next_node_id = next_node_id
                    .checked_add(1)
                    .ok_or(CodeParquetError::Invalid("node ID overflow"))?;
            }
        }
        if next_node_id != total_capacity {
            return Err(CodeParquetError::Invalid(
                "code row count does not match total capacity",
            ));
        }
        if checksum.finalize() != expected_checksum {
            return Err(CodeParquetError::Invalid(
                "logical checksum does not match compressed-code rows",
            ));
        }
        Ok(())
    }

    fn code_checksum(&self) -> u32 {
        let mut checksum = crc32fast::Hasher::new();
        for node_id in 0..self.total() {
            checksum.update(&(node_id as u32).to_le_bytes());
            checksum.update(self.code(node_id));
        }
        checksum.finalize()
    }
}

fn code_schema(code_width: i32) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("node_id", DataType::UInt32, false),
        Field::new("code", DataType::FixedSizeBinary(code_width), false),
    ]))
}

fn code_metadata(
    layout: CodeParquetLayout,
    bytes_per_code: usize,
    checksum: u32,
) -> Result<Vec<KeyValue>, CodeParquetError> {
    Ok([
        (ARTIFACT_TYPE_KEY, CODES_ARTIFACT_TYPE.to_owned()),
        (ENCODING_VERSION_KEY, CODES_ENCODING_VERSION.to_string()),
        (NBITS_KEY, layout.nbits.to_string()),
        (BYTES_PER_CODE_KEY, bytes_per_code.to_string()),
        (TOTAL_CAPACITY_KEY, layout.total_capacity()?.to_string()),
        (MUTABLE_CAPACITY_KEY, layout.mutable_capacity.to_string()),
        (FROZEN_POINTS_KEY, layout.frozen_points.to_string()),
        (CHECKSUM_KEY, format!("{checksum:08x}")),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue {
        key: key.to_owned(),
        value: Some(value),
    })
    .collect())
}

fn validate_code_metadata(
    metadata: Option<&Vec<KeyValue>>,
    expected: CodeParquetLayout,
    bytes_per_code: usize,
) -> Result<(), CodeParquetError> {
    require_metadata(metadata, ARTIFACT_TYPE_KEY, CODES_ARTIFACT_TYPE)?;
    require_metadata(
        metadata,
        ENCODING_VERSION_KEY,
        &CODES_ENCODING_VERSION.to_string(),
    )?;
    require_metadata(metadata, NBITS_KEY, &expected.nbits.to_string())?;
    require_metadata(metadata, BYTES_PER_CODE_KEY, &bytes_per_code.to_string())?;
    require_metadata(
        metadata,
        TOTAL_CAPACITY_KEY,
        &expected.total_capacity()?.to_string(),
    )?;
    require_metadata(
        metadata,
        MUTABLE_CAPACITY_KEY,
        &expected.mutable_capacity.to_string(),
    )?;
    require_metadata(
        metadata,
        FROZEN_POINTS_KEY,
        &expected.frozen_points.to_string(),
    )
}

fn parse_checksum(metadata: Option<&Vec<KeyValue>>) -> Result<u32, CodeParquetError> {
    let value = metadata_value(metadata, CHECKSUM_KEY)?;
    value
        .and_then(|value| u32::from_str_radix(value, 16).ok())
        .ok_or_else(|| CodeParquetError::InvalidMetadata {
            key: CHECKSUM_KEY,
            value: value.map(str::to_owned),
        })
}

fn require_metadata(
    metadata: Option<&Vec<KeyValue>>,
    key: &'static str,
    expected: &str,
) -> Result<(), CodeParquetError> {
    let value = metadata_value(metadata, key)?;
    if value != Some(expected) {
        return Err(CodeParquetError::InvalidMetadata {
            key,
            value: value.map(str::to_owned),
        });
    }
    Ok(())
}

fn metadata_value<'a>(
    metadata: Option<&'a Vec<KeyValue>>,
    key: &'static str,
) -> Result<Option<&'a str>, CodeParquetError> {
    let mut values = metadata
        .into_iter()
        .flatten()
        .filter(|entry| entry.key == key);
    let value = values.next().and_then(|entry| entry.value.as_deref());
    if values.next().is_some() {
        return Err(CodeParquetError::InvalidMetadata {
            key,
            value: value.map(str::to_owned),
        });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use diskann_quantization::algorithms::transforms::{TargetDim, TransformKind};
    use diskann_quantization::alloc::{GlobalAllocator, Poly};
    use diskann_quantization::spherical::iface::Impl;
    use diskann_quantization::spherical::{SphericalQuantizer, SupportedMetric};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    fn plan() -> Impl<2> {
        let centroid = Poly::from_iter([0.25, -0.5, 0.75, 1.0].into_iter(), GlobalAllocator)
            .expect("allocate centroid");
        let mut rng = StdRng::seed_from_u64(23);
        let quantizer = SphericalQuantizer::generate(
            centroid,
            1.5,
            TransformKind::DoubleHadamard {
                target_dim: TargetDim::Same,
            },
            SupportedMetric::SquaredL2,
            Some(2.0),
            &mut rng,
            GlobalAllocator,
        )
        .expect("create quantizer");
        Impl::new(quantizer).expect("create spherical plan")
    }

    #[tokio::test]
    async fn code_parquet_round_trip_preserves_every_allocated_slot() {
        let layout = CodeParquetLayout {
            nbits: 2,
            mutable_capacity: 4,
            frozen_points: 1,
        };
        let original = SphericalStore::new(plan(), 5, None);
        let mut codes = Vec::new();
        for node_id in 0..5_u8 {
            codes.extend(std::iter::repeat_n(
                node_id.wrapping_mul(17),
                original.bytes(),
            ));
        }
        original.import_codes(&codes).expect("populate code store");

        let directory = tempfile::tempdir().expect("create temporary directory");
        let path = directory.path().join("codes.parquet");
        let output = tokio::fs::File::create(&path).await.expect("create output");
        let summary = original
            .write_codes_parquet(output, layout)
            .await
            .expect("write code artifact");
        assert_eq!(summary.rows, 5);

        let restored = SphericalStore::new(plan(), 5, None);
        let input = tokio::fs::File::open(&path).await.expect("open input");
        restored
            .read_codes_parquet(input, layout)
            .await
            .expect("read code artifact");
        for node_id in 0..5 {
            assert_eq!(restored.code(node_id), original.code(node_id));
        }
    }

    #[tokio::test]
    async fn code_parquet_rejects_layout_mismatch() {
        let layout = CodeParquetLayout {
            nbits: 2,
            mutable_capacity: 4,
            frozen_points: 1,
        };
        let original = SphericalStore::new(plan(), 5, None);
        let directory = tempfile::tempdir().expect("create temporary directory");
        let path = directory.path().join("codes.parquet");
        let output = tokio::fs::File::create(&path).await.expect("create output");
        original
            .write_codes_parquet(output, layout)
            .await
            .expect("write code artifact");

        let restored = SphericalStore::new(plan(), 5, None);
        let input = tokio::fs::File::open(&path).await.expect("open input");
        assert!(
            restored
                .read_codes_parquet(input, CodeParquetLayout { nbits: 4, ..layout })
                .await
                .is_err()
        );
    }
}
