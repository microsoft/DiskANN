/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Arrow and Parquet persistence for spherical quantizer plans.

use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Float32Builder, ListBuilder, UInt32Builder};
use arrow_array::types::{Float32Type, UInt32Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, ListArray, RecordBatch, StringArray, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use flatbuffers::FlatBufferBuilder;
use futures_util::StreamExt;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use thiserror::Error;

use super::iface::{Constructible, Impl, Quantizer};
use crate::alloc::{AllocatorError, GlobalAllocator};
use crate::flatbuffers as fb;

/// The Parquet artifact type for a spherical quantizer plan.
pub const ARTIFACT_TYPE: &str = "vector.quantizer";
/// The first version of the quantizer Parquet schema.
pub const ENCODING_VERSION: u16 = 1;

const ARTIFACT_TYPE_KEY: &str = "diskann.artifact-type";
const ENCODING_VERSION_KEY: &str = "diskann.encoding-version";
const INPUT_DIMENSION_KEY: &str = "diskann.input-dimension";
const TRANSFORMED_DIMENSION_KEY: &str = "diskann.transformed-dimension";
const CHECKSUM_KEY: &str = "diskann.logical-checksum-crc32";
const TRANSFORM_KIND: &str = "double_hadamard";

/// Information about a completed quantizer Parquet write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteSummary {
    /// Number of logical rows written.
    pub rows: u64,
    /// CRC32 of the canonical logical quantizer state.
    pub checksum: u32,
}

/// A failure while encoding or decoding a quantizer Parquet artifact.
#[derive(Debug, Error)]
pub enum Error {
    /// Arrow rejected a schema, array, or record batch.
    #[error("invalid quantizer Arrow data")]
    Arrow(#[from] arrow_schema::ArrowError),
    /// Parquet I/O or decoding failed.
    #[error("quantizer Parquet I/O failed")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// The existing quantizer serializer could not allocate its temporary representation.
    #[error("unable to serialize spherical quantizer")]
    Allocator(#[from] AllocatorError),
    /// The logical schema or metadata is incompatible.
    #[error("invalid quantizer Parquet artifact: {0}")]
    Invalid(&'static str),
    /// The Arrow schema does not match the format contract.
    #[error("invalid quantizer Parquet schema: expected {expected}, found {actual}")]
    Schema {
        /// Expected schema.
        expected: String,
        /// Actual schema.
        actual: String,
    },
    /// A metadata value is missing or malformed.
    #[error("invalid quantizer Parquet metadata {key:?}: {value:?}")]
    InvalidMetadata {
        /// Metadata key.
        key: &'static str,
        /// Supplied value, if present.
        value: Option<String>,
    },
    /// Reconstructing the validated logical plan failed.
    #[error("unable to reconstruct spherical quantizer: {0}")]
    Deserialization(#[from] super::iface::DeserializationError),
}

#[derive(Debug)]
struct QuantizerRow {
    nbits: u8,
    metric: &'static str,
    centroid: Vec<f32>,
    mean_norm: f32,
    pre_scale: f32,
    signs0: Vec<bool>,
    signs1: Vec<bool>,
    subsample: Option<Vec<u32>>,
}

/// Writes one spherical quantizer plan as a logical one-row Parquet table.
///
/// # Errors
///
/// Returns an error if the plan does not use `DoubleHadamard`, its existing canonical
/// serializer fails, or Arrow/Parquet cannot write the artifact.
pub async fn write<W>(
    plan: &(dyn Quantizer + Send + Sync),
    writer: W,
) -> Result<WriteSummary, Error>
where
    W: AsyncFileWriter + Send + 'static,
{
    let serialized = plan.serialize(GlobalAllocator)?;
    let row = decode_flatbuffer(&serialized)?;
    let checksum = checksum(&row);
    let properties = WriterProperties::builder()
        .set_key_value_metadata(Some(metadata(&row, checksum)))
        .build();
    let schema = schema();
    let batch = to_batch(schema.clone(), &row)?;
    let mut writer = AsyncArrowWriter::try_new(writer, schema, Some(properties))?;
    writer.write(&batch).await?;
    writer.finish().await?;
    Ok(WriteSummary { rows: 1, checksum })
}

/// Reads one logical quantizer table and recreates the bit-width-specific plan.
///
/// # Errors
///
/// Returns an error for an incompatible schema or metadata, malformed logical state,
/// checksum failure, unsupported transform, or Parquet I/O failure.
pub async fn read<const NBITS: usize, R>(reader: R) -> Result<Impl<NBITS>, Error>
where
    R: AsyncFileReader + Unpin + Send + 'static,
    Impl<NBITS>: Constructible<GlobalAllocator>,
{
    let builder = ParquetRecordBatchStreamBuilder::new(reader).await?;
    if builder.schema().fields() != schema().fields() {
        return Err(Error::Schema {
            expected: format!("{:?}", schema()),
            actual: format!("{:?}", builder.schema()),
        });
    }
    let metadata = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .cloned();
    require_metadata(metadata.as_ref(), ARTIFACT_TYPE_KEY, ARTIFACT_TYPE)?;
    require_metadata(
        metadata.as_ref(),
        ENCODING_VERSION_KEY,
        &ENCODING_VERSION.to_string(),
    )?;

    let mut stream = builder.build()?;
    let mut row = None;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        for index in 0..batch.num_rows() {
            if row.is_some() {
                return Err(Error::Invalid(
                    "quantizer artifact must contain exactly one row",
                ));
            }
            row = Some(from_batch(&batch, index)?);
        }
    }
    let row = row.ok_or(Error::Invalid(
        "quantizer artifact must contain exactly one row",
    ))?;
    if usize::from(row.nbits) != NBITS {
        return Err(Error::Invalid(
            "quantizer bit width does not match requested plan",
        ));
    }
    require_metadata(
        metadata.as_ref(),
        INPUT_DIMENSION_KEY,
        &row.centroid.len().to_string(),
    )?;
    require_metadata(
        metadata.as_ref(),
        TRANSFORMED_DIMENSION_KEY,
        &transformed_dimension(&row).to_string(),
    )?;
    let expected_checksum = metadata_value(metadata.as_ref(), CHECKSUM_KEY)?
        .and_then(|value| u32::from_str_radix(value, 16).ok())
        .ok_or_else(|| Error::InvalidMetadata {
            key: CHECKSUM_KEY,
            value: metadata_value(metadata.as_ref(), CHECKSUM_KEY)
                .ok()
                .flatten()
                .map(str::to_owned),
        })?;
    if checksum(&row) != expected_checksum {
        return Err(Error::Invalid(
            "logical checksum does not match quantizer rows",
        ));
    }
    reconstruct::<NBITS>(&row)
}

fn schema() -> SchemaRef {
    let float_item = Arc::new(Field::new("item", DataType::Float32, false));
    let bool_item = Arc::new(Field::new("item", DataType::Boolean, false));
    let uint_item = Arc::new(Field::new("item", DataType::UInt32, false));
    Arc::new(Schema::new(vec![
        Field::new("nbits", DataType::UInt8, false),
        Field::new("metric", DataType::Utf8, false),
        Field::new("centroid", DataType::List(float_item), false),
        Field::new("mean_norm", DataType::Float32, false),
        Field::new("pre_scale", DataType::Float32, false),
        Field::new("transform_kind", DataType::Utf8, false),
        Field::new(
            "transform_signs_0",
            DataType::List(bool_item.clone()),
            false,
        ),
        Field::new("transform_signs_1", DataType::List(bool_item), false),
        Field::new("transform_subsample", DataType::List(uint_item), true),
    ]))
}

fn decode_flatbuffer(serialized: &[u8]) -> Result<QuantizerRow, Error> {
    if !fb::spherical::quantizer_buffer_has_identifier(serialized) {
        return Err(Error::Invalid(
            "canonical quantizer has an invalid identifier",
        ));
    }
    let root = fb::spherical::root_as_quantizer(serialized)
        .map_err(|_| Error::Invalid("canonical quantizer is malformed"))?;
    let quantizer = root.quantizer();
    let transform = quantizer
        .transform()
        .transform_as_double_hadamard()
        .ok_or(Error::Invalid(
            "only DoubleHadamard quantizers are supported",
        ))?;
    let nbits =
        u8::try_from(root.nbits()).map_err(|_| Error::Invalid("quantizer bit width exceeds u8"))?;
    let metric = match quantizer.metric() {
        fb::spherical::SupportedMetric::SquaredL2 => "l2",
        fb::spherical::SupportedMetric::InnerProduct => "inner_product",
        fb::spherical::SupportedMetric::Cosine => "cosine",
        _ => return Err(Error::Invalid("quantizer metric is unsupported")),
    };
    Ok(QuantizerRow {
        nbits,
        metric,
        centroid: quantizer.centroid().iter().collect(),
        mean_norm: quantizer.mean_norm(),
        pre_scale: quantizer.pre_scale(),
        signs0: transform.signs0().iter().collect(),
        signs1: transform.signs1().iter().collect(),
        subsample: transform
            .subsample()
            .map(|values| values.iter().collect::<Vec<_>>()),
    })
}

fn to_batch(schema: SchemaRef, row: &QuantizerRow) -> Result<RecordBatch, Error> {
    let mut centroid = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::Float32,
        false,
    )));
    centroid.values().append_slice(&row.centroid);
    centroid.append(true);

    let mut signs0 = ListBuilder::new(BooleanBuilder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::Boolean,
        false,
    )));
    for value in &row.signs0 {
        signs0.values().append_value(*value);
    }
    signs0.append(true);

    let mut signs1 = ListBuilder::new(BooleanBuilder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::Boolean,
        false,
    )));
    for value in &row.signs1 {
        signs1.values().append_value(*value);
    }
    signs1.append(true);

    let mut subsample = ListBuilder::new(UInt32Builder::new()).with_field(Arc::new(Field::new(
        "item",
        DataType::UInt32,
        false,
    )));
    if let Some(values) = &row.subsample {
        subsample.values().append_slice(values);
        subsample.append(true);
    } else {
        subsample.append(false);
    }

    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt8Array::from(vec![row.nbits])) as ArrayRef,
            Arc::new(StringArray::from(vec![row.metric])),
            Arc::new(centroid.finish()),
            Arc::new(Float32Array::from(vec![row.mean_norm])),
            Arc::new(Float32Array::from(vec![row.pre_scale])),
            Arc::new(StringArray::from(vec![TRANSFORM_KIND])),
            Arc::new(signs0.finish()),
            Arc::new(signs1.finish()),
            Arc::new(subsample.finish()),
        ],
    )?)
}

fn from_batch(batch: &RecordBatch, row: usize) -> Result<QuantizerRow, Error> {
    let nbits = primitive::<UInt8Array>(batch, 0)?.value(row);
    let metric = match string(batch, 1, row)? {
        "l2" => "l2",
        "inner_product" => "inner_product",
        "cosine" => "cosine",
        _ => return Err(Error::Invalid("quantizer metric is unsupported")),
    };
    let centroid = primitive_list::<Float32Type>(batch, 2, row)?;
    if centroid.is_empty() || centroid.iter().any(|value| !value.is_finite()) {
        return Err(Error::Invalid("quantizer centroid is empty or non-finite"));
    }
    let mean_norm = primitive::<Float32Array>(batch, 3)?.value(row);
    let pre_scale = primitive::<Float32Array>(batch, 4)?.value(row);
    if !mean_norm.is_finite() || mean_norm <= 0.0 || !pre_scale.is_finite() || pre_scale <= 0.0 {
        return Err(Error::Invalid(
            "quantizer norm and scale must be finite and positive",
        ));
    }
    if string(batch, 5, row)? != TRANSFORM_KIND {
        return Err(Error::Invalid("quantizer transform is unsupported"));
    }
    let signs0 = bool_list(batch, 6, row)?;
    let signs1 = bool_list(batch, 7, row)?;
    if signs0.is_empty() || signs1.len() < signs0.len() || signs0.len() != centroid.len() {
        return Err(Error::Invalid(
            "quantizer transform dimensions are inconsistent",
        ));
    }
    let subsample_array = list(batch, 8)?;
    let subsample = if subsample_array.is_null(row) {
        None
    } else {
        Some(primitive_list::<UInt32Type>(batch, 8, row)?)
    };
    if let Some(values) = &subsample
        && (values.is_empty()
            || !values.windows(2).all(|pair| pair[0] < pair[1])
            || values
                .last()
                .is_some_and(|last| *last as usize >= signs1.len()))
    {
        return Err(Error::Invalid("quantizer subsample is invalid"));
    }
    Ok(QuantizerRow {
        nbits,
        metric,
        centroid,
        mean_norm,
        pre_scale,
        signs0,
        signs1,
        subsample,
    })
}

fn reconstruct<const NBITS: usize>(row: &QuantizerRow) -> Result<Impl<NBITS>, Error>
where
    Impl<NBITS>: Constructible<GlobalAllocator>,
{
    let mut builder = FlatBufferBuilder::new();
    let centroid = builder.create_vector(&row.centroid);
    let signs0 = builder.create_vector(&row.signs0);
    let signs1 = builder.create_vector(&row.signs1);
    let subsample = row
        .subsample
        .as_ref()
        .map(|values| builder.create_vector(values));
    let double_hadamard = fb::transforms::DoubleHadamard::create(
        &mut builder,
        &fb::transforms::DoubleHadamardArgs {
            signs0: Some(signs0),
            signs1: Some(signs1),
            subsample,
        },
    );
    let transform = fb::transforms::Transform::create(
        &mut builder,
        &fb::transforms::TransformArgs {
            transform_type: fb::transforms::TransformKind::DoubleHadamard,
            transform: Some(double_hadamard.as_union_value()),
        },
    );
    let metric = match row.metric {
        "l2" => fb::spherical::SupportedMetric::SquaredL2,
        "inner_product" => fb::spherical::SupportedMetric::InnerProduct,
        "cosine" => fb::spherical::SupportedMetric::Cosine,
        _ => return Err(Error::Invalid("quantizer metric is unsupported")),
    };
    let quantizer = fb::spherical::SphericalQuantizer::create(
        &mut builder,
        &fb::spherical::SphericalQuantizerArgs {
            centroid: Some(centroid),
            transform: Some(transform),
            metric,
            mean_norm: row.mean_norm,
            pre_scale: row.pre_scale,
        },
    );
    let root = fb::spherical::Quantizer::create(
        &mut builder,
        &fb::spherical::QuantizerArgs {
            quantizer: Some(quantizer),
            nbits: u32::from(row.nbits),
        },
    );
    fb::spherical::finish_quantizer_buffer(&mut builder, root);
    Ok(Impl::<NBITS>::try_deserialize(
        builder.finished_data(),
        GlobalAllocator,
    )?)
}

fn primitive<T: Array + 'static>(batch: &RecordBatch, column: usize) -> Result<&T, Error> {
    batch
        .column(column)
        .as_any()
        .downcast_ref::<T>()
        .ok_or(Error::Invalid("quantizer column has an unexpected type"))
}

fn string(batch: &RecordBatch, column: usize, row: usize) -> Result<&str, Error> {
    let array = primitive::<StringArray>(batch, column)?;
    if array.is_null(row) {
        return Err(Error::Invalid("required quantizer value is null"));
    }
    Ok(array.value(row))
}

fn list(batch: &RecordBatch, column: usize) -> Result<&ListArray, Error> {
    primitive::<ListArray>(batch, column)
}

fn primitive_list<T>(
    batch: &RecordBatch,
    column: usize,
    row: usize,
) -> Result<Vec<T::Native>, Error>
where
    T: arrow_array::types::ArrowPrimitiveType,
{
    let list = list(batch, column)?;
    if list.is_null(row) {
        return Err(Error::Invalid("required quantizer list is null"));
    }
    let values = list.value(row);
    let values = values
        .as_any()
        .downcast_ref::<arrow_array::PrimitiveArray<T>>()
        .ok_or(Error::Invalid(
            "quantizer list values have an unexpected type",
        ))?;
    if values.null_count() != 0 {
        return Err(Error::Invalid("quantizer list contains null values"));
    }
    Ok(values.values().to_vec())
}

fn bool_list(batch: &RecordBatch, column: usize, row: usize) -> Result<Vec<bool>, Error> {
    let list = list(batch, column)?;
    if list.is_null(row) {
        return Err(Error::Invalid("required quantizer list is null"));
    }
    let values = list.value(row);
    let values = values
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or(Error::Invalid(
            "quantizer sign values have an unexpected type",
        ))?;
    if values.null_count() != 0 {
        return Err(Error::Invalid("quantizer sign list contains null values"));
    }
    Ok(values.iter().map(|value| value.unwrap_or(false)).collect())
}

fn transformed_dimension(row: &QuantizerRow) -> usize {
    row.subsample.as_ref().map_or(row.signs1.len(), Vec::len)
}

fn metadata(row: &QuantizerRow, checksum: u32) -> Vec<KeyValue> {
    [
        (ARTIFACT_TYPE_KEY, ARTIFACT_TYPE.to_owned()),
        (ENCODING_VERSION_KEY, ENCODING_VERSION.to_string()),
        (INPUT_DIMENSION_KEY, row.centroid.len().to_string()),
        (
            TRANSFORMED_DIMENSION_KEY,
            transformed_dimension(row).to_string(),
        ),
        (CHECKSUM_KEY, format!("{checksum:08x}")),
    ]
    .into_iter()
    .map(|(key, value)| KeyValue {
        key: key.to_owned(),
        value: Some(value),
    })
    .collect()
}

fn require_metadata(
    metadata: Option<&Vec<KeyValue>>,
    key: &'static str,
    expected: &str,
) -> Result<(), Error> {
    let value = metadata_value(metadata, key)?;
    if value != Some(expected) {
        return Err(Error::InvalidMetadata {
            key,
            value: value.map(str::to_owned),
        });
    }
    Ok(())
}

fn metadata_value<'a>(
    metadata: Option<&'a Vec<KeyValue>>,
    key: &'static str,
) -> Result<Option<&'a str>, Error> {
    let mut values = metadata
        .into_iter()
        .flatten()
        .filter(|entry| entry.key == key);
    let value = values.next().and_then(|entry| entry.value.as_deref());
    if values.next().is_some() {
        return Err(Error::InvalidMetadata {
            key,
            value: value.map(str::to_owned),
        });
    }
    Ok(value)
}

fn checksum(row: &QuantizerRow) -> u32 {
    let mut checksum = crc32fast::Hasher::new();
    checksum.update(&[row.nbits]);
    update_bytes(&mut checksum, row.metric.as_bytes());
    update_f32s(&mut checksum, &row.centroid);
    checksum.update(&row.mean_norm.to_le_bytes());
    checksum.update(&row.pre_scale.to_le_bytes());
    update_bytes(&mut checksum, TRANSFORM_KIND.as_bytes());
    update_bools(&mut checksum, &row.signs0);
    update_bools(&mut checksum, &row.signs1);
    match &row.subsample {
        Some(values) => {
            checksum.update(&[1]);
            checksum.update(&(values.len() as u64).to_le_bytes());
            for value in values {
                checksum.update(&value.to_le_bytes());
            }
        }
        None => checksum.update(&[0]),
    }
    checksum.finalize()
}

fn update_bytes(checksum: &mut crc32fast::Hasher, values: &[u8]) {
    checksum.update(&(values.len() as u64).to_le_bytes());
    checksum.update(values);
}

fn update_f32s(checksum: &mut crc32fast::Hasher, values: &[f32]) {
    checksum.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        checksum.update(&value.to_le_bytes());
    }
}

fn update_bools(checksum: &mut crc32fast::Hasher, values: &[bool]) {
    checksum.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        checksum.update(&[u8::from(*value)]);
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::algorithms::transforms::{TargetDim, TransformKind};
    use crate::alloc::Poly;
    use crate::spherical::{SphericalQuantizer, SupportedMetric};

    fn quantizer() -> SphericalQuantizer {
        let centroid = Poly::from_iter([0.25, -0.5, 0.75, 1.0].into_iter(), GlobalAllocator)
            .expect("allocate centroid");
        let mut rng = StdRng::seed_from_u64(17);
        SphericalQuantizer::generate(
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
        .expect("create quantizer")
    }

    macro_rules! quantizer_round_trip_test {
        ($name:ident, $bits:literal) => {
            #[tokio::test]
            async fn $name() {
                let directory = tempfile::tempdir().expect("create temporary directory");
                let path = directory.path().join("quantizer.parquet");
                let original = Impl::<$bits>::new(quantizer()).expect("create spherical plan");
                let output = tokio::fs::File::create(&path).await.expect("create output");
                let summary = write(&original, output).await.expect("write quantizer");
                assert_eq!(summary.rows, 1);

                let input = tokio::fs::File::open(&path).await.expect("open input");
                let restored = read::<$bits, _>(input).await.expect("read quantizer");
                let original_bytes =
                    Quantizer::serialize(&original, GlobalAllocator).expect("serialize original");
                let restored_bytes =
                    Quantizer::serialize(&restored, GlobalAllocator).expect("serialize restored");
                assert_eq!(&*original_bytes, &*restored_bytes);
            }
        };
    }

    quantizer_round_trip_test!(quantizer_parquet_round_trip_1_bit, 1);
    quantizer_round_trip_test!(quantizer_parquet_round_trip_2_bit, 2);
    quantizer_round_trip_test!(quantizer_parquet_round_trip_4_bit, 4);

    #[tokio::test]
    async fn quantizer_parquet_rejects_requested_bit_width_mismatch() {
        let directory = tempfile::tempdir().expect("create temporary directory");
        let path = directory.path().join("quantizer.parquet");
        let output = tokio::fs::File::create(&path).await.expect("create output");
        let plan = Impl::<2>::new(quantizer()).expect("create spherical plan");
        write(&plan, output).await.expect("write quantizer");

        let input = tokio::fs::File::open(&path).await.expect("open input");
        assert!(read::<4, _>(input).await.is_err());
    }
}
