/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::{BufReader, Read}, path::Path};

use anyhow::Context;
use bit_set::BitSet;
use diskann::utils::IntoUsize;
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_providers::storage::StorageReadProvider;
use diskann_quantization::multi_vector::{Mat, MatRef, Standard};
use diskann_utils::views::Matrix;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) struct BinFile<'a>(pub(crate) &'a Path);

/// Load a dataset or query set in `.bin` form from disk and return the result as a
/// row-major matrix.
#[inline(never)]
pub(crate) fn load_dataset<T>(path: BinFile<'_>) -> anyhow::Result<Matrix<T>>
where
    T: Copy + bytemuck::Pod,
{
    let (data, num_data, data_dim) = diskann_providers::utils::file_util::load_bin::<T, _>(
        &diskann_providers::storage::FileStorageProvider,
        &path.0.to_string_lossy(),
        0,
    )?;
    Ok(Matrix::try_from(data.into(), num_data, data_dim).map_err(|err| err.as_static())?)
}

/// Helper trait to load a `Matrix<Self>` from source files that potentially have a different
/// type.
pub(crate) trait ConvertingLoad: Sized {
    /// Return an error if the provided `data_type` cannot be loaded and converted to `Self`.
    fn check_converting_load(data_type: DataType) -> anyhow::Result<()>;

    /// Attempt to load the data at `path` as a `Matrix<Self>` assuming the on-disk
    /// representation has the encoding specified by `data_type`.
    ///
    /// If `data_type` is not compatible with `Self`, return an error.
    #[cfg(any(
        feature = "spherical-quantization",
        feature = "minmax-quantization",
        feature = "product-quantization"
    ))]
    fn converting_load(path: BinFile<'_>, data_type: DataType) -> anyhow::Result<Matrix<Self>>;
}

impl ConvertingLoad for f32 {
    fn check_converting_load(data_type: DataType) -> anyhow::Result<()> {
        let compatible = matches!(
            data_type,
            DataType::Float32 | DataType::Float16 | DataType::UInt8 | DataType::Int8
        );
        if compatible {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "data type {:?} is not supported for loading `f32` data",
                data_type
            ))
        }
    }

    #[inline(never)]
    #[cfg(any(
        feature = "spherical-quantization",
        feature = "minmax-quantization",
        feature = "product-quantization"
    ))]
    fn converting_load(path: BinFile<'_>, data_type: DataType) -> anyhow::Result<Matrix<f32>> {
        #[inline(never)]
        fn convert<T, U>(from: diskann_utils::views::MatrixView<T>) -> Matrix<U>
        where
            U: Default + Clone + From<T>,
            T: Copy,
        {
            let mut to = Matrix::new(U::default(), from.nrows(), from.ncols());
            std::iter::zip(to.as_mut_slice().iter_mut(), from.as_slice().iter())
                .for_each(|(t, f)| *t = (*f).into());
            to
        }
        match data_type {
            DataType::Float32 => load_dataset::<f32>(path),
            DataType::Float16 => Ok(convert(load_dataset::<half::f16>(path)?.as_view())),
            DataType::UInt8 => Ok(convert(load_dataset::<u8>(path)?.as_view())),
            DataType::Int8 => Ok(convert(load_dataset::<i8>(path)?.as_view())),
            _ => Err(anyhow::anyhow!(
                "data type {:?} is not supported for loading `f32` data",
                data_type
            )),
        }
    }
}

/// Load a groundtruth set from disk and return the  result as a row-major matrix.
pub(crate) fn load_groundtruth(path: BinFile<'_>) -> anyhow::Result<Matrix<u32>> {
    let provider = diskann_providers::storage::FileStorageProvider;
    let mut file = provider
        .open_reader(&path.0.to_string_lossy())
        .with_context(|| format!("while opening {}", path.0.display()))?;

    let (num_points, dim) = {
        let mut buffer = [0u8; std::mem::size_of::<u32>()];
        file.read_exact(&mut buffer)?;
        let num_points = u32::from_le_bytes(buffer).into_usize();

        file.read_exact(&mut buffer)?;
        let dim = u32::from_le_bytes(buffer).into_usize();
        (num_points, dim)
    };

    let mut groundtruth = Matrix::<u32>::new(0, num_points, dim);
    let groundtruth_slice: &mut [u8] = bytemuck::cast_slice_mut(groundtruth.as_mut_slice());
    file.read_exact(groundtruth_slice)?;
    Ok(groundtruth)
}

/// Load a range groundtruth set from disk
/// Range ground truth consists of a header with the number of points and
/// the total number of range results, then a `num_points` size array detailing
/// the number of results for each point, then the ground truth ids and distances
/// for all points in two contiguous arrays
/// We do not return groundtruth distances because there is no use for them in tie breaking
pub(crate) fn load_range_groundtruth(path: BinFile<'_>) -> anyhow::Result<Vec<Vec<u32>>> {
    let provider = diskann_providers::storage::FileStorageProvider;
    let mut file = provider
        .open_reader(&path.0.to_string_lossy())
        .with_context(|| format!("while opening {}", path.0.display()))?;

    let (num_points, total_results) = {
        let mut buffer = [0u8; std::mem::size_of::<u32>()];
        file.read_exact(&mut buffer)?;
        let num_points = u32::from_le_bytes(buffer).into_usize();

        file.read_exact(&mut buffer)?;
        let total_results = u32::from_le_bytes(buffer).into_usize();
        (num_points, total_results)
    };

    let mut sizes_and_ids: Vec<u32> = vec![0u32; num_points + total_results];
    let result_sizes_slice: &mut [u8] = bytemuck::cast_slice_mut(sizes_and_ids.as_mut_slice());
    file.read_exact(result_sizes_slice)?;

    let mut groundtruth_ids = Vec::<Vec<u32>>::with_capacity(num_points);
    let mut idx = 0;
    let sizes = &sizes_and_ids[..num_points];
    let ids = &sizes_and_ids[num_points..];
    for size in sizes {
        groundtruth_ids.push(ids[idx..idx + *size as usize].to_vec());
        idx += *size as usize;
    }
    Ok(groundtruth_ids)
}

// Helper struct for serializing BitSet as Vec<u8> (raw storage)
#[derive(Serialize, Deserialize)]
struct SerializableBitSet(Vec<u8>);

impl From<&BitSet> for SerializableBitSet {
    fn from(bs: &BitSet) -> Self {
        SerializableBitSet(bs.get_ref().to_bytes())
    }
}

impl From<SerializableBitSet> for BitSet {
    fn from(val: SerializableBitSet) -> Self {
        BitSet::from_bytes(&val.0)
    }
}

//////////////////
// Multi Vector //
//////////////////

/// Errors that can occur when loading multi-vectors from a binary file.
#[derive(Debug, Error)]
pub enum MultiVectorLoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("Unexpected end of file while reading {context}")]
    UnexpectedEof { context: &'static str },

    #[error("Invalid header: K={k}, D={d} (both must be > 0)")]
    InvalidHeader { k: u32, d: u32 },
}

/// Result of reading a single multi-vector: `Some((matrix, dimension))` or `None` for clean EOF.
type SingleMultiVectorResult<T> = Result<Option<(Mat<Standard<T>>, usize)>, MultiVectorLoadError>;

/// Read raw f16 multi-vector data from the reader into the buffer.
///
/// Returns `Ok(Some(mat_ref))` on success where `mat_ref` is a view over the buffer,
/// `Ok(None)` on clean EOF, or an error if the file is malformed.
fn read_multi_vector_raw<'a, R>(
    reader: &mut R,
    expected_dim: Option<usize>,
    buffer: &'a mut Vec<f16>,
) -> Result<Option<MatRef<'a, Standard<f16>>>, MultiVectorLoadError>
where
    R: Read,
{
    // Read first byte of K - EOF here means clean end of file
    let mut first_byte = [0u8; 1];
    match reader.read_exact(&mut first_byte) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(MultiVectorLoadError::Io(e)),
    }

    // Read remaining 3 bytes of K
    let mut k_rest = [0u8; 3];
    reader
        .read_exact(&mut k_rest)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::UnexpectedEof => MultiVectorLoadError::UnexpectedEof {
                context: "K header",
            },
            _ => MultiVectorLoadError::Io(e),
        })?;
    let k = u32::from_le_bytes([first_byte[0], k_rest[0], k_rest[1], k_rest[2]]);

    // Read D (dimension)
    let mut d_buf = [0u8; 4];
    reader
        .read_exact(&mut d_buf)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::UnexpectedEof => MultiVectorLoadError::UnexpectedEof {
                context: "D header",
            },
            _ => MultiVectorLoadError::Io(e),
        })?;
    let d = u32::from_le_bytes(d_buf);

    // Validate header
    if k == 0 || d == 0 {
        return Err(MultiVectorLoadError::InvalidHeader { k, d });
    }

    let k = k as usize;
    let d = d as usize;

    // Validate dimension consistency
    if let Some(expected) = expected_dim {
        if d != expected {
            return Err(MultiVectorLoadError::DimensionMismatch { expected, found: d });
        }
    }

    // Read K*D f16 values (reuse buffer, resizing as needed)
    let num_elements = k * d;
    buffer.resize(num_elements, f16::ZERO);
    let byte_buf: &mut [u8] = bytemuck::must_cast_slice_mut(buffer.as_mut_slice());
    reader.read_exact(byte_buf).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => MultiVectorLoadError::UnexpectedEof {
            context: "vector data",
        },
        _ => MultiVectorLoadError::Io(e),
    })?;

    // Return a view over the buffer
    let mat_ref = MatRef::new(Standard::new(k, d), buffer.as_slice())
        .expect("buffer size matches k * d");
    Ok(Some(mat_ref))
}

/// Read a single multi-vector from the reader, converting to type `T`.
///
/// This is a thin wrapper over `read_multi_vector_raw` that handles the type conversion.
/// Only the conversion loop is monomorphized per `T`.
fn read_single_multi_vector<T, R>(
    reader: &mut R,
    expected_dim: Option<usize>,
    buffer: &mut Vec<f16>,
) -> SingleMultiVectorResult<T>
where
    T: Copy + Default + From<f16>,
    R: Read,
{
    let Some(src) = read_multi_vector_raw(reader, expected_dim, buffer)? else {
        return Ok(None);
    };

    // Create Mat and populate with converted data
    let mut mat = Mat::new(
        Standard::new(src.num_vectors(), src.vector_dim()),
        T::default(),
    )
    .expect("valid matrix layout");

    for (dst_row, src_row) in mat.rows_mut().zip(src.rows()) {
        for (dst, &src_val) in dst_row.iter_mut().zip(src_row.iter()) {
            *dst = T::from(src_val);
        }
    }

    Ok(Some((mat, src.vector_dim())))
}

/// Load multi-vectors from a binary file.
///
/// Each multi-vector is encoded as:
/// - K (u32): number of vectors in this multi-vector
/// - D (u32): dimension of each vector
/// - K×D f16 values: vector data in row-major order
///
/// Returns a Vec of `Mat<Standard<T>>` where each Mat represents one multi-vector.
/// All multi-vectors must have the same dimension D, but may have different K values.
///
/// # Errors
///
/// - `MultiVectorLoadError::Io`: IO error reading the file
/// - `MultiVectorLoadError::DimensionMismatch`: D values differ between multi-vectors
/// - `MultiVectorLoadError::UnexpectedEof`: File ends mid-record
/// - `MultiVectorLoadError::InvalidHeader`: K or D is zero
pub fn load_multi_vectors<T>(path: impl AsRef<Path>) -> Result<Vec<Mat<Standard<T>>>, MultiVectorLoadError>
where
    T: Copy + Default + From<f16>,
{
    load_multi_vectors_inner::<T>(path.as_ref())
}

fn load_multi_vectors_inner<T>(path: &Path) -> Result<Vec<Mat<Standard<T>>>, MultiVectorLoadError>
where
    T: Copy + Default + From<f16>,
{
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut result = Vec::new();
    let mut expected_dim: Option<usize> = None;
    let mut buffer: Vec<f16> = Vec::new();

    while let Some((mat, dim)) = read_single_multi_vector(&mut reader, expected_dim, &mut buffer)? {
        expected_dim = Some(dim);
        result.push(mat);
    }

    Ok(result)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod multi_vector_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_multi_vector(file: &mut impl Write, k: u32, d: u32, values: &[f16]) {
        file.write_all(&k.to_le_bytes()).unwrap();
        file.write_all(&d.to_le_bytes()).unwrap();
        file.write_all(bytemuck::cast_slice(values)).unwrap();
    }

    #[test]
    fn test_empty_file() {
        let file = NamedTempFile::new().unwrap();
        let result: Vec<Mat<Standard<f32>>> = load_multi_vectors(file.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_multi_vector() {
        let mut file = NamedTempFile::new().unwrap();
        // K=2 vectors, D=3 dimensions
        let values: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(5.0),
            f16::from_f32(6.0),
        ];
        write_multi_vector(&mut file, 2, 3, &values);
        file.flush().unwrap();

        let result: Vec<Mat<Standard<f32>>> = load_multi_vectors(file.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_vectors(), 2);
        assert_eq!(result[0].vector_dim(), 3);

        let row0 = result[0].get_row(0).unwrap();
        assert_eq!(row0, &[1.0, 2.0, 3.0]);

        let row1 = result[0].get_row(1).unwrap();
        assert_eq!(row1, &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_multiple_multi_vectors_varying_k() {
        let mut file = NamedTempFile::new().unwrap();

        // First multi-vector: K=2, D=2
        let values1: Vec<f16> = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        write_multi_vector(&mut file, 2, 2, &values1);

        // Second multi-vector: K=3, D=2 (same D, different K)
        let values2: Vec<f16> = vec![
            f16::from_f32(5.0),
            f16::from_f32(6.0),
            f16::from_f32(7.0),
            f16::from_f32(8.0),
            f16::from_f32(9.0),
            f16::from_f32(10.0),
        ];
        write_multi_vector(&mut file, 3, 2, &values2);
        file.flush().unwrap();

        let result: Vec<Mat<Standard<f32>>> = load_multi_vectors(file.path()).unwrap();
        assert_eq!(result.len(), 2);

        assert_eq!(result[0].num_vectors(), 2);
        assert_eq!(result[0].vector_dim(), 2);

        assert_eq!(result[1].num_vectors(), 3);
        assert_eq!(result[1].vector_dim(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut file = NamedTempFile::new().unwrap();

        // First: K=1, D=2
        let values1: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
        write_multi_vector(&mut file, 1, 2, &values1);

        // Second: K=1, D=3 (different D - should fail)
        let values2: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(3.0)];
        write_multi_vector(&mut file, 1, 3, &values2);
        file.flush().unwrap();

        let result: Result<Vec<Mat<Standard<f32>>>, _> = load_multi_vectors(file.path());
        assert!(matches!(
            result,
            Err(MultiVectorLoadError::DimensionMismatch {
                expected: 2,
                found: 3
            })
        ));
    }

    #[test]
    fn test_truncated_file() {
        let mut file = NamedTempFile::new().unwrap();
        // Write only K, no D or data
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let result: Result<Vec<Mat<Standard<f32>>>, _> = load_multi_vectors(file.path());
        assert!(matches!(
            result,
            Err(MultiVectorLoadError::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn test_invalid_header_k_zero() {
        let mut file = NamedTempFile::new().unwrap();
        write_multi_vector(&mut file, 0, 3, &[]);
        file.flush().unwrap();

        let result: Result<Vec<Mat<Standard<f32>>>, _> = load_multi_vectors(file.path());
        assert!(matches!(
            result,
            Err(MultiVectorLoadError::InvalidHeader { k: 0, d: 3 })
        ));
    }

    #[test]
    fn test_invalid_header_d_zero() {
        let mut file = NamedTempFile::new().unwrap();
        write_multi_vector(&mut file, 2, 0, &[]);
        file.flush().unwrap();

        let result: Result<Vec<Mat<Standard<f32>>>, _> = load_multi_vectors(file.path());
        assert!(matches!(
            result,
            Err(MultiVectorLoadError::InvalidHeader { k: 2, d: 0 })
        ));
    }
}

