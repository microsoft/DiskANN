/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! File loaders for the multi-vector rerank benchmark.
//!
//! - [`load_dataset`] reads a `.fbin` (header `u32 npoints, u32 ndims` + row-major data)
//!   via [`diskann_utils::io::read_bin`].
//! - [`load_groundtruth`] reads a variable-k groundtruth file (header
//!   `u32 nqueries, u32 total_results` + `nqueries` u32 per-query sizes + `total_results`
//!   flat u32 ids). One outer `Vec` entry per query; inner `Vec<u32>` is that query's
//!   relevant doc-ids. Format and parser ported from
//!   `diskann-benchmark/src/utils/datafiles.rs:126-155` (`load_range_groundtruth`).
//! - [`load_multi_vectors`] reads the per-record `.mvbin` format defined on
//!   `origin/users/mhildebr/multi-vector` (a stream of `u32 K | u32 D | K*D f16`
//!   records, no top-level header). Ported here to keep our crate self-contained.

use std::{
    io::{BufReader, Read},
    path::Path,
};

use anyhow::Context;
use diskann::utils::IntoUsize;
use diskann_providers::storage::StorageReadProvider;
use diskann_quantization::multi_vector::{Mat, MatRef, Standard};
use diskann_utils::views::Matrix;
use half::f16;
use thiserror::Error;

/// A thin newtype that marks a path as referring to a binary data file used by this
/// crate. Lets callers be explicit about file role at the call site.
pub(crate) struct BinFile<'a>(pub(crate) &'a Path);

/// Load a dataset or query set in `.fbin` form from disk and return the result as a
/// row-major matrix.
///
/// On-disk layout: `u32 npoints LE | u32 ndims LE | npoints * ndims * sizeof(T) bytes`.
#[inline(never)]
pub(crate) fn load_dataset<T>(path: BinFile<'_>) -> anyhow::Result<Matrix<T>>
where
    T: Copy + bytemuck::Pod,
{
    let data = diskann_utils::io::read_bin::<T>(
        &mut diskann_providers::storage::FileStorageProvider
            .open_reader(&path.0.to_string_lossy())?,
    )?;
    Ok(data)
}

/// Load a variable-k groundtruth set from disk.
///
/// On-disk layout (matches `diskann-benchmark`'s `load_range_groundtruth`):
/// ```text
/// u32 LE num_queries
/// u32 LE total_results          (= sum of per-query sizes)
/// num_queries * u32 LE sizes
/// total_results * u32 LE ids    (flat, in query order; no distance slab)
/// ```
///
/// Returns one `Vec<u32>` per query containing that query's relevant doc-ids. A query
/// with no positive judgments comes back as an empty inner vector.
pub(crate) fn load_groundtruth(path: BinFile<'_>) -> anyhow::Result<Vec<Vec<u32>>> {
    let provider = diskann_providers::storage::FileStorageProvider;
    let mut file = provider
        .open_reader(&path.0.to_string_lossy())
        .with_context(|| format!("while opening {}", path.0.display()))?;

    let mut buffer = [0u8; std::mem::size_of::<u32>()];
    file.read_exact(&mut buffer)?;
    let num_queries = u32::from_le_bytes(buffer).into_usize();
    file.read_exact(&mut buffer)?;
    let total_results = u32::from_le_bytes(buffer).into_usize();

    let mut sizes_and_ids: Vec<u32> = vec![0u32; num_queries + total_results];
    let sizes_and_ids_bytes: &mut [u8] = bytemuck::cast_slice_mut(sizes_and_ids.as_mut_slice());
    file.read_exact(sizes_and_ids_bytes)?;

    let (sizes, ids) = sizes_and_ids.split_at(num_queries);

    let mut groundtruth = Vec::<Vec<u32>>::with_capacity(num_queries);
    let mut cursor = 0usize;
    for size in sizes {
        let n = *size as usize;
        if cursor + n > ids.len() {
            anyhow::bail!(
                "groundtruth file is truncated: query #{} declares {} ids but only {} remain",
                groundtruth.len(),
                n,
                ids.len() - cursor,
            );
        }
        groundtruth.push(ids[cursor..cursor + n].to_vec());
        cursor += n;
    }
    if cursor != total_results {
        anyhow::bail!(
            "groundtruth sizes sum to {} but header declared total_results = {}",
            cursor,
            total_results,
        );
    }
    Ok(groundtruth)
}

//////////////////
// Multi Vector //
//////////////////

/// Errors that can occur when loading multi-vectors from a `.mvbin` file.
#[derive(Debug, Error)]
pub(crate) enum MultiVectorLoadError {
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
    // Read first byte of K — EOF here means clean end of file.
    let mut first_byte = [0u8; 1];
    match reader.read_exact(&mut first_byte) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(MultiVectorLoadError::Io(e)),
    }

    // Read remaining 3 bytes of K.
    let mut k_rest = [0u8; 3];
    reader.read_exact(&mut k_rest).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => MultiVectorLoadError::UnexpectedEof {
            context: "K header",
        },
        _ => MultiVectorLoadError::Io(e),
    })?;
    let k = u32::from_le_bytes([first_byte[0], k_rest[0], k_rest[1], k_rest[2]]);

    // Read D (dimension).
    let mut d_buf = [0u8; 4];
    reader.read_exact(&mut d_buf).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => MultiVectorLoadError::UnexpectedEof {
            context: "D header",
        },
        _ => MultiVectorLoadError::Io(e),
    })?;
    let d = u32::from_le_bytes(d_buf);

    if k == 0 || d == 0 {
        return Err(MultiVectorLoadError::InvalidHeader { k, d });
    }

    let k = k as usize;
    let d = d as usize;

    if let Some(expected) = expected_dim {
        if d != expected {
            return Err(MultiVectorLoadError::DimensionMismatch { expected, found: d });
        }
    }

    // Read K*D f16 values (reuse buffer, resizing as needed).
    let num_elements = k * d;
    buffer.resize(num_elements, f16::ZERO);
    let byte_buf: &mut [u8] = bytemuck::must_cast_slice_mut(buffer.as_mut_slice());
    reader.read_exact(byte_buf).map_err(|e| match e.kind() {
        std::io::ErrorKind::UnexpectedEof => MultiVectorLoadError::UnexpectedEof {
            context: "vector data",
        },
        _ => MultiVectorLoadError::Io(e),
    })?;

    let mat_ref = MatRef::new(Standard::new(k, d).unwrap(), buffer.as_slice())
        .expect("buffer size matches k * d");
    Ok(Some(mat_ref))
}

/// Read a single multi-vector from the reader, converting to type `T`.
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

    let mut mat = Mat::new(
        Standard::new(src.num_vectors(), src.vector_dim()).unwrap(),
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

/// Load multi-vectors from a `.mvbin` binary file.
///
/// File layout: a concatenation of records, each
/// - `u32 LE K` — number of sub-vectors in this multi-vector
/// - `u32 LE D` — dimension (must be consistent across the file)
/// - `K*D` f16 values in row-major order
///
/// Returns one [`Mat<Standard<T>>`] per record. The on-disk dtype is always f16; the
/// loader upcasts (or no-ops) into `T` via `T: From<f16>`.
pub(crate) fn load_multi_vectors<T>(
    path: &Path,
) -> Result<Vec<Mat<Standard<T>>>, MultiVectorLoadError>
where
    T: Copy + Default + From<f16>,
{
    load_multi_vectors_inner::<T>(path)
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
    fn empty_file_yields_empty_vec() {
        let file = NamedTempFile::new().unwrap();
        let result: Vec<Mat<Standard<f32>>> = load_multi_vectors::<f32>(file.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn single_record_roundtrips_and_upcasts_to_f32() {
        let mut file = NamedTempFile::new().unwrap();
        let values: Vec<f16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        write_multi_vector(&mut file, 2, 3, &values);
        file.flush().unwrap();

        let result: Vec<Mat<Standard<f32>>> = load_multi_vectors(file.path()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_vectors(), 2);
        assert_eq!(result[0].vector_dim(), 3);
        assert_eq!(result[0].get_row(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(result[0].get_row(1).unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn multiple_records_with_varying_k_same_d() {
        let mut file = NamedTempFile::new().unwrap();
        let values1: Vec<f16> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        write_multi_vector(&mut file, 2, 2, &values1);
        let values2: Vec<f16> = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        write_multi_vector(&mut file, 3, 2, &values2);
        file.flush().unwrap();

        let result: Vec<Mat<Standard<f32>>> = load_multi_vectors(file.path()).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].num_vectors(), 2);
        assert_eq!(result[1].num_vectors(), 3);
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let mut file = NamedTempFile::new().unwrap();
        write_multi_vector(&mut file, 1, 2, &[f16::ONE, f16::ONE]);
        write_multi_vector(&mut file, 1, 3, &[f16::ONE, f16::ONE, f16::ONE]);
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
    fn truncated_record_is_an_error() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let result: Result<Vec<Mat<Standard<f32>>>, _> = load_multi_vectors::<f32>(file.path());
        assert!(matches!(
            result,
            Err(MultiVectorLoadError::UnexpectedEof { .. })
        ));
    }
}

#[cfg(test)]
mod groundtruth_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_gt(file: &mut impl Write, sizes: &[u32], ids: &[u32]) {
        let total: u32 = sizes.iter().sum();
        assert_eq!(total as usize, ids.len(), "test fixture is inconsistent");
        file.write_all(&(sizes.len() as u32).to_le_bytes()).unwrap();
        file.write_all(&total.to_le_bytes()).unwrap();
        for s in sizes {
            file.write_all(&s.to_le_bytes()).unwrap();
        }
        for i in ids {
            file.write_all(&i.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn empty_gt_yields_zero_queries() {
        let mut file = NamedTempFile::new().unwrap();
        write_gt(&mut file, &[], &[]);
        file.flush().unwrap();

        let gt = load_groundtruth(BinFile(file.path())).unwrap();
        assert!(gt.is_empty());
    }

    #[test]
    fn fixed_one_per_query_roundtrips() {
        let mut file = NamedTempFile::new().unwrap();
        write_gt(&mut file, &[1, 1, 1], &[10, 20, 30]);
        file.flush().unwrap();

        let gt = load_groundtruth(BinFile(file.path())).unwrap();
        assert_eq!(gt, vec![vec![10], vec![20], vec![30]]);
    }

    #[test]
    fn variable_per_query_with_gap() {
        let mut file = NamedTempFile::new().unwrap();
        // q0: 2 relevant, q1: 0 relevant, q2: 3 relevant.
        write_gt(&mut file, &[2, 0, 3], &[10, 11, 20, 21, 22]);
        file.flush().unwrap();

        let gt = load_groundtruth(BinFile(file.path())).unwrap();
        assert_eq!(gt.len(), 3);
        assert_eq!(gt[0], vec![10, 11]);
        assert_eq!(gt[1], Vec::<u32>::new());
        assert_eq!(gt[2], vec![20, 21, 22]);
    }
}
