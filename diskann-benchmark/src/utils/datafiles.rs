/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Read, path::Path};

use anyhow::Context;
use bit_set::BitSet;
use diskann::utils::IntoUsize;
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_providers::storage::StorageReadProvider;
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

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
