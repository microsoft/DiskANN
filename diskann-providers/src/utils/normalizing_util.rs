/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    io::{BufReader, BufWriter, Read, Write},
    sync::atomic::{AtomicBool, Ordering},
};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
use diskann::{ANNError, ANNResult};
use diskann_vector::{Norm, norm::FastL2Norm};
use rayon::prelude::*;
use tracing::info;

use super::{AsThreadPool, ParallelIteratorInPool, RayonThreadPool, read_metadata, write_metadata};
use crate::forward_threadpool;

/// The normalizing_utils derives from the DiskANN c++ utils.
pub fn normalize_data_file<StorageProvider, Pool>(
    in_file_name: &str,
    out_file_name: &str,
    storage_provider: &StorageProvider,
    pool: Pool,
) -> ANNResult<()>
where
    StorageProvider: StorageReadProvider + StorageWriteProvider,
    Pool: AsThreadPool,
{
    let mut reader = BufReader::new(storage_provider.open_reader(in_file_name)?);
    let mut writer = BufWriter::new(storage_provider.create_for_write(out_file_name)?);

    let metadata = read_metadata(&mut reader)?;
    let (npts, ndims) = (metadata.npoints, metadata.ndims);

    write_metadata(&mut writer, npts, ndims)?;

    info!("Normalizing FLOAT vectors in file: {}", in_file_name);
    info!("Dataset: #pts = {}, # dims = {}", npts, ndims);

    let blk_size = 131072;
    let nblks = npts.div_ceil(blk_size);
    info!("# blks: {}", nblks);

    forward_threadpool!(pool = pool);
    for i in 0..nblks {
        let cblk_size = std::cmp::min(npts - i * blk_size, blk_size);
        block_convert(&mut writer, &mut reader, cblk_size, ndims, pool)?;
    }

    info!("Wrote normalized points to file: {}", out_file_name);
    Ok(())
}

// Read block size buffer, convert original vector to normalized vector and write to file.
fn block_convert<W: Write, R: Read>(
    //writer: &mut BufWriter<FileStorageProvider::Writer>,
    writer: &mut W,
    reader: &mut R,
    cblk_size: usize,
    ndims: usize,
    pool: &RayonThreadPool,
) -> ANNResult<()> {
    let mut buffer = vec![0; ndims * cblk_size * std::mem::size_of::<f32>()];

    match reader.read_exact(&mut buffer) {
        Ok(()) => {
            let mut float_data = buffer
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<f32>>();
            normalize_data_internal(&mut float_data, ndims, pool)?;

            for &normalized_value in &float_data {
                writer.write_all(&normalized_value.to_le_bytes())?;
            }
        }
        Err(_err) => {
            return ANNResult::Err(ANNError::log_file_handle_error(
                "Error while reading vectors",
            ));
        }
    }
    Ok(())
}

pub fn normalize_data_internal_no_cblas<Pool: AsThreadPool>(
    data: &mut [f32],
    ndims: usize,
    pool: Pool,
) -> ANNResult<()> {
    let zero_norm: AtomicBool = AtomicBool::new(false);

    forward_threadpool!(pool = pool);
    data.par_chunks_mut(ndims).for_each_in_pool(pool, |chunk| {
        let norm_pt = chunk.iter().map(|val| val * val).sum::<f32>().sqrt();
        if norm_pt != 0.0 {
            chunk.iter_mut().for_each(|val| *val /= norm_pt);
        } else {
            zero_norm.store(true, Ordering::SeqCst);
        }
    });

    if zero_norm.load(Ordering::SeqCst) {
        return Err(ANNError::log_index_error(
            "Zero norm encountered. Unable to compute with zero-norm vector.",
        ));
    }

    Ok(())
}

pub fn normalize_data_internal<Pool: AsThreadPool>(
    data: &mut [f32],
    ndims: usize,
    pool: Pool,
) -> ANNResult<()> {
    let zero_norm = AtomicBool::new(false);

    forward_threadpool!(pool = pool);
    data.par_chunks_mut(ndims).for_each_in_pool(pool, |chunk| {
        let norm_pt: f32 = (FastL2Norm).evaluate(&*chunk);
        if norm_pt != 0.0 {
            chunk.iter_mut().for_each(|val| *val /= norm_pt);
        } else {
            zero_norm.store(true, Ordering::SeqCst);
        }
    });

    if zero_norm.load(Ordering::SeqCst) {
        return Err(ANNError::log_index_error(
            "Zero norm encountered. Unable to compute with zero-norm vector.",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod normalizing_utils_test {
    use crate::storage::{StorageReadProvider, VirtualStorageProvider};
    use vfs::{MemoryFS, OverlayFS, PhysicalFS};

    use super::*;
    use crate::utils::{create_thread_pool_for_test, storage_utils::*};

    #[test]
    fn test_normalize_data_file() {
        let in_file_name = "/test_data/sift/siftsmall_learn_256pts.fbin";
        let norm_file_name = "/test_data/sift/siftsmall_learn_256pts_normalized.fbin";
        let out_file_name = "/siftsmall_learn_256pts_normalized.fbin";

        let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let base_filesystem = PhysicalFS::new(workspace_root);
        let memory_filesystem = MemoryFS::new();
        let vfs = OverlayFS::new(&[memory_filesystem.into(), base_filesystem.into()]);
        let storage_provider = VirtualStorageProvider::new(vfs);
        let pool = create_thread_pool_for_test();
        normalize_data_file(in_file_name, out_file_name, &storage_provider, &pool).unwrap();

        let (load_data, load_num_pts, load_dims) =
            load_bin::<f32, _>(&mut storage_provider.open_reader(out_file_name).unwrap(), 0)
                .unwrap();
        storage_provider
            .delete(out_file_name)
            .expect("Should be able to delete temp file");

        let (norm_data, norm_num_pts, norm_dims) = load_bin::<f32, _>(
            &mut storage_provider.open_reader(norm_file_name).unwrap(),
            0,
        )
        .unwrap();

        assert_eq!(load_num_pts, norm_num_pts);
        assert_eq!(load_dims, norm_dims);
        assert_eq!(load_data, norm_data);
    }
}
