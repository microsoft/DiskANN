/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashSet, fs::File, io::Read, mem::size_of, num::NonZeroUsize};

use bytemuck::cast_slice;
use diskann::{ANNError, ANNResult};
use diskann_providers::{
    common::AlignedBoxWithSlice, storage::FileStorageProvider, utils::DatasetDto,
};
use diskann_utils::io::Metadata;
use num_cpus;

pub struct TruthSet {
    pub index_nodes: Vec<u32>,
    pub distances: Option<Vec<f32>>,
    pub index_num_points: usize,
    pub index_dimension: usize,
}

pub fn calculate_recall(
    num_queries: usize,
    gold_std: &[u32],
    gs_dist: &Option<Vec<f32>>,
    dim_gs: usize,
    our_results: &[u32],
    dim_or: u32,
    recall_at: u32,
) -> ANNResult<f64> {
    let mut total_recall: f64 = 0.0;
    let (mut gt, mut res): (HashSet<u32>, HashSet<u32>) = (HashSet::new(), HashSet::new());

    for i in 0..num_queries {
        gt.clear();
        res.clear();

        let gt_slice = &gold_std[dim_gs * i..];
        let res_slice = &our_results[dim_or as usize * i..];
        let mut tie_breaker = recall_at as usize;

        if gs_dist.is_some() {
            tie_breaker = (recall_at - 1) as usize;
            let gt_dist_vec = &gs_dist.as_ref().ok_or_else(|| {
                ANNError::log_ground_truth_error("Ground Truth file not loaded".to_string())
            })?[dim_gs * i..];
            while tie_breaker < dim_gs
                && gt_dist_vec[tie_breaker] == gt_dist_vec[(recall_at - 1) as usize]
            {
                tie_breaker += 1;
            }
        }

        (0..tie_breaker).for_each(|idx| {
            gt.insert(gt_slice[idx]);
        });

        (0..recall_at as usize).for_each(|idx| {
            res.insert(res_slice[idx]);
        });

        let mut cur_recall: u32 = 0;
        for v in gt.iter() {
            if res.contains(v) {
                cur_recall += 1;
            }
        }

        total_recall += cur_recall as f64;
    }

    Ok(total_recall / num_queries as f64 * (100.0 / recall_at as f64))
}

pub fn get_graph_num_frozen_points(graph_file: &str) -> ANNResult<NonZeroUsize> {
    let mut file = File::open(graph_file)?;
    let mut usize_buffer = [0; size_of::<usize>()];
    let mut u32_buffer = [0; size_of::<u32>()];

    file.read_exact(&mut usize_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut u32_buffer)?;
    file.read_exact(&mut usize_buffer)?;
    let file_frozen_pts = usize::from_le_bytes(usize_buffer);

    NonZeroUsize::new(file_frozen_pts).ok_or_else(|| {
        ANNError::log_index_config_error(
            "num_frozen_pts".to_string(),
            "num_frozen_pts is zero in saved file".to_string(),
        )
    })
}

#[inline]
pub fn load_truthset(bin_file: &str) -> ANNResult<TruthSet> {
    let mut file = File::open(bin_file)?;
    let actual_file_size = file.metadata()?.len() as usize;

    let metadata = Metadata::read(&mut file)?;
    let (npts, dim) = metadata.into_dims();

    tracing::info!("Metadata: #pts = {npts}, #dims = {dim}... ");

    let expected_file_size_with_dists: usize =
        2 * npts * dim * size_of::<u32>() + 2 * size_of::<u32>();
    let expected_file_size_just_ids: usize = npts * dim * size_of::<u32>() + 2 * size_of::<u32>();

    let truthset_type : i32 = match actual_file_size
    {
        // This is in the C++ code, but nothing is done in this case. Keeping it here for future reference just in case.
        // expected_file_size_just_ids => 2,
        x if x == expected_file_size_with_dists => 1,
        _ => return Err(ANNError::log_index_error(format_args!("Error. File size mismatch. File should have bin format, with npts followed by ngt
                                                        followed by npts*ngt ids and optionally followed by npts*ngt distance values; actual size: {}, expected: {} or {}",
                                                        actual_file_size,
                                                        expected_file_size_with_dists,
                                                        expected_file_size_just_ids)))
    };

    let mut ids: Vec<u32> = vec![0; npts * dim];
    let mut buffer = vec![0; npts * dim * size_of::<u32>()];
    file.read_exact(&mut buffer)?;
    ids.clone_from_slice(cast_slice::<u8, u32>(&buffer));

    if truthset_type == 1 {
        let mut dists: Vec<f32> = vec![0.0; npts * dim];
        let mut buffer = vec![0; npts * dim * size_of::<f32>()];
        file.read_exact(&mut buffer)?;
        dists.clone_from_slice(cast_slice::<u8, f32>(&buffer));

        return Ok(TruthSet {
            index_nodes: ids,
            distances: Some(dists),
            index_num_points: npts,
            index_dimension: dim,
        });
    }

    Ok(TruthSet {
        index_nodes: ids,
        distances: None,
        index_num_points: npts,
        index_dimension: dim,
    })
}

#[inline]
pub fn load_aligned_bin<T: Default + Copy + Sized + bytemuck::Pod>(
    bin_file: &str,
) -> ANNResult<(AlignedBoxWithSlice<T>, usize, usize, usize)> {
    let storage_provider = FileStorageProvider;
    diskann_providers::utils::load_aligned_bin(&storage_provider, bin_file)
}

#[inline]
pub fn load_aligned_from_vector<T: Default + Copy + Sized>(
    input_data: Vec<Vec<T>>,
) -> ANNResult<(AlignedBoxWithSlice<T>, usize, usize, usize)> {
    let t_size = size_of::<T>();
    let npts = input_data.len();
    let dim = input_data[0].len();

    let rounded_dim = dim.next_multiple_of(8);
    tracing::info!("Metadata: #pts = {npts}, #dims = {dim}, aligned_dim = {rounded_dim}...");
    let alloc_size = npts * rounded_dim;
    let alignment = 8 * t_size;

    tracing::info!(
        "allocating aligned memory of {} bytes... ",
        alloc_size * t_size
    );

    if !(alloc_size * t_size).is_multiple_of(alignment) {
        return Err(ANNError::log_index_error(format_args!(
            "Requested memory size is not a multiple of {}. Can not be allocated.",
            alignment
        )));
    }

    let mut data = AlignedBoxWithSlice::<T>::new(alloc_size, alignment)?;
    let dto = DatasetDto {
        data: &mut data,
        rounded_dim,
    };

    let (_, _) = copy_aligned_data_from_vector(input_data, dto, 0)?;

    Ok((data, npts, dim, rounded_dim))
}

#[inline]
pub fn copy_aligned_data_from_vector<T: Default + Copy>(
    input_data: Vec<Vec<T>>,
    dataset_dto: DatasetDto<T>,
    pts_offset: usize,
) -> std::io::Result<(usize, usize)> {
    let npts = input_data.len();
    let dim = input_data[0].len();
    let rounded_dim = dataset_dto.rounded_dim;
    let offset = pts_offset * rounded_dim;

    for (i, input) in input_data.iter().enumerate().take(npts) {
        let data_slice =
            &mut dataset_dto.data[offset + i * rounded_dim..offset + i * rounded_dim + dim];
        data_slice.copy_from_slice(input);

        let remaining = &mut dataset_dto.data
            [offset + i * rounded_dim + dim..offset + i * rounded_dim + rounded_dim];
        remaining.fill(T::default());
    }

    Ok((npts, dim))
}

pub fn get_num_threads(num_threads: Option<u32>) -> u32 {
    match num_threads {
        Some(0) => num_cpus::get() as u32,
        Some(x) => x,
        None => num_cpus::get() as u32,
    }
}
