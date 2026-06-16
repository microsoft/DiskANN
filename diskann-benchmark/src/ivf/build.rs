/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF (Inverted File) index build phase.
//!
//! Builds a flat IVF index using Lloyd's k-means for centroid training, then assigns each
//! vector to its nearest centroid and writes the index to disk.
//!
//! ## On-disk layout
//!
//! ```text
//! <index_dir>/
//!   ivf_meta.bin       — 16-byte header (ndims, nlist, npoints, metric)
//!   ivf_centroids.bin  — row-major f32 centroids (nlist × ndims)
//!   clusters/
//!     cluster_0000.bin — records for cluster 0
//!     cluster_0001.bin — records for cluster 1
//!     …
//! ```
//!
//! Each cluster file uses an append-friendly record layout:
//!
//! ```text
//! [count: u32]                        ← number of vectors in this cluster
//! [id₀: u32][vec₀: ndims × f32]      ← record 0
//! [id₁: u32][vec₁: ndims × f32]      ← record 1
//! …
//! ```
//!
//! To append a new vector: seek to end, write `[id][vec]`, then update the count
//! at offset 0.

use std::{fmt, fs, io::Write, path::Path, time::Instant};

use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool};
use diskann_utils::views::Matrix;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    inputs::ivf::IvfBuild,
    utils::{datafiles, SimilarityMeasure},
};

fn metric_to_u32(m: SimilarityMeasure) -> u32 {
    match m {
        SimilarityMeasure::SquaredL2 => 0,
        SimilarityMeasure::InnerProduct => 1,
        SimilarityMeasure::Cosine => 2,
        SimilarityMeasure::CosineNormalized => 3,
    }
}

pub(super) fn u32_to_metric(v: u32) -> anyhow::Result<SimilarityMeasure> {
    match v {
        0 => Ok(SimilarityMeasure::SquaredL2),
        1 => Ok(SimilarityMeasure::InnerProduct),
        2 => Ok(SimilarityMeasure::Cosine),
        3 => Ok(SimilarityMeasure::CosineNormalized),
        _ => anyhow::bail!("unknown metric {}", v),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct IvfBuildStats {
    build_time: MicroSeconds,
    nlist: u32,
    npoints: u32,
    ndims: u32,
}

impl IvfBuildStats {
    pub(super) fn build_time_seconds(&self) -> f64 {
        self.build_time.as_seconds()
    }
}

impl fmt::Display for IvfBuildStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IVF Build: {:.3}s", self.build_time_seconds())?;
        writeln!(
            f,
            "  {} points, {} dims, {} clusters",
            self.npoints, self.ndims, self.nlist
        )
    }
}

/// Run Lloyd's k-means to compute centroids.
fn kmeans(
    data: &Matrix<f32>,
    k: usize,
    max_iters: u32,
    num_threads: usize,
    metric: SimilarityMeasure,
) -> Vec<f32> {
    let n = data.nrows();
    let d = data.ncols();
    let dist_fn = super::distance_fn(metric);

    // Initialize centroids with k-means++ seeding
    let mut centroids = vec![0.0f32; k * d];
    let mut rng = StdRng::seed_from_u64(42);

    // Pick first centroid uniformly at random
    let first = rng.random_range(0..n);
    centroids[..d].copy_from_slice(data.row(first));

    // Pick remaining centroids proportional to squared distance
    let mut min_dists = vec![f32::MAX; n];
    for c_idx in 1..k {
        // Update min_dists with the last added centroid
        let last_centroid = &centroids[(c_idx - 1) * d..c_idx * d];
        for (md, i) in min_dists.iter_mut().zip(0..n) {
            let dist = dist_fn(data.row(i), last_centroid);
            if dist < *md {
                *md = dist;
            }
        }

        let total: f64 = min_dists.iter().map(|&x| x as f64).sum();
        if total <= 0.0 {
            // Degenerate case: just pick random
            let idx = rng.random_range(0..n);
            centroids[c_idx * d..(c_idx + 1) * d].copy_from_slice(data.row(idx));
            continue;
        }

        let threshold = rng.random::<f64>() * total;
        let mut cumulative = 0.0f64;
        let mut chosen = n - 1;
        for (i, &dist) in min_dists.iter().enumerate() {
            cumulative += dist as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids[c_idx * d..(c_idx + 1) * d].copy_from_slice(data.row(chosen));
    }

    // Build thread pool
    let pool = create_thread_pool(num_threads).unwrap();

    let mut assignments = vec![0u32; n];

    for _iter in 0..max_iters {
        // Assignment step: each point → nearest centroid
        assignments
            .par_iter_mut()
            .enumerate()
            .for_each_in_pool(pool.as_ref(), |(i, assign)| {
                let point = data.row(i);
                let mut best_dist = f32::MAX;
                let mut best_c = 0u32;
                for c in 0..k {
                    let centroid = &centroids[c * d..(c + 1) * d];
                    let dist = dist_fn(point, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c as u32;
                    }
                }
                *assign = best_c;
            });

        // Update step: recompute centroids
        let mut new_centroids = vec![0.0f64; k * d];
        let mut counts = vec![0u64; k];

        for (i, &c) in assignments.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1;
            let point = data.row(i);
            for (j, &val) in point.iter().enumerate() {
                new_centroids[c * d + j] += val as f64;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for j in 0..d {
                    centroids[c * d + j] = (new_centroids[c * d + j] * inv) as f32;
                }
            }
        }
    }

    centroids
}

pub(super) fn build_ivf_index(params: &IvfBuild) -> anyhow::Result<IvfBuildStats> {
    let start = Instant::now();

    // Load the data
    let data: Matrix<f32> = datafiles::load_dataset(datafiles::BinFile(&params.data))?;
    let npoints = data.nrows();
    let ndims = data.ncols();
    let nlist = params.nlist.get() as usize;

    anyhow::ensure!(nlist <= npoints, "nlist ({nlist}) > npoints ({npoints})");

    // Run k-means
    let centroids = kmeans(
        &data,
        nlist,
        params.kmeans_iterations.get(),
        params.num_threads.get(),
        params.distance,
    );

    // Assign each vector to its nearest centroid
    let dist_fn = super::distance_fn(params.distance);
    let mut assignments = vec![0u32; npoints];
    for (assign, i) in assignments.iter_mut().zip(0..npoints) {
        let point = data.row(i);
        let mut best_dist = f32::MAX;
        let mut best_c = 0u32;
        for c in 0..nlist {
            let centroid = &centroids[c * ndims..(c + 1) * ndims];
            let dist = dist_fn(point, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_c = c as u32;
            }
        }
        *assign = best_c;
    }

    // Group vectors by cluster
    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nlist];
    for (i, &c) in assignments.iter().enumerate() {
        clusters[c as usize].push(i as u32);
    }

    // Write to disk
    let save_dir = Path::new(&params.save_path);
    fs::create_dir_all(save_dir)?;

    // 1) Meta file
    {
        let mut f = fs::File::create(save_dir.join("ivf_meta.bin"))?;
        f.write_all(&(ndims as u32).to_le_bytes())?;
        f.write_all(&params.nlist.get().to_le_bytes())?;
        f.write_all(&(npoints as u32).to_le_bytes())?;
        f.write_all(&metric_to_u32(params.distance).to_le_bytes())?;
    }

    // 2) Centroids file: nlist × ndims f32 values
    {
        let bytes: &[u8] = bytemuck::cast_slice(&centroids);
        let mut f = fs::File::create(save_dir.join("ivf_centroids.bin"))?;
        f.write_all(bytes)?;
    }

    // 3) One file per cluster (append-friendly record layout)
    //    Each file: [count: u32] then count × [id: u32][vec: ndims × f32]
    let clusters_dir = save_dir.join("clusters");
    fs::create_dir_all(&clusters_dir)?;

    for (c_idx, cluster_ids) in clusters.iter().enumerate() {
        let filename = format!("cluster_{:04}.bin", c_idx);
        let mut f = fs::File::create(clusters_dir.join(filename))?;

        let count = cluster_ids.len() as u32;
        f.write_all(&count.to_le_bytes())?;

        for &vid in cluster_ids {
            f.write_all(&vid.to_le_bytes())?;
            let row = data.row(vid as usize);
            let row_bytes: &[u8] = bytemuck::cast_slice(row);
            f.write_all(row_bytes)?;
        }
    }

    let build_time: MicroSeconds = start.elapsed().into();

    Ok(IvfBuildStats {
        build_time,
        nlist: params.nlist.get(),
        npoints: npoints as u32,
        ndims: ndims as u32,
    })
}
