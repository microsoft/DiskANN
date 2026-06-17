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
//! ### Unquantized (full precision)
//!
//! ```text
//! <index_dir>/
//!   ivf_meta.bin       — header (ndims, nlist, npoints, metric, quantized=0)
//!   ivf_centroids.bin  — row-major f32 centroids (nlist × ndims)
//!   clusters/
//!     cluster_0000.bin — [count:u32] then count × [id:u32][vec: ndims×f32]
//!     …
//! ```
//!
//! ### Quantized (MinMax)
//!
//! ```text
//! <index_dir>/
//!   ivf_meta.bin       — header (ndims, nlist, npoints, metric, quantized=1, nbits, grid_scale)
//!   ivf_centroids.bin  — row-major f32 centroids (nlist × ndims)
//!   vectors.bin        — flat blob of all f32 vectors in ID order (for reranking)
//!   clusters/
//!     cluster_0000.bin — [count:u32] then count × [id:u32][quantized_bytes]
//!     …
//! ```

use std::{fmt, fs, io::Write, num::NonZeroUsize, path::Path, time::Instant};

use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_quantization::{
    algorithms::kmeans::{lloyds::lloyds, plusplus::kmeans_plusplus_into},
    algorithms::transforms::NullTransform,
    bits::{Representation, Unsigned},
    minmax::{Data, DataMutRef, MinMaxQuantizer},
    num::Positive,
    CompressInto,
};
use diskann_utils::views::Matrix;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::{
    inputs::ivf::{IvfBuild, QuantizationConfig},
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

pub(super) fn build_ivf_index(params: &IvfBuild) -> anyhow::Result<IvfBuildStats> {
    let start = Instant::now();

    // Load the data
    let data: Matrix<f32> = datafiles::load_dataset(datafiles::BinFile(&params.data))?;
    let npoints = data.nrows();
    let ndims = data.ncols();
    let nlist = params.nlist.get() as usize;

    anyhow::ensure!(nlist <= npoints, "nlist ({nlist}) > npoints ({npoints})");

    // Run k-means (always uses SquaredL2 for clustering, regardless of search metric)
    let mut centroids = Matrix::new(0.0f32, nlist, ndims);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    kmeans_plusplus_into(centroids.as_mut_view(), data.as_view(), &mut rng)?;
    let (assignments, _residual): (Vec<u32>, f32) =
        lloyds(data.as_view(), centroids.as_mut_view(), params.kmeans_iterations.get() as usize);

    // Group vectors by cluster
    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nlist];
    for (i, &c) in assignments.iter().enumerate() {
        clusters[c as usize].push(i as u32);
    }

    // Write to disk
    let save_dir = Path::new(&params.save_path);
    fs::create_dir_all(save_dir)?;

    // 1) Meta file
    write_meta(save_dir, ndims, npoints, params)?;

    // 2) Centroids file: nlist × ndims f32 values
    {
        let bytes: &[u8] = bytemuck::cast_slice(centroids.as_slice());
        let mut f = fs::File::create(save_dir.join("ivf_centroids.bin"))?;
        f.write_all(bytes)?;
    }

    // 3) Cluster files + optional vectors.bin
    let clusters_dir = save_dir.join("clusters");
    fs::create_dir_all(&clusters_dir)?;

    match &params.quantization {
        None => write_clusters_full_precision(&clusters_dir, &clusters, &data)?,
        Some(qconfig) => {
            // Write vectors.bin for reranking
            write_vectors_blob(save_dir, &data)?;
            // Write quantized cluster files (dispatch on NBITS)
            match qconfig.nbits.as_usize() {
                1 => write_clusters_quantized::<1>(&clusters_dir, &clusters, &data, ndims, qconfig)?,
                4 => write_clusters_quantized::<4>(&clusters_dir, &clusters, &data, ndims, qconfig)?,
                8 => write_clusters_quantized::<8>(&clusters_dir, &clusters, &data, ndims, qconfig)?,
                _ => unreachable!("nbits validated to be 1, 4, or 8"),
            }
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

/// Write the meta file with quantization info.
///
/// Layout: `[ndims:u32][nlist:u32][npoints:u32][metric:u32][quantized:u8][nbits:u8][grid_scale:f32]`
fn write_meta(
    save_dir: &Path,
    ndims: usize,
    npoints: usize,
    params: &IvfBuild,
) -> anyhow::Result<()> {
    let mut f = fs::File::create(save_dir.join("ivf_meta.bin"))?;
    f.write_all(&(ndims as u32).to_le_bytes())?;
    f.write_all(&params.nlist.get().to_le_bytes())?;
    f.write_all(&(npoints as u32).to_le_bytes())?;
    f.write_all(&metric_to_u32(params.distance).to_le_bytes())?;

    match &params.quantization {
        None => {
            f.write_all(&[0u8])?; // quantized = false
        }
        Some(qconfig) => {
            f.write_all(&[1u8])?; // quantized = true
            f.write_all(&[qconfig.nbits.as_u8()])?;
            f.write_all(&qconfig.grid_scale.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Write full-precision cluster files.
fn write_clusters_full_precision(
    clusters_dir: &Path,
    clusters: &[Vec<u32>],
    data: &Matrix<f32>,
) -> anyhow::Result<()> {
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
    Ok(())
}

/// Write `vectors.bin`: flat blob of all f32 vectors in original ID order.
fn write_vectors_blob(save_dir: &Path, data: &Matrix<f32>) -> anyhow::Result<()> {
    let mut f = fs::File::create(save_dir.join("vectors.bin"))?;
    let bytes: &[u8] = bytemuck::cast_slice(data.as_slice());
    f.write_all(bytes)?;
    Ok(())
}

/// Write quantized cluster files for a given bit width.
///
/// Each cluster file: `[count:u32]` then `count × [id:u32][quantized_record]`
/// where `quantized_record` is `canonical_bytes(ndims)` bytes containing
/// MinMax compensation metadata + packed codes.
fn write_clusters_quantized<const NBITS: usize>(
    clusters_dir: &Path,
    clusters: &[Vec<u32>],
    data: &Matrix<f32>,
    ndims: usize,
    qconfig: &QuantizationConfig,
) -> anyhow::Result<()>
where
    Unsigned: Representation<NBITS>,
{
    let quantizer = MinMaxQuantizer::new(
        diskann_quantization::algorithms::transforms::Transform::Null(
            NullTransform::new(NonZeroUsize::new(ndims).unwrap()),
        ),
        Positive::new(qconfig.grid_scale)?,
    );

    let output_dim = quantizer.output_dim();
    let record_bytes = Data::<NBITS>::canonical_bytes(output_dim);
    let mut record_buf = vec![0u8; record_bytes];

    for (c_idx, cluster_ids) in clusters.iter().enumerate() {
        let filename = format!("cluster_{:04}.bin", c_idx);
        let mut f = fs::File::create(clusters_dir.join(filename))?;

        let count = cluster_ids.len() as u32;
        f.write_all(&count.to_le_bytes())?;

        for &vid in cluster_ids {
            f.write_all(&vid.to_le_bytes())?;

            // Compress the vector into the record buffer
            record_buf.fill(0);
            let data_mut = DataMutRef::<NBITS>::from_canonical_front_mut(
                &mut record_buf[..record_bytes],
                output_dim,
            )?;
            quantizer.compress_into(data.row(vid as usize), data_mut)?;
            f.write_all(&record_buf)?;
        }
    }

    Ok(())
}

