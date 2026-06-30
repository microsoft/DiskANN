/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! On-disk layout for the inverted lists and the index metadata.
//!
//! Two files are written next to a user-supplied path prefix:
//!
//! * `<prefix>.graphivf_lists` — the inverted lists. For every cluster, in
//!   ascending cluster-id order, the bytes are `[ids: u32 x count][vectors:
//!   f32 x dim x count]`, packed back-to-back with no per-list padding. The
//!   whole file is zero-padded up to a 512-byte multiple so that sector-aligned
//!   reads never run past the end of the file.
//! * `<prefix>.graphivf_meta` — a compact header plus the per-cluster point
//!   counts. Byte offsets are recomputed from the counts on load.
//!
//! Because every list is variable length, a read for cluster `c` reads the
//! smallest 512-aligned byte window that fully contains the list and indexes
//! into it; this avoids wasting disk space on padding when lists are tiny (the
//! expected regime, with one centroid per ~10-20 points).

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::utils::VectorRepr;
use diskann_utils::views::MatrixView;

use crate::{
    params::{GraphParams, Metric},
    GraphIvfError, Result,
};

/// Sector alignment used for all disk reads (matches `A512`).
pub(crate) const ALIGN: u64 = 512;

/// Alignment of each cluster record's start within the list file. Keeping every
/// record start a multiple of 4 keeps the leading `u32` ids 4-byte aligned (and
/// the trailing vectors aligned for any element type of size <= 4) regardless of
/// the stored vector format. For `f32` lists this padding is always zero.
const RECORD_ALIGN: u64 = 4;

const MAGIC: u32 = 0x4756_4947; // "GIVF" little-endian
const VERSION: u32 = 1;

/// Bytes of a cluster's list actually occupied by its ids and vectors (no
/// trailing record padding): `count` u32 ids followed by `count * dim`
/// components of `element_size` bytes each.
fn used_bytes(count: usize, dim: usize, element_size: usize) -> u64 {
    (count * (4 + dim * element_size)) as u64
}

/// On-disk stride of a cluster's list: [`used_bytes`] rounded up to
/// [`RECORD_ALIGN`] so the next cluster starts aligned.
fn record_bytes(count: usize, dim: usize, element_size: usize) -> u64 {
    align_up(used_bytes(count, dim, element_size), RECORD_ALIGN)
}

fn align_down(value: u64, align: u64) -> u64 {
    value - (value % align)
}

fn align_up(value: u64, align: u64) -> u64 {
    align_down(value + align - 1, align)
}

/// Describes where every cluster lives in the list file and how the index was
/// built (so the centroid graph can be rebuilt on load).
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub dim: usize,
    pub metric: Metric,
    /// Size in bytes of one stored vector component (`size_of::<T>()`). Persisted
    /// so a load can sanity-check the requested element type against what was
    /// written. This is a size check, not a full type check (it does not
    /// distinguish equally sized types such as `i8` and `u8`).
    pub element_size: usize,
    pub num_points: u64,
    pub graph: GraphParams,
    /// Number of points in each cluster, indexed by cluster id.
    pub counts: Vec<u32>,
    /// Prefix-sum byte offsets into the list file; `offsets[c]` is the start of
    /// cluster `c` and `offsets[num_clusters]` is the total data length.
    pub offsets: Vec<u64>,
}

impl Layout {
    pub(crate) fn num_clusters(&self) -> usize {
        self.counts.len()
    }
}

/// The 512-aligned read window for a single cluster.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ClusterWindow {
    /// Sector-aligned start offset to read from.
    pub aligned_start: u64,
    /// Sector-aligned length to read (multiple of [`ALIGN`]).
    pub aligned_len: usize,
    /// Offset of the cluster's first byte within the read buffer.
    pub inner_offset: usize,
    /// Number of points in the cluster.
    pub count: usize,
}

fn compute_offsets(counts: &[u32], dim: usize, element_size: usize) -> Vec<u64> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut acc = 0u64;
    for &c in counts {
        offsets.push(acc);
        acc += record_bytes(c as usize, dim, element_size);
    }
    offsets.push(acc);
    offsets
}

/// Compute the sector-aligned read window for cluster `c`.
pub(crate) fn cluster_window(layout: &Layout, c: usize) -> ClusterWindow {
    let count = layout.counts[c] as usize;
    let start = layout.offsets[c];
    let len = used_bytes(count, layout.dim, layout.element_size);
    let aligned_start = align_down(start, ALIGN);
    let aligned_end = align_up(start + len, ALIGN);
    ClusterWindow {
        aligned_start,
        aligned_len: (aligned_end - aligned_start) as usize,
        inner_offset: (start - aligned_start) as usize,
        count,
    }
}

/// Borrow the ids and (flattened) vectors of a cluster out of a read buffer.
///
/// The returned vector slice has `count * dim` elements of type `T` in row-major
/// order. `T` must match the element type the lists were written with.
pub(crate) fn parse_cluster<'a, T: VectorRepr>(
    buf: &'a [u8],
    window: &ClusterWindow,
    dim: usize,
) -> (&'a [u32], &'a [T]) {
    let count = window.count;
    let ids_start = window.inner_offset;
    let ids_end = ids_start + count * 4;
    let ids: &[u32] = bytemuck::cast_slice(&buf[ids_start..ids_end]);
    let vec_end = ids_end + count * dim * std::mem::size_of::<T>();
    let vectors: &[T] = bytemuck::cast_slice(&buf[ids_end..vec_end]);
    (ids, vectors)
}

/// Write the inverted lists to `path` encoding vectors as `T`, returning the
/// per-cluster counts and the derived byte offsets.
///
/// `assignments[p]` is the centroid id that corpus point `p` was assigned to.
/// Input vectors are always `f32` and are encoded to `T` via
/// [`num_traits::FromPrimitive::from_f32`].
pub(crate) fn write_lists<T: VectorRepr>(
    path: &Path,
    data: MatrixView<'_, f32>,
    assignments: &[u32],
    num_clusters: usize,
) -> Result<(Vec<u32>, Vec<u64>)> {
    let dim = data.ncols();
    let elem_size = std::mem::size_of::<T>();

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
    for (pid, &c) in assignments.iter().enumerate() {
        buckets[c as usize].push(pid as u32);
    }
    let counts: Vec<u32> = buckets.iter().map(|b| b.len() as u32).collect();
    let offsets = compute_offsets(&counts, dim, elem_size);

    let mut writer = BufWriter::new(File::create(path)?);
    let mut written: u64 = 0;
    let mut encoded: Vec<T> = Vec::with_capacity(dim);
    for bucket in &buckets {
        writer.write_all(bytemuck::cast_slice(bucket))?;
        for &pid in bucket {
            encoded.clear();
            for &v in data.row(pid as usize) {
                encoded.push(T::from_f32(v).ok_or_else(|| {
                    GraphIvfError::invalid("corpus value not representable in target vector type")
                })?);
            }
            writer.write_all(bytemuck::cast_slice(&encoded))?;
        }
        // Pad the record up to RECORD_ALIGN so the next cluster starts aligned.
        let used = used_bytes(bucket.len(), dim, elem_size);
        let rec_pad = (align_up(used, RECORD_ALIGN) - used) as usize;
        if rec_pad > 0 {
            writer.write_all(&[0u8; RECORD_ALIGN as usize][..rec_pad])?;
        }
        written += record_bytes(bucket.len(), dim, elem_size);
    }

    let pad = (align_up(written, ALIGN) - written) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
    }
    writer.flush()?;

    Ok((counts, offsets))
}

/// Like [`write_lists`], but the vectors are already stored in the target
/// representation `T` and are copied verbatim instead of being encoded from
/// `f32`.
///
/// Each row of `data` is one stored vector of `data.ncols()` `T` elements (for
/// whole-vector quantized formats this is the canonical width, e.g. 404 for
/// 8-bit MinMax at dimension 384, not the logical dimension). This is the
/// counterpart used when the corpus is supplied pre-compressed.
pub(crate) fn write_lists_stored<T: VectorRepr>(
    path: &Path,
    data: MatrixView<'_, T>,
    assignments: &[u32],
    num_clusters: usize,
) -> Result<(Vec<u32>, Vec<u64>)> {
    let dim = data.ncols();
    let elem_size = std::mem::size_of::<T>();

    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
    for (pid, &c) in assignments.iter().enumerate() {
        buckets[c as usize].push(pid as u32);
    }
    let counts: Vec<u32> = buckets.iter().map(|b| b.len() as u32).collect();
    let offsets = compute_offsets(&counts, dim, elem_size);

    let mut writer = BufWriter::new(File::create(path)?);
    let mut written: u64 = 0;
    for bucket in &buckets {
        writer.write_all(bytemuck::cast_slice(bucket))?;
        for &pid in bucket {
            writer.write_all(bytemuck::cast_slice(data.row(pid as usize)))?;
        }
        // Pad the record up to RECORD_ALIGN so the next cluster starts aligned.
        let used = used_bytes(bucket.len(), dim, elem_size);
        let rec_pad = (align_up(used, RECORD_ALIGN) - used) as usize;
        if rec_pad > 0 {
            writer.write_all(&[0u8; RECORD_ALIGN as usize][..rec_pad])?;
        }
        written += record_bytes(bucket.len(), dim, elem_size);
    }

    let pad = (align_up(written, ALIGN) - written) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
    }
    writer.flush()?;

    Ok((counts, offsets))
}

/// Serialize the index metadata to `path`.
pub(crate) fn write_metadata(path: &Path, layout: &Layout) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_u32::<LittleEndian>(MAGIC)?;
    w.write_u32::<LittleEndian>(VERSION)?;
    w.write_u32::<LittleEndian>(layout.metric.as_u8() as u32)?;
    w.write_u32::<LittleEndian>(layout.element_size as u32)?;
    w.write_u32::<LittleEndian>(layout.dim as u32)?;
    w.write_u64::<LittleEndian>(layout.num_points)?;
    w.write_u64::<LittleEndian>(layout.num_clusters() as u64)?;
    w.write_u32::<LittleEndian>(layout.graph.degree as u32)?;
    w.write_u32::<LittleEndian>(layout.graph.l_build as u32)?;
    w.write_f32::<LittleEndian>(layout.graph.slack)?;
    w.write_f32::<LittleEndian>(layout.graph.alpha)?;
    w.write_all(bytemuck::cast_slice(&layout.counts))?;
    w.flush()?;
    Ok(())
}

/// Read the index metadata from `path` and reconstruct the [`Layout`].
pub(crate) fn read_metadata(path: &Path) -> Result<Layout> {
    let mut r = BufReader::new(File::open(path)?);
    let magic = r.read_u32::<LittleEndian>()?;
    if magic != MAGIC {
        return Err(GraphIvfError::malformed("bad metadata magic"));
    }
    let version = r.read_u32::<LittleEndian>()?;
    if version != VERSION {
        return Err(GraphIvfError::malformed(format!(
            "unsupported metadata version {version}"
        )));
    }
    let metric = Metric::from_u8(r.read_u32::<LittleEndian>()? as u8)
        .ok_or_else(|| GraphIvfError::malformed("unknown metric"))?;
    let element_size = r.read_u32::<LittleEndian>()? as usize;
    let dim = r.read_u32::<LittleEndian>()? as usize;
    let num_points = r.read_u64::<LittleEndian>()?;
    let num_clusters = r.read_u64::<LittleEndian>()? as usize;
    let graph = GraphParams {
        degree: r.read_u32::<LittleEndian>()? as usize,
        l_build: r.read_u32::<LittleEndian>()? as usize,
        slack: r.read_f32::<LittleEndian>()?,
        alpha: r.read_f32::<LittleEndian>()?,
    };

    let mut counts = vec![0u32; num_clusters];
    r.read_exact(bytemuck::cast_slice_mut(&mut counts))?;
    let offsets = compute_offsets(&counts, dim, element_size);

    Ok(Layout {
        dim,
        metric,
        element_size,
        num_points,
        graph,
        counts,
        offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_utils::views::Matrix;
    use diskann_vector::Half;
    use std::fs;

    const F32_SZ: usize = 4;
    const F16_SZ: usize = 2;

    #[test]
    fn record_bytes_counts_ids_and_vectors() {
        // 3 points x dim 4 (f32): 3 u32 ids + 3*4 f32 = 12 + 48 = 60 bytes.
        assert_eq!(record_bytes(3, 4, F32_SZ), 60);
        assert_eq!(record_bytes(0, 4, F32_SZ), 0);
        assert_eq!(record_bytes(1, 1, F32_SZ), 8);
    }

    #[test]
    fn record_bytes_pads_f16_to_four() {
        // 1 point x dim 3 (f16): 4 (id) + 3*2 = 10 used, padded up to 12.
        assert_eq!(used_bytes(1, 3, F16_SZ), 10);
        assert_eq!(record_bytes(1, 3, F16_SZ), 12);
        // Even used length needs no padding.
        assert_eq!(used_bytes(1, 4, F16_SZ), 12);
        assert_eq!(record_bytes(1, 4, F16_SZ), 12);
        // Every record start stays a multiple of 4.
        for count in 0..8 {
            for dim in 1..8 {
                assert_eq!(record_bytes(count, dim, F16_SZ) % RECORD_ALIGN, 0);
            }
        }
    }

    #[test]
    fn align_helpers_round_to_sector() {
        assert_eq!(align_down(0, ALIGN), 0);
        assert_eq!(align_down(511, ALIGN), 0);
        assert_eq!(align_down(512, ALIGN), 512);
        assert_eq!(align_down(1025, ALIGN), 1024);

        assert_eq!(align_up(0, ALIGN), 0);
        assert_eq!(align_up(1, ALIGN), 512);
        assert_eq!(align_up(512, ALIGN), 512);
        assert_eq!(align_up(513, ALIGN), 1024);
    }

    #[test]
    fn compute_offsets_is_prefix_sum() {
        let counts = [2u32, 0, 3];
        let dim = 4;
        let offsets = compute_offsets(&counts, dim, F32_SZ);
        // offsets has num_clusters + 1 entries.
        assert_eq!(offsets.len(), 4);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], record_bytes(2, dim, F32_SZ));
        // Empty cluster does not advance the offset.
        assert_eq!(offsets[2], record_bytes(2, dim, F32_SZ));
        assert_eq!(
            offsets[3],
            record_bytes(2, dim, F32_SZ) + record_bytes(3, dim, F32_SZ)
        );
    }

    #[test]
    fn cluster_window_is_sector_aligned_and_contains_list() {
        let dim = 4;
        let counts = vec![2u32, 100, 1];
        let offsets = compute_offsets(&counts, dim, F32_SZ);
        let layout = Layout {
            dim,
            metric: Metric::L2,
            element_size: F32_SZ,
            num_points: 103,
            graph: GraphParams::default(),
            counts: counts.clone(),
            offsets,
        };

        for (c, &cnt) in counts.iter().enumerate() {
            let w = cluster_window(&layout, c);
            // Window start and length are sector-aligned.
            assert_eq!(w.aligned_start % ALIGN, 0);
            assert_eq!(w.aligned_len as u64 % ALIGN, 0);
            assert_eq!(w.count, cnt as usize);

            let start = layout.offsets[c];
            let len = used_bytes(cnt as usize, dim, F32_SZ);
            // The aligned window fully contains the cluster's bytes.
            assert!(w.aligned_start <= start);
            assert_eq!(w.inner_offset as u64, start - w.aligned_start);
            assert!(w.aligned_start + w.aligned_len as u64 >= start + len);
        }
    }

    #[test]
    fn cluster_window_handles_empty_cluster() {
        let dim = 8;
        let counts = vec![0u32];
        let offsets = compute_offsets(&counts, dim, F32_SZ);
        let layout = Layout {
            dim,
            metric: Metric::L2,
            element_size: F32_SZ,
            num_points: 0,
            graph: GraphParams::default(),
            counts,
            offsets,
        };
        let w = cluster_window(&layout, 0);
        assert_eq!(w.count, 0);
        assert_eq!(w.aligned_len, 0);
        assert_eq!(w.inner_offset, 0);
    }

    /// Build a tiny corpus and round-trip it through `write_lists` +
    /// `parse_cluster`, reading the on-disk bytes back exactly as the searcher
    /// would (smallest enclosing sector-aligned window per cluster).
    #[test]
    fn write_then_parse_round_trips_lists() {
        let dim = 3;
        let num_points = 5;
        // Rows are easy to recognize: point p has all components == p.
        let mut raw = vec![0.0f32; num_points * dim];
        for p in 0..num_points {
            for d in 0..dim {
                raw[p * dim + d] = p as f32;
            }
        }
        let matrix = Matrix::try_from(raw.into_boxed_slice(), num_points, dim).unwrap();

        // Cluster 0: points 0, 3; cluster 1: empty; cluster 2: points 1, 2, 4.
        let assignments = [0u32, 2, 2, 0, 2];
        let num_clusters = 3;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lists.bin");
        let (counts, offsets) =
            write_lists::<f32>(&path, matrix.as_view(), &assignments, num_clusters).unwrap();
        assert_eq!(counts, vec![2, 0, 3]);

        let layout = Layout {
            dim,
            metric: Metric::L2,
            element_size: F32_SZ,
            num_points: num_points as u64,
            graph: GraphParams::default(),
            counts,
            offsets,
        };

        // File is padded to a sector multiple.
        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64 % ALIGN, 0);

        // Expected membership per cluster.
        let expected: [Vec<u32>; 3] = [vec![0, 3], vec![], vec![1, 2, 4]];
        for (c, want) in expected.iter().enumerate() {
            let w = cluster_window(&layout, c);
            if w.count == 0 {
                assert!(want.is_empty());
                continue;
            }
            let slice = &bytes[w.aligned_start as usize..w.aligned_start as usize + w.aligned_len];
            let (ids, vectors) = parse_cluster::<f32>(slice, &w, dim);
            assert_eq!(ids, want.as_slice());
            // Each stored vector equals its point id broadcast across dims.
            for (vec, &id) in vectors.chunks_exact(dim).zip(ids.iter()) {
                assert!(vec.iter().all(|&x| x == id as f32));
            }
        }
    }

    /// Same round-trip for f16 lists, where odd-length records exercise the
    /// 4-byte record padding and the ids must stay 4-byte aligned.
    #[test]
    fn write_then_parse_round_trips_f16_lists() {
        let dim = 3; // odd dim => 10-byte records, padded to 12.
        let num_points = 5;
        let mut raw = vec![0.0f32; num_points * dim];
        for p in 0..num_points {
            for d in 0..dim {
                raw[p * dim + d] = p as f32;
            }
        }
        let matrix = Matrix::try_from(raw.into_boxed_slice(), num_points, dim).unwrap();
        let assignments = [0u32, 2, 2, 0, 2];
        let num_clusters = 3;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("lists_f16.bin");
        let (counts, offsets) =
            write_lists::<Half>(&path, matrix.as_view(), &assignments, num_clusters).unwrap();
        assert_eq!(counts, vec![2, 0, 3]);

        let layout = Layout {
            dim,
            metric: Metric::L2,
            element_size: F16_SZ,
            num_points: num_points as u64,
            graph: GraphParams::default(),
            counts,
            offsets,
        };

        let bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64 % ALIGN, 0);

        let expected: [Vec<u32>; 3] = [vec![0, 3], vec![], vec![1, 2, 4]];
        for (c, want) in expected.iter().enumerate() {
            let w = cluster_window(&layout, c);
            if w.count == 0 {
                continue;
            }
            // Record start (and thus the ids) must be 4-byte aligned.
            assert_eq!(layout.offsets[c] % RECORD_ALIGN, 0);
            let slice = &bytes[w.aligned_start as usize..w.aligned_start as usize + w.aligned_len];
            let (ids, vectors) = parse_cluster::<Half>(slice, &w, dim);
            assert_eq!(ids, want.as_slice());
            for (vec, &id) in vectors.chunks_exact(dim).zip(ids.iter()) {
                assert!(vec.iter().all(|&x| x.to_f32() == id as f32));
            }
        }
    }

    #[test]
    fn metadata_round_trips() {
        let dim = 7;
        let counts = vec![3u32, 0, 5, 1];
        let offsets = compute_offsets(&counts, dim, F16_SZ);
        let layout = Layout {
            dim,
            metric: Metric::Cosine,
            element_size: F16_SZ,
            num_points: 9,
            graph: GraphParams {
                degree: 40,
                slack: 1.5,
                l_build: 96,
                alpha: 1.3,
            },
            counts: counts.clone(),
            offsets,
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        write_metadata(&path, &layout).unwrap();
        let loaded = read_metadata(&path).unwrap();

        assert_eq!(loaded.dim, dim);
        assert_eq!(loaded.metric, Metric::Cosine);
        assert_eq!(loaded.element_size, F16_SZ);
        assert_eq!(loaded.num_points, 9);
        assert_eq!(loaded.counts, counts);
        assert_eq!(loaded.num_clusters(), 4);
        assert_eq!(loaded.graph.degree, 40);
        assert_eq!(loaded.graph.l_build, 96);
        assert_eq!(loaded.graph.slack, 1.5);
        assert_eq!(loaded.graph.alpha, 1.3);
        // Offsets are recomputed identically from the persisted counts.
        assert_eq!(loaded.offsets, compute_offsets(&counts, dim, F16_SZ));
    }

    #[test]
    fn read_metadata_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        fs::write(&path, [0u8; 64]).unwrap();
        assert!(read_metadata(&path).is_err());
    }
}
