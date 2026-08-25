/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! On-disk layout for the inverted lists and the index metadata.
//!
//! Four files are written next to a user-supplied path prefix:
//!
//! * `<prefix>.graphivf_lists` — the inverted lists. For every cluster, in
//!   ascending cluster-id order, the bytes are `[ids: u32 x count][vectors:
//!   T x dim x count]`, packed back-to-back with no per-list padding. The
//!   whole file is zero-padded up to a 512-byte multiple so that sector-aligned
//!   reads never run past the end of the file.
//! * `<prefix>.graphivf_meta` — a compact header plus the per-cluster point
//!   counts. Byte offsets are recomputed from the counts on load.
//! * `<prefix>.graphivf_centroids.fbin` — the centroid matrix, always `f32`.
//! * `<prefix>.graphivf_graph` — the centroid graph's adjacency, written
//!   whenever the index was built with one. Absent for an exact-routed build,
//!   which never constructs a graph; the metadata says which of the two applies
//!   so a load never has to probe the filesystem to find out.
//!
//! Because every list is variable length, a read for cluster `c` reads the
//! smallest 512-aligned byte window that fully contains the list and indexes
//! into it; this avoids wasting disk space on padding when lists are tiny (the
//! expected regime, with only tens of points per centroid).

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::utils::VectorRepr;
use diskann_utils::io::write_bin;
use diskann_utils::views::MatrixView;

use crate::{
    centroids::GraphSnapshot,
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

/// Current metadata version, written by every save.
///
/// Version 1 recorded only the centroid graph's *recipe* and rebuilt the graph
/// on every load. Version 2 additionally records whether the graph itself was
/// persisted. Version 1 files still load: they simply take the rebuild path,
/// which is what they have always done.
const VERSION: u32 = 2;

/// Oldest metadata version still readable.
const MIN_VERSION: u32 = 1;

const GRAPH_MAGIC: u32 = 0x4752_4947; // "GIRG" little-endian
const GRAPH_VERSION: u32 = 1;

/// Assignment sentinel for a corpus row that is not in the index — either never
/// inserted or since deleted. [`write_lists_stored`] skips these rows, so a
/// build that does not cover the whole corpus still writes a well-formed index.
pub(crate) const NOT_INDEXED: u32 = u32::MAX;

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
    /// Whether the centroid graph was saved alongside the index.
    ///
    /// `false` when the index was built with exact routing, and so has no graph,
    /// or when it predates graph persistence. Recorded rather than inferred from
    /// whether the file happens to exist, so that a missing graph file is a
    /// detectable fault instead of a silent rebuild.
    pub has_graph: bool,
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
///
/// Rows whose assignment is [`NOT_INDEXED`] are skipped, which is how an online
/// build flushes a corpus some of whose points were deleted or never inserted.
/// The ids written into each list are the original row indices either way, so a
/// partial index still refers to points by their corpus position.
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
        if c == NOT_INDEXED {
            continue;
        }
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

/// Write the centroid matrix (always `f32`) to `path` in the `.fbin` format.
///
/// Shared by the batch build and the online-clusterer flush so both persist
/// centroids identically.
pub(crate) fn write_centroids(path: &Path, centroids: MatrixView<'_, f32>) -> Result<()> {
    let mut file = File::create(path)?;
    write_bin(centroids, &mut file).map_err(|e| GraphIvfError::malformed(e.to_string()))?;
    Ok(())
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
    w.write_u32::<LittleEndian>(layout.has_graph as u32)?;
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
    if !(MIN_VERSION..=VERSION).contains(&version) {
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

    // Version 1 never persisted a graph, so the field is absent and a rebuild is
    // the only option.
    let has_graph = if version >= 2 {
        match r.read_u32::<LittleEndian>()? {
            0 => false,
            1 => true,
            other => {
                return Err(GraphIvfError::malformed(format!(
                    "invalid centroid graph flag {other}"
                )))
            }
        }
    } else {
        false
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
        has_graph,
        counts,
        offsets,
    })
}

/// Write a centroid graph's adjacency to `path`.
pub(crate) fn write_graph(path: &Path, snapshot: &GraphSnapshot) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_u32::<LittleEndian>(GRAPH_MAGIC)?;
    w.write_u32::<LittleEndian>(GRAPH_VERSION)?;
    w.write_u64::<LittleEndian>(snapshot.num_centroids() as u64)?;

    write_edges(&mut w, &snapshot.start)?;
    for neighbors in &snapshot.adjacency {
        write_edges(&mut w, neighbors)?;
    }
    w.flush()?;
    Ok(())
}

fn write_edges(w: &mut impl Write, edges: &[u32]) -> Result<()> {
    w.write_u32::<LittleEndian>(edges.len() as u32)?;
    w.write_all(bytemuck::cast_slice(edges))?;
    Ok(())
}

/// Read a centroid graph's adjacency from `path`.
///
/// # Errors
///
/// Returns an error if the file is not a centroid graph, was written by a newer
/// version, or declares an adjacency list longer than the node count allows.
pub(crate) fn read_graph(path: &Path) -> Result<GraphSnapshot> {
    let mut r = BufReader::new(File::open(path)?);
    let magic = r.read_u32::<LittleEndian>()?;
    if magic != GRAPH_MAGIC {
        return Err(GraphIvfError::malformed("bad centroid graph magic"));
    }
    let version = r.read_u32::<LittleEndian>()?;
    if version != GRAPH_VERSION {
        return Err(GraphIvfError::malformed(format!(
            "unsupported centroid graph version {version}"
        )));
    }
    let num_centroids = r.read_u64::<LittleEndian>()? as usize;

    // A node cannot have more distinct out-edges than there are nodes to point
    // at, start point included. Bounding each list by that keeps a corrupt
    // length from turning into a huge allocation.
    let max_edges = num_centroids + 1;
    let start = read_edges(&mut r, max_edges)?;
    let adjacency = (0..num_centroids)
        .map(|_| read_edges(&mut r, max_edges))
        .collect::<Result<Vec<_>>>()?;

    Ok(GraphSnapshot { adjacency, start })
}

fn read_edges(r: &mut impl Read, max_edges: usize) -> Result<Vec<u32>> {
    let len = r.read_u32::<LittleEndian>()? as usize;
    if len > max_edges {
        return Err(GraphIvfError::malformed(format!(
            "centroid graph adjacency list of length {len} exceeds the {max_edges} nodes available"
        )));
    }
    let mut edges = vec![0u32; len];
    r.read_exact(bytemuck::cast_slice_mut(&mut edges))?;
    Ok(edges)
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
            has_graph: false,
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
            has_graph: false,
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
            has_graph: false,
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
            has_graph: false,
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
            has_graph: true,
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
        assert!(loaded.has_graph);
        // Offsets are recomputed identically from the persisted counts.
        assert_eq!(loaded.offsets, compute_offsets(&counts, dim, F16_SZ));
    }

    /// An index saved without a centroid graph must say so, rather than leaving
    /// a load to guess from whether a file happens to exist.
    #[test]
    fn metadata_records_absent_graph() {
        let counts = vec![1u32];
        let offsets = compute_offsets(&counts, 2, F32_SZ);
        let layout = Layout {
            dim: 2,
            metric: Metric::L2,
            element_size: F32_SZ,
            num_points: 1,
            graph: GraphParams::default(),
            has_graph: false,
            counts,
            offsets,
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta.bin");
        write_metadata(&path, &layout).unwrap();
        assert!(!read_metadata(&path).unwrap().has_graph);
    }

    /// Metadata written before graph persistence existed still loads, and takes
    /// the rebuild path because it records no graph.
    #[test]
    fn version_1_metadata_still_loads() {
        let counts = [2u32, 1];
        let dim = 3;
        let mut raw = Vec::new();
        raw.write_u32::<LittleEndian>(MAGIC).unwrap();
        raw.write_u32::<LittleEndian>(1).unwrap(); // version 1
        raw.write_u32::<LittleEndian>(Metric::L2.as_u8() as u32)
            .unwrap();
        raw.write_u32::<LittleEndian>(F32_SZ as u32).unwrap();
        raw.write_u32::<LittleEndian>(dim as u32).unwrap();
        raw.write_u64::<LittleEndian>(3).unwrap();
        raw.write_u64::<LittleEndian>(counts.len() as u64).unwrap();
        raw.write_u32::<LittleEndian>(32).unwrap(); // degree
        raw.write_u32::<LittleEndian>(64).unwrap(); // l_build
        raw.write_f32::<LittleEndian>(1.2).unwrap(); // slack
        raw.write_f32::<LittleEndian>(1.2).unwrap(); // alpha
        raw.write_all(bytemuck::cast_slice(&counts)).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meta_v1.bin");
        fs::write(&path, &raw).unwrap();

        let loaded = read_metadata(&path).unwrap();
        assert!(!loaded.has_graph);
        assert_eq!(loaded.graph.degree, 32);
        assert_eq!(loaded.graph.l_build, 64);
        assert_eq!(loaded.counts, counts);
    }

    /// Adjacency and the start point's edges both have to survive a round trip,
    /// or a restored graph is not the graph that was saved.
    #[test]
    fn graph_round_trips() {
        let snapshot = GraphSnapshot {
            adjacency: vec![vec![1, 2], vec![], vec![0, 1, u32::MAX]],
            start: vec![2, 0],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.bin");
        write_graph(&path, &snapshot).unwrap();

        assert_eq!(read_graph(&path).unwrap(), snapshot);
    }

    /// A truncated or corrupt adjacency length must be rejected rather than
    /// turned into a huge allocation.
    #[test]
    fn graph_rejects_oversized_adjacency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph_bad.bin");

        let mut raw = Vec::new();
        raw.write_u32::<LittleEndian>(GRAPH_MAGIC).unwrap();
        raw.write_u32::<LittleEndian>(GRAPH_VERSION).unwrap();
        raw.write_u64::<LittleEndian>(2).unwrap(); // two centroids
        raw.write_u32::<LittleEndian>(9999).unwrap(); // start point degree
        fs::write(&path, &raw).unwrap();

        assert!(read_graph(&path).is_err());
    }

    #[test]
    fn read_metadata_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.bin");
        fs::write(&path, [0u8; 64]).unwrap();
        assert!(read_metadata(&path).is_err());
    }
}
