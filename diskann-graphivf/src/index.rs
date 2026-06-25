/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The user-facing [`GraphIvfIndex`] and its per-thread [`Searcher`].

use std::{
    ffi::OsString,
    fs::File,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use diskann::{utils::VectorRepr, ANNError};
use diskann_disk::utils::{
    aligned_file_reader::{
        traits::{AlignedFileReader, AlignedReaderFactory},
        AlignedRead,
    },
    AlignedFileReaderFactory,
};
use diskann_providers::{
    index::diskann_async::MemoryIndex,
    utils::{create_thread_pool, ParallelIteratorInPool, RayonThreadPool},
};
use diskann_quantization::alloc::{AlignedAllocator, Poly};
use diskann_utils::{
    io::{read_bin, write_bin},
    views::{Matrix, MatrixView},
};
use diskann_vector::{distance::Metric as VMetric, PreprocessedDistanceFunction};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use tokio::runtime::Runtime;

use crate::{
    centroids,
    params::{BuildParams, Metric, SearchParams},
    profile::{BuildProfile, SearchProfile},
    storage::{self, Layout},
    GraphIvfError, Result,
};

const LISTS_SUFFIX: &str = ".graphivf_lists";
const META_SUFFIX: &str = ".graphivf_meta";
const CENTROIDS_SUFFIX: &str = ".graphivf_centroids.fbin";

/// The platform-specific aligned reader produced by [`AlignedFileReaderFactory`]
/// (io_uring on Linux, IOCP on Windows, buffered elsewhere).
type ListReader = <AlignedFileReaderFactory as AlignedReaderFactory>::AlignedReaderType;
/// The alignment required by [`ListReader`] (512 bytes for direct I/O, 1 byte
/// for the buffered fallback).
type ListAlign = <ListReader as AlignedFileReader>::Alignment;

/// A hybrid graph + clustered-IVF index.
///
/// Holds the in-memory centroid graph and the path to the on-disk inverted
/// lists. Use [`GraphIvfIndex::searcher`] to obtain a [`Searcher`] for querying;
/// create one searcher per thread for parallel search.
///
/// The type parameter `T` is the element type of the stored inverted-list
/// vectors (e.g. `f32` or [`diskann_vector::Half`]); any [`VectorRepr`] type is
/// supported. The centroid graph is always full-precision `f32`; only the
/// on-disk corpus vectors use `T`.
pub struct GraphIvfIndex<T: VectorRepr = f32> {
    centroids: MemoryIndex<f32>,
    layout: Arc<Layout>,
    lists_path: PathBuf,
    metric: Metric,
    dim: usize,
    _marker: PhantomData<fn() -> T>,
}

impl<T: VectorRepr> std::fmt::Debug for GraphIvfIndex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphIvfIndex")
            .field("metric", &self.metric)
            .field("element_type", &std::any::type_name::<T>())
            .field("dim", &self.dim)
            .field("num_clusters", &self.layout.num_clusters())
            .field("lists_path", &self.lists_path)
            .finish_non_exhaustive()
    }
}

impl<T: VectorRepr> GraphIvfIndex<T> {
    /// Build an index from `data` (one row per corpus vector) and write the
    /// on-disk artifacts using `prefix` (e.g. `/data/index` produces
    /// `/data/index.graphivf_lists`, `.graphivf_meta`, `.graphivf_centroids.fbin`).
    ///
    /// The inverted-list vectors are encoded as `T`; the input is always `f32`.
    pub fn build(data: MatrixView<'_, f32>, params: &BuildParams, prefix: &Path) -> Result<Self> {
        Self::build_profiled(data, params, prefix).map(|(index, _)| index)
    }

    /// Like [`build`](Self::build), but also returns a [`BuildProfile`]
    /// attributing the build wall-clock to its individual stages.
    pub fn build_profiled(
        data: MatrixView<'_, f32>,
        params: &BuildParams,
        prefix: &Path,
    ) -> Result<(Self, BuildProfile)> {
        let mut profile = BuildProfile::default();
        let total_start = Instant::now();

        let num_points = data.nrows();
        let dim = data.ncols();
        params.validate(num_points, dim)?;

        // For cosine, work on an L2-normalized copy of the corpus.
        let normalize_start = Instant::now();
        let normalized_storage: Option<Matrix<f32>> = if params.metric.normalizes() {
            let mut buf = data.as_slice().to_vec();
            for row in buf.chunks_mut(dim) {
                normalize(row);
            }
            Some(
                Matrix::try_from(buf.into_boxed_slice(), num_points, dim)
                    .map_err(|_| GraphIvfError::invalid("normalized matrix shape mismatch"))?,
            )
        } else {
            None
        };
        profile.normalize = normalize_start.elapsed();
        let work: MatrixView<'_, f32> = match &normalized_storage {
            Some(m) => m.as_view(),
            None => data,
        };

        let pool = create_thread_pool(params.num_threads)?;

        // 1. Sample the corpus and run k-means to obtain centroids. The sample
        //    and k-means phases are timed inside `run_kmeans`.
        let sample_size = params.effective_sample_size(num_points);
        let centroids_mat = run_kmeans(work, sample_size, params, &pool, &mut profile)?;

        let index = Self::finalize_build(
            work,
            centroids_mat,
            params,
            prefix,
            num_points,
            dim,
            &pool,
            &mut profile,
        )?;
        profile.total = total_start.elapsed();
        Ok((index, profile))
    }

    /// Build an index by refining an existing set of `seed_centroids` with
    /// `params.kmeans_iters` Lloyd iterations over the **entire** `data` (not a
    /// sample), then constructing the graph and inverted lists exactly as
    /// [`build_profiled`](Self::build_profiled).
    ///
    /// `seed_centroids` must be a `params.num_clusters x dim` matrix (e.g. the
    /// centroids written by a previous build, read back from
    /// `*.graphivf_centroids.fbin`). The k-means++ seeding and corpus sampling
    /// steps are skipped; `params.sample_size` is ignored.
    pub fn build_from_seed_centroids_profiled(
        data: MatrixView<'_, f32>,
        seed_centroids: MatrixView<'_, f32>,
        params: &BuildParams,
        prefix: &Path,
    ) -> Result<(Self, BuildProfile)> {
        let mut profile = BuildProfile::default();
        let total_start = Instant::now();

        let num_points = data.nrows();
        let dim = data.ncols();
        params.validate(num_points, dim)?;
        if seed_centroids.nrows() != params.num_clusters || seed_centroids.ncols() != dim {
            return Err(GraphIvfError::invalid(format!(
                "seed centroids shape {}x{} does not match num_clusters {} x dim {dim}",
                seed_centroids.nrows(),
                seed_centroids.ncols(),
                params.num_clusters
            )));
        }

        // For cosine, work on an L2-normalized copy of the corpus.
        let normalize_start = Instant::now();
        let normalized_storage: Option<Matrix<f32>> = if params.metric.normalizes() {
            let mut buf = data.as_slice().to_vec();
            for row in buf.chunks_mut(dim) {
                normalize(row);
            }
            Some(
                Matrix::try_from(buf.into_boxed_slice(), num_points, dim)
                    .map_err(|_| GraphIvfError::invalid("normalized matrix shape mismatch"))?,
            )
        } else {
            None
        };
        profile.normalize = normalize_start.elapsed();
        let work: MatrixView<'_, f32> = match &normalized_storage {
            Some(m) => m.as_view(),
            None => data,
        };

        let pool = create_thread_pool(params.num_threads)?;

        // 1. Refine the seed centroids with Lloyd iterations over the full corpus.
        //    No sampling and no k-means++ seeding (`profile.sample` stays zero).
        let kmeans_start = Instant::now();
        let mut centers = seed_centroids.as_slice().to_vec();
        let mut cancel = false;
        diskann_disk::utils::run_lloyds(
            work.as_slice(),
            num_points,
            dim,
            &mut centers,
            params.num_clusters,
            params.kmeans_iters,
            &mut cancel,
            pool.as_ref(),
        )?;
        profile.kmeans = kmeans_start.elapsed();
        let centroids_mat = Matrix::try_from(centers.into_boxed_slice(), params.num_clusters, dim)
            .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))?;

        let index = Self::finalize_build(
            work,
            centroids_mat,
            params,
            prefix,
            num_points,
            dim,
            &pool,
            &mut profile,
        )?;
        profile.total = total_start.elapsed();
        Ok((index, profile))
    }

    /// Finalize a build once centroids are known: persist the centroids, build
    /// the in-memory graph, assign every corpus point, and write the inverted
    /// lists and metadata. Shared by [`build_profiled`](Self::build_profiled)
    /// and [`build_from_seed_centroids_profiled`](Self::build_from_seed_centroids_profiled).
    #[allow(clippy::too_many_arguments)]
    fn finalize_build(
        work: MatrixView<'_, f32>,
        centroids_mat: Matrix<f32>,
        params: &BuildParams,
        prefix: &Path,
        num_points: usize,
        dim: usize,
        pool: &RayonThreadPool,
        profile: &mut BuildProfile,
    ) -> Result<Self> {
        // Persist centroids so the graph can be rebuilt on load.
        let write_centroids_start = Instant::now();
        let centroids_path = with_suffix(prefix, CENTROIDS_SUFFIX);
        let mut centroids_file = File::create(&centroids_path)?;
        write_bin(centroids_mat.as_view(), &mut centroids_file)
            .map_err(|e| GraphIvfError::malformed(e.to_string()))?;
        profile.write_centroids = write_centroids_start.elapsed();

        // 2. Build the in-memory graph over the centroids.
        let build_graph_start = Instant::now();
        let centroids = centroids::build(centroids_mat, &params.graph, params.num_threads)?;
        profile.build_graph = build_graph_start.elapsed();

        // 3. Assign every corpus point to its nearest centroid via graph search.
        let assign_start = Instant::now();
        let assignments = assign(&centroids, work, params.assign_l, pool)?;
        profile.assign = assign_start.elapsed();

        // 4. Write the inverted lists and the metadata.
        let write_lists_start = Instant::now();
        let lists_path = with_suffix(prefix, LISTS_SUFFIX);
        let (counts, offsets) =
            storage::write_lists::<T>(&lists_path, work, &assignments, params.num_clusters)?;
        profile.write_lists = write_lists_start.elapsed();

        let layout = Layout {
            dim,
            metric: params.metric,
            element_size: std::mem::size_of::<T>(),
            num_points: num_points as u64,
            graph: params.graph,
            counts,
            offsets,
        };
        let write_metadata_start = Instant::now();
        storage::write_metadata(&with_suffix(prefix, META_SUFFIX), &layout)?;
        profile.write_metadata = write_metadata_start.elapsed();

        Ok(Self {
            centroids,
            layout: Arc::new(layout),
            lists_path,
            metric: params.metric,
            dim,
            _marker: PhantomData,
        })
    }

    /// Load a previously built index. The centroid graph is rebuilt in memory
    /// from the persisted centroids using `num_threads` workers.
    ///
    /// # Errors
    ///
    /// Returns an error if the persisted element size does not match `T`. This
    /// is a size check (e.g. it distinguishes `f32` from [`diskann_vector::Half`]),
    /// not a full type check.
    pub fn load(prefix: &Path, num_threads: usize) -> Result<Self> {
        let layout = storage::read_metadata(&with_suffix(prefix, META_SUFFIX))?;
        let want = std::mem::size_of::<T>();
        if layout.element_size != want {
            return Err(GraphIvfError::invalid(format!(
                "index was written with {}-byte elements but {}-byte ({}) was requested",
                layout.element_size,
                want,
                std::any::type_name::<T>(),
            )));
        }
        let metric = layout.metric;
        let dim = layout.dim;

        let mut centroids_file = File::open(with_suffix(prefix, CENTROIDS_SUFFIX))?;
        let centroids_mat: Matrix<f32> =
            read_bin(&mut centroids_file).map_err(|e| GraphIvfError::malformed(e.to_string()))?;
        let centroids = centroids::build(centroids_mat, &layout.graph, num_threads)?;

        Ok(Self {
            centroids,
            layout: Arc::new(layout),
            lists_path: with_suffix(prefix, LISTS_SUFFIX),
            metric,
            dim,
            _marker: PhantomData,
        })
    }

    /// Number of clusters in the index.
    pub fn num_clusters(&self) -> usize {
        self.layout.num_clusters()
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Create a [`Searcher`]. Each searcher owns its own disk reader and small
    /// runtime, so allocate one per thread when searching in parallel.
    pub fn searcher(&self) -> Result<Searcher<T>> {
        let factory = AlignedFileReaderFactory::new(self.lists_path.to_string_lossy().into_owned());
        let reader = factory.build()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(ANNError::from)?;
        Ok(Searcher {
            reader,
            runtime,
            centroids: Arc::clone(&self.centroids),
            layout: Arc::clone(&self.layout),
            metric: self.metric,
            dim: self.dim,
            cids: Vec::new(),
            cdist: Vec::new(),
            windows: Vec::new(),
            scratch: Poly::<[u8], AlignedAllocator>::broadcast(0u8, 0, AlignedAllocator::A512)
                .map_err(|e| GraphIvfError::invalid(format!("aligned allocation failed: {e:?}")))?,
            _marker: PhantomData,
        })
    }
}

/// A single-threaded query handle into a [`GraphIvfIndex`].
///
/// Holds a disk reader, a current-thread runtime to drive the (in-memory) graph
/// search, and reusable scratch buffers. Not shareable across threads — clone
/// more searchers from the index for parallelism. `T` is the stored
/// inverted-list element type.
pub struct Searcher<T: VectorRepr = f32> {
    reader: ListReader,
    runtime: Runtime,
    centroids: MemoryIndex<f32>,
    layout: Arc<Layout>,
    metric: Metric,
    dim: usize,
    cids: Vec<u32>,
    cdist: Vec<f32>,
    /// Per-query read windows, reused across queries to avoid reallocation.
    windows: Vec<storage::ClusterWindow>,
    /// One reusable 512-aligned buffer holding every probed cluster's read
    /// window back-to-back. Grown on demand; never shrunk.
    scratch: Poly<[u8], AlignedAllocator>,
    _marker: PhantomData<fn() -> T>,
}

impl<T: VectorRepr> std::fmt::Debug for Searcher<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Searcher")
            .field("metric", &self.metric)
            .field("element_type", &std::any::type_name::<T>())
            .field("dim", &self.dim)
            .field("num_clusters", &self.layout.num_clusters())
            .finish_non_exhaustive()
    }
}

impl<T: VectorRepr> Searcher<T> {
    /// Return the `k` approximate nearest neighbors of `query` as `(id,
    /// distance)` pairs, sorted by ascending distance. `id` is the row index of
    /// the vector in the original corpus; `distance` is squared-L2.
    ///
    /// `query` is in the stored element type `T` and is scored directly against
    /// the `T` corpus. For cosine, normalize `query` before calling — the index
    /// does not normalize it (the corpus is normalized at build time).
    pub fn search(
        &mut self,
        query: &[T],
        k: usize,
        params: &SearchParams,
    ) -> Result<Vec<(u32, f32)>> {
        self.search_profiled(query, k, params)
            .map(|(results, _)| results)
    }

    /// Like [`search`](Self::search), but also returns a [`SearchProfile`]
    /// attributing the query wall-clock to its individual stages.
    pub fn search_profiled(
        &mut self,
        query: &[T],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<(u32, f32)>, SearchProfile)> {
        params.validate(self.layout.num_clusters())?;
        if k == 0 {
            return Err(GraphIvfError::invalid("k must be non-zero"));
        }
        if query.len() != self.dim {
            return Err(GraphIvfError::invalid(format!(
                "query has dim {} but index has dim {}",
                query.len(),
                self.dim
            )));
        }

        let mut profile = SearchProfile::default();
        let total_start = Instant::now();

        // Preprocess the query once into a `T`-space scorer reused across every
        // candidate. Query and corpus are both `T`, so scoring runs directly
        // over `T` via `T::QueryDistance` with no per-candidate decode. Cosine
        // is reduced to squared-L2 (the caller normalizes the query, matching
        // the corpus normalized at build time).
        let preprocess_start = Instant::now();
        let scorer = T::query_distance(query, VMetric::L2);

        // The centroid graph is full-precision `f32`, so decode the query to
        // `f32` once for the centroid KNN (a no-op when `T == f32`).
        let q_f32 = T::as_f32(query).map_err(|e| GraphIvfError::Ann(e.into()))?;
        profile.preprocess = preprocess_start.elapsed();

        // 1. Find the nearest `nlist` centroids via graph search.
        let centroid_search_start = Instant::now();
        let nlist = params.nlist;
        let l = params.effective_l();
        self.cids.clear();
        self.cids.resize(nlist, 0);
        self.cdist.clear();
        self.cdist.resize(nlist, 0.0);
        let n = centroids::search(
            &self.centroids,
            &self.runtime,
            &q_f32,
            l,
            &mut self.cids,
            &mut self.cdist,
        )?;
        profile.centroid_search = centroid_search_start.elapsed();

        // 2. Build a sector-aligned read window per non-empty cluster and carve
        //    each window a disjoint slice of one reusable aligned buffer.
        let plan_io_start = Instant::now();
        let dim = self.dim;
        self.windows.clear();
        let mut total_len: usize = 0;
        for &c in &self.cids[..n] {
            let window = storage::cluster_window(&self.layout, c as usize);
            if window.count == 0 {
                continue;
            }
            total_len += window.aligned_len;
            self.windows.push(window);
        }

        // Grow the reusable scratch buffer only when this query needs more space
        // than any previous one; the steady state performs no allocation and no
        // zeroing (the disk read overwrites every byte that is later parsed).
        if self.scratch.len() < total_len {
            self.scratch = Poly::<[u8], AlignedAllocator>::broadcast(
                0u8,
                total_len,
                AlignedAllocator::A512,
            )
            .map_err(|e| GraphIvfError::invalid(format!("aligned allocation failed: {e:?}")))?;
        }

        // Carve one disjoint, 512-aligned sub-slice per window. The allocator
        // aligns the base to 512 and every `aligned_len` is a multiple of 512,
        // so each successive sub-slice start stays sector-aligned.
        let mut reads = Vec::with_capacity(self.windows.len());
        let mut rest: &mut [u8] = &mut self.scratch[..total_len];
        for window in &self.windows {
            let (head, tail) = rest.split_at_mut(window.aligned_len);
            reads.push(AlignedRead::<u8, ListAlign>::new(
                window.aligned_start,
                head,
            )?);
            rest = tail;
        }
        profile.plan_io = plan_io_start.elapsed();

        // 3. Issue all reads for this query as a single batch.
        //    TODO(perf): dedupe overlapping 512-byte windows of adjacent
        //    clusters that share a sector.
        let disk_read_start = Instant::now();
        self.reader.read(&mut reads)?;
        profile.disk_read = disk_read_start.elapsed();
        // Release the mutable borrows of `scratch` before scoring reads it.
        drop(reads);

        // 4. Exhaustively score the query against the fetched vectors. Both the
        //    query and the stored corpus vectors are `T`; scoring runs directly
        //    over `T` via the preprocessed `T::QueryDistance` with no decode.
        //    TODO(perf): this scan is embarrassingly parallel across clusters.
        let score_start = Instant::now();
        let mut candidates: Vec<(u32, f32)> = Vec::new();
        let mut offset = 0usize;
        for window in &self.windows {
            let buf = &self.scratch[offset..offset + window.aligned_len];
            let (ids, vectors) = storage::parse_cluster::<T>(buf, window, dim);
            for (vec, &id) in vectors.chunks_exact(dim).zip(ids.iter()) {
                candidates.push((id, scorer.evaluate_similarity(vec)));
            }
            offset += window.aligned_len;
        }
        profile.score = score_start.elapsed();

        let topk_start = Instant::now();
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        profile.topk = topk_start.elapsed();

        profile.total = total_start.elapsed();
        Ok((candidates, profile))
    }
}

/// Sample the corpus and run k-means, returning the centroids as a `k x dim`
/// matrix. Records the sampling and k-means wall-clock into `profile`.
fn run_kmeans(
    work: MatrixView<'_, f32>,
    sample_size: usize,
    params: &BuildParams,
    pool: &RayonThreadPool,
    profile: &mut BuildProfile,
) -> Result<Matrix<f32>> {
    let num_points = work.nrows();
    let dim = work.ncols();
    let mut rng = StdRng::seed_from_u64(params.seed);

    let sample_start = Instant::now();
    let sample_idx = rand::seq::index::sample(&mut rng, num_points, sample_size).into_vec();
    let mut sample = vec![0.0f32; sample_size * dim];
    for (i, &p) in sample_idx.iter().enumerate() {
        sample[i * dim..(i + 1) * dim].copy_from_slice(work.row(p));
    }
    profile.sample = sample_start.elapsed();

    let kmeans_start = Instant::now();
    let mut centers = vec![0.0f32; params.num_clusters * dim];
    let mut cancel = false;
    diskann_disk::utils::k_means_clustering(
        &sample,
        sample_size,
        dim,
        &mut centers,
        params.num_clusters,
        params.kmeans_iters,
        &mut rng,
        &mut cancel,
        pool.as_ref(),
    )?;
    profile.kmeans = kmeans_start.elapsed();

    Matrix::try_from(centers.into_boxed_slice(), params.num_clusters, dim)
        .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))
}

/// Assign each corpus point to its nearest centroid via graph search.
///
/// Parallelized across chunks of points; each worker drives the (in-memory)
/// graph search with its own current-thread runtime.
fn assign(
    centroids: &MemoryIndex<f32>,
    work: MatrixView<'_, f32>,
    assign_l: usize,
    pool: &RayonThreadPool,
) -> Result<Vec<u32>> {
    let num_points = work.nrows();
    let mut assignments = vec![0u32; num_points];
    const CHUNK: usize = 256;

    assignments
        .par_chunks_mut(CHUNK)
        .enumerate()
        .try_for_each_in_pool(pool.as_ref(), |(ci, out)| -> Result<()> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .map_err(ANNError::from)?;
            let mut ids = [0u32; 1];
            let mut dist = [0.0f32; 1];
            for (j, slot) in out.iter_mut().enumerate() {
                let pid = ci * CHUNK + j;
                centroids::search(
                    centroids,
                    &runtime,
                    work.row(pid),
                    assign_l,
                    &mut ids,
                    &mut dist,
                )?;
                *slot = ids[0];
            }
            Ok(())
        })?;

    Ok(assignments)
}

/// L2-normalize a vector in place (no-op for a zero vector).
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Append `suffix` to a path prefix, e.g. `/a/idx` + `.graphivf_meta`.
fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s: OsString = prefix.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}
