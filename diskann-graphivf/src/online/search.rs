/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query handle for an online clusterer's current in-memory partition.

use diskann::{utils::VectorRepr, ANNError};
use diskann_vector::PreprocessedDistanceFunction;
use tokio::runtime::Runtime;

use super::OnlineClusterer;
use crate::{cluster::sq_l2, params::SearchParams, GraphIvfError, Result};

/// Work performed by one in-memory online query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OnlineSearchStats {
    /// Corpus vectors scored across the probed inverted lists.
    pub points_scanned: usize,
}

/// How closely the centroid graph reproduced an exact centroid ranking for one
/// query.
///
/// Separates the two ways a probe can miss: the graph selected the wrong
/// clusters, or it selected the right ones and they did not hold the neighbors.
/// Only the first is a search-quality problem.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CentroidRecall {
    /// Clusters asked for, i.e. [`SearchParams::nlist`].
    pub requested: usize,
    /// Clusters the centroid graph actually returned. Falls short of
    /// `requested` only if the walk could not reach that many.
    pub retrieved: usize,
    /// Retrieved clusters that are genuinely among the nearest `requested`.
    pub matched: usize,
}

impl CentroidRecall {
    /// [`matched`](Self::matched) as a fraction of
    /// [`requested`](Self::requested), in `0.0..=1.0`.
    pub fn recall(&self) -> f32 {
        if self.requested == 0 {
            return 1.0;
        }
        self.matched as f32 / self.requested as f32
    }
}

/// A single-threaded query handle into a live [`OnlineClusterer`].
///
/// Answers queries against the in-memory `f32` corpus, so it measures the
/// partition the online build has reached without flushed-index quantization
/// error. Not shareable across threads; open one handle per worker.
pub struct OnlineSearcher<'a> {
    clusterer: &'a OnlineClusterer,
    runtime: Runtime,
    cids: Vec<u32>,
    cdist: Vec<f32>,
    scanned: u64,
}

impl std::fmt::Debug for OnlineSearcher<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnlineSearcher")
            .field("dim", &self.clusterer.dim)
            .field("num_clusters", &self.clusterer.num_clusters())
            .finish_non_exhaustive()
    }
}

impl<'a> OnlineSearcher<'a> {
    pub(super) fn new(clusterer: &'a OnlineClusterer) -> Result<Self> {
        clusterer.ensure_healthy()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(ANNError::from)?;
        Ok(Self {
            clusterer,
            runtime,
            cids: Vec::new(),
            cdist: Vec::new(),
            scanned: 0,
        })
    }

    /// Corpus vectors scored across every query this handle has answered.
    pub fn points_scanned(&self) -> u64 {
        self.scanned
    }

    /// Return the `k` approximate nearest neighbors of `query` as `(id,
    /// distance)` pairs sorted by ascending distance.
    ///
    /// This convenience method allocates its result vector. Call
    /// [`search_into`](Self::search_into) to reuse caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is zero, `query` has the wrong dimension,
    /// `params` is invalid for the current cluster count, or centroid-graph
    /// navigation fails.
    pub fn search(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<Vec<(u32, f32)>> {
        let mut results = Vec::new();
        self.search_into(query, k, params, &mut results)?;
        Ok(results)
    }

    /// Write the `k` approximate nearest neighbors of `query` into `results`
    /// and return the work performed by this query.
    ///
    /// `results` is cleared after validation and graph navigation succeed, then
    /// reused for sorted output. Retaining it across calls avoids the allocation
    /// and copy imposed by [`search`](Self::search).
    ///
    /// Centroid navigation always uses L2. Candidate scoring honors the online
    /// build metric; inner-product distance is negated so smaller remains better.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`search`](Self::search). On error, `results`
    /// is left unchanged.
    pub fn search_into(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        results: &mut Vec<(u32, f32)>,
    ) -> Result<OnlineSearchStats> {
        let clusterer = self.clusterer;
        params.validate(clusterer.num_clusters())?;
        if k == 0 {
            return Err(GraphIvfError::invalid("k must be non-zero"));
        }
        self.check_dim(query)?;
        let found = self.select_centroids(query, params)?;

        let scorer = f32::query_distance(query, clusterer.params.metric.search_metric());
        results.clear();
        for &cid in &self.cids[..found] {
            for &pid in clusterer.partition.members(cid) {
                results.push((
                    pid,
                    scorer.evaluate_similarity(clusterer.points.row(pid as usize)),
                ));
            }
        }
        let points_scanned = results.len();
        self.scanned += points_scanned as u64;

        if results.len() > k {
            results.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
            results.truncate(k);
        }
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(OnlineSearchStats { points_scanned })
    }

    /// Score the centroids `params` selects for `query` against an exact scan of
    /// every live centroid.
    ///
    /// Costs one full pass over the centroid table per call, so this is a
    /// diagnostic and not something to run on a query path. It answers whether
    /// a given [`centroid_search_alpha`](SearchParams::centroid_search_alpha) is
    /// wide enough for the graph as it stands — which is a property of the graph
    /// too, not just of the beam, and drifts as splits and merges churn it.
    ///
    /// # Errors
    ///
    /// Returns an error if `query` has the wrong dimension, `params` is invalid
    /// for the current cluster count, or centroid-graph navigation fails.
    pub fn centroid_recall(
        &mut self,
        query: &[f32],
        params: &SearchParams,
    ) -> Result<CentroidRecall> {
        let clusterer = self.clusterer;
        params.validate(clusterer.num_clusters())?;
        self.check_dim(query)?;

        // Take the exact cutoff first. It reduces to a single scalar, so the
        // centroid buffers are free for the graph walk that follows.
        self.resize_centroid_buffers(params.nlist);
        let ranked = clusterer
            .centroids
            .exact_search(query, &mut self.cids, &mut self.cdist)?;
        if ranked < params.nlist {
            return Err(GraphIvfError::invalid(format!(
                "fewer than {} live centroids for an exact ranking",
                params.nlist
            )));
        }
        // Comparing against a distance rather than an id set counts a tie as
        // correct instead of penalizing whichever tie-break the graph took.
        let cutoff = self.cdist[params.nlist - 1];

        let found = self.select_centroids(query, params)?;
        let matched = self.cids[..found]
            .iter()
            .filter(|&&cid| {
                clusterer
                    .centroids
                    .get(cid)
                    .is_some_and(|c| sq_l2(query, c) <= cutoff)
            })
            .count();

        Ok(CentroidRecall {
            requested: params.nlist,
            retrieved: found,
            matched,
        })
    }

    fn check_dim(&self, query: &[f32]) -> Result<()> {
        if query.len() != self.clusterer.dim {
            return Err(GraphIvfError::invalid(format!(
                "query has dim {} but index has dim {}",
                query.len(),
                self.clusterer.dim
            )));
        }
        Ok(())
    }

    /// Size `self.cids` / `self.cdist` to hold `n` centroid results.
    fn resize_centroid_buffers(&mut self, n: usize) {
        self.cids.clear();
        self.cids.resize(n, 0);
        self.cdist.clear();
        self.cdist.resize(n, 0.0);
    }

    /// Fill `self.cids` / `self.cdist` with the centroids `params` selects and
    /// return how many the walk actually reached.
    fn select_centroids(&mut self, query: &[f32], params: &SearchParams) -> Result<usize> {
        self.resize_centroid_buffers(params.nlist);
        self.clusterer.centroids.search(
            &self.runtime,
            query,
            params.effective_l(),
            &mut self.cids,
            &mut self.cdist,
        )
    }
}
