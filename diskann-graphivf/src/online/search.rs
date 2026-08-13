/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query handle for an online clusterer's current in-memory partition.

use diskann::{utils::VectorRepr, ANNError};
use diskann_vector::PreprocessedDistanceFunction;
use tokio::runtime::Runtime;

use super::OnlineClusterer;
use crate::{params::SearchParams, GraphIvfError, Result};

/// Work performed by one in-memory online query.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OnlineSearchStats {
    /// Corpus vectors scored across the probed inverted lists.
    pub points_scanned: usize,
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
        if query.len() != clusterer.dim {
            return Err(GraphIvfError::invalid(format!(
                "query has dim {} but index has dim {}",
                query.len(),
                clusterer.dim
            )));
        }

        self.cids.clear();
        self.cids.resize(params.nlist, 0);
        self.cdist.clear();
        self.cdist.resize(params.nlist, 0.0);
        let found = clusterer.centroids.search(
            &self.runtime,
            query,
            params.effective_l(),
            &mut self.cids,
            &mut self.cdist,
        )?;

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
}
