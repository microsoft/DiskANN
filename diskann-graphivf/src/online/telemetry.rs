/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Build telemetry and stable CSV serialization.

use std::path::Path;

/// One cluster-split event recorded during an online build.
///
/// A batched insert splits every overflowing cluster, so it emits one event per
/// split parent. Events from one batch share `insert_index` and `live_after`.
#[derive(Debug, Clone, Copy)]
pub struct SplitEvent {
    /// Number of inserts completed when this split fired.
    pub insert_index: u64,
    /// The centroid id that was split and retired.
    pub cluster: u32,
    /// Size of the split cluster at split time.
    pub cluster_size: usize,
    /// Live neighbor clusters drawn into reassignment, excluding the children.
    pub num_neighbors: usize,
    /// Points in the parent posting plus the nearby postings examined by LIRE.
    pub region_points: usize,
    /// Deduplicated points surviving LIRE's two necessary-condition filters and
    /// sent to the final global NPA check, attributed once within the batch.
    pub npa_candidates: usize,
    /// Points that actually changed cluster during reassignment.
    pub num_reassigned: usize,
    /// Live centroid count immediately after the batch's splits.
    pub live_after: usize,
    /// Wall-clock of this parent's balanced fit and condition filtering,
    /// microseconds.
    pub two_means_us: u64,
    /// Wall-clock of this parent's reassignment pass, microseconds.
    pub reassign_us: u64,
    /// Attributed algorithm time (`two_means_us + reassign_us`). This excludes
    /// neighborhood search and graph publication, which are shared by the
    /// batch; [`BuildTelemetry::split_us`] measures the complete pass.
    pub total_us: u64,
}

/// One LIRE merge event recorded during a delete.
///
/// A merge retires one underfull centroid and globally routes only that
/// centroid's remaining members. A batched delete emits one event per
/// retirement; all events from the batch share `op_index` and `live_after`.
#[derive(Debug, Clone, Copy)]
pub struct MergeEvent {
    /// Total operations (inserts plus deletes) completed when this merge fired.
    pub op_index: u64,
    /// The underfull centroid that was merged and retired.
    pub victim: u32,
    /// Points the victim still held when it was merged.
    pub victim_size: usize,
    /// Landing-site count; LIRE records one capacity-compatible merge target.
    pub num_neighbors: usize,
    /// Points that actually changed cluster.
    pub num_reassigned: usize,
    /// Live centroid count immediately after the batch's retirements.
    pub live_after: usize,
    /// Wall-clock of the landing-site search, microseconds.
    pub search_us: u64,
    /// Wall-clock of reassignment, microseconds.
    pub reassign_us: u64,
    /// Attributed algorithm time (`search_us + reassign_us`). This excludes
    /// graph retirement, which is shared by the batch; [`BuildTelemetry::merge_us`]
    /// measures the complete pass.
    pub total_us: u64,
}

/// Telemetry accumulated over an online build.
///
/// Always collected. Split and merge event vectors remain separate because
/// their clocks, fields, and stable downstream CSV schemas differ.
#[derive(Debug, Clone, Default)]
pub struct BuildTelemetry {
    /// Total points inserted.
    pub total_inserts: u64,
    /// Total splits performed.
    pub total_splits: u64,
    /// Points that changed cluster, summed across split passes.
    pub total_reassigned: u64,
    /// Cumulative insert-routing time, microseconds.
    pub routing_us: u64,
    /// Complete cumulative split time, microseconds.
    pub split_us: u64,
    /// Total points deleted.
    pub total_deletes: u64,
    /// Total merges performed.
    pub total_merges: u64,
    /// Points that changed cluster, summed across merge passes.
    pub total_merge_reassigned: u64,
    /// Cumulative point-removal time, excluding merge handling, microseconds.
    pub delete_us: u64,
    /// Complete cumulative merge time, microseconds.
    pub merge_us: u64,
    /// Ordered per-split records.
    pub splits: Vec<SplitEvent>,
    /// Ordered per-merge records.
    pub merges: Vec<MergeEvent>,
}

impl BuildTelemetry {
    /// Write the stable per-split CSV schema to `path`.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from creating or writing the file.
    pub fn write_csv(&self, path: &Path) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(64 + self.splits.len() * 48);
        out.push_str(
            "insert_index,cluster,cluster_size,num_neighbors,region_points,npa_candidates,num_reassigned,\
             live_after,two_means_us,reassign_us,total_us\n",
        );
        for event in &self.splits {
            let _ = writeln!(
                out,
                "{},{},{},{},{},{},{},{},{},{},{}",
                event.insert_index,
                event.cluster,
                event.cluster_size,
                event.num_neighbors,
                event.region_points,
                event.npa_candidates,
                event.num_reassigned,
                event.live_after,
                event.two_means_us,
                event.reassign_us,
                event.total_us,
            );
        }
        std::fs::write(path, out)
    }

    /// Write the stable per-merge CSV schema to `path`.
    ///
    /// Kept separate from [`write_csv`](Self::write_csv) so adding merges does
    /// not change the split schema consumed by downstream analysis.
    ///
    /// # Errors
    ///
    /// Returns any I/O error from creating or writing the file.
    pub fn write_merges_csv(&self, path: &Path) -> std::io::Result<()> {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(64 + self.merges.len() * 48);
        out.push_str(
            "op_index,victim,victim_size,num_neighbors,num_reassigned,\
             live_after,search_us,reassign_us,total_us\n",
        );
        for event in &self.merges {
            let _ = writeln!(
                out,
                "{},{},{},{},{},{},{},{},{}",
                event.op_index,
                event.victim,
                event.victim_size,
                event.num_neighbors,
                event.num_reassigned,
                event.live_after,
                event.search_us,
                event.reassign_us,
                event.total_us,
            );
        }
        std::fs::write(path, out)
    }
}
