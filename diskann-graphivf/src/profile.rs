/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fine-grained wall-clock breakdowns for build and search, suitable for
//! constructing "performance layer cakes" (per-component latency attribution).
//!
//! [`BuildProfile`] attributes a single index build across its stages, and
//! [`SearchProfile`] attributes a single query across its stages. Both are
//! returned by the `*_profiled` entry points on
//! [`GraphIvfIndex`](crate::GraphIvfIndex) /
//! [`Searcher`](crate::Searcher); the non-profiled variants discard them.
//!
//! Each phase is independent wall-clock time; `total` is end-to-end and may
//! exceed the phase sum by a small unattributed remainder (allocation, control
//! flow), surfaced as `other` in the [`std::fmt::Display`] output.

use std::{fmt, time::Duration};

use serde::{Deserialize, Serialize};

/// Per-phase wall-clock breakdown of a single index build.
///
/// Phases are sequential and non-overlapping. Thread-pool setup and small
/// bookkeeping are not attributed to any phase and appear as `other` in the
/// [`Display`](std::fmt::Display) layer cake.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct BuildProfile {
    /// L2-normalizing the corpus (cosine only; zero for L2).
    pub normalize: Duration,
    /// Sampling the corpus and gathering the k-means training set.
    pub sample: Duration,
    /// Lloyd's k-means iterations over the sample.
    pub kmeans: Duration,
    /// Persisting the centroid matrix to disk.
    pub write_centroids: Duration,
    /// Building the in-memory centroid (Vamana) graph.
    pub build_graph: Duration,
    /// Assigning every corpus point to its nearest centroid via graph search.
    pub assign: Duration,
    /// Encoding and writing the per-cluster inverted lists to disk.
    pub write_lists: Duration,
    /// Writing the index metadata file.
    pub write_metadata: Duration,
    /// End-to-end build wall-clock (the sum of the phases plus unattributed
    /// remainder).
    pub total: Duration,
}

impl BuildProfile {
    /// The phases in execution order paired with their labels (excludes
    /// [`total`](Self::total)).
    pub fn phases(&self) -> [(&'static str, Duration); 8] {
        [
            ("normalize", self.normalize),
            ("sample", self.sample),
            ("kmeans", self.kmeans),
            ("write_centroids", self.write_centroids),
            ("build_graph", self.build_graph),
            ("assign", self.assign),
            ("write_lists", self.write_lists),
            ("write_metadata", self.write_metadata),
        ]
    }
}

impl fmt::Display for BuildProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_layer_cake(f, "Build latency breakdown", &self.phases(), self.total)
    }
}

/// Per-phase wall-clock breakdown of a single query.
///
/// Phases are sequential and non-overlapping. Validation and small bookkeeping
/// are not attributed to any phase and appear as `other` in the
/// [`Display`](std::fmt::Display) layer cake.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SearchProfile {
    /// Building the `T`-space scorer and decoding the query to `f32` for the
    /// centroid graph.
    pub preprocess: Duration,
    /// Finding the nearest `nlist` centroids via graph search.
    pub centroid_search: Duration,
    /// Computing per-cluster read windows and allocating aligned buffers.
    pub plan_io: Duration,
    /// The batched disk read of the selected inverted lists.
    pub disk_read: Duration,
    /// Scoring the query against every fetched corpus vector.
    pub score: Duration,
    /// Selecting and sorting the top-k candidates.
    pub topk: Duration,
    /// End-to-end query wall-clock (the sum of the phases plus unattributed
    /// remainder).
    pub total: Duration,
}

impl SearchProfile {
    /// The phases in execution order paired with their labels (excludes
    /// [`total`](Self::total)).
    pub fn phases(&self) -> [(&'static str, Duration); 6] {
        [
            ("preprocess", self.preprocess),
            ("centroid_search", self.centroid_search),
            ("plan_io", self.plan_io),
            ("disk_read", self.disk_read),
            ("score", self.score),
            ("topk", self.topk),
        ]
    }
}

impl fmt::Display for SearchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_layer_cake(f, "Search latency breakdown", &self.phases(), self.total)
    }
}

/// Render a labeled per-phase breakdown with each phase's share of `total`,
/// plus an `other` line for the unattributed remainder.
fn write_layer_cake(
    f: &mut fmt::Formatter<'_>,
    title: &str,
    phases: &[(&str, Duration)],
    total: Duration,
) -> fmt::Result {
    let total_ns = total.as_nanos().max(1);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let pct = |d: Duration| d.as_nanos() as f64 / total_ns as f64 * 100.0;

    writeln!(f, "{title} (total {:.3} ms):", ms(total))?;
    let mut attributed: u128 = 0;
    for (label, d) in phases {
        attributed += d.as_nanos();
        writeln!(f, "  {label:>16}: {:>10.3} ms ({:>5.1}%)", ms(*d), pct(*d))?;
    }
    let other = Duration::from_nanos((total.as_nanos().saturating_sub(attributed)) as u64);
    writeln!(
        f,
        "  {:>16}: {:>10.3} ms ({:>5.1}%)",
        "other",
        ms(other),
        pct(other)
    )
}
