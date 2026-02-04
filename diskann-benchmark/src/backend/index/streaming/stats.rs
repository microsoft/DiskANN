/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::borrow::Cow;

use diskann_benchmark_runner::utils::{fmt::Table, percentiles, MicroSeconds};

use crate::{
    backend::index::{build::BuildStats, result::SearchResults},
    utils::DisplayWrapper,
};

/// Statistics reported for each step in a streaming pipeline.
#[derive(Debug, serde::Serialize)]
pub(crate) enum StreamStats {
    Search(Vec<SearchResults>),
    Insert(BuildStats),
    Replace(BuildStats),
    Delete(GenericStats),
    Maintain {
        drop_deleted: GenericStats,
        release: GenericStats,
    },
}

impl StreamStats {
    /// Returns `true` is `self` is the [`Self::Search`] variant.
    pub(crate) fn is_search(&self) -> bool {
        matches!(self, Self::Search(_))
    }

    /// Returns `true` is `self` is the [`Self::Maintain`] variant.
    pub(crate) fn is_maintain(&self) -> bool {
        matches!(self, Self::Maintain { .. })
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Search(_) => "search",
            Self::Insert(_) => "insert",
            Self::Replace(_) => "replace",
            Self::Delete(_) => "delete",
            Self::Maintain { .. } => "maintain",
        }
    }
}

impl std::fmt::Display for StreamStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Search(results) => {
                write!(f, "{}", DisplayWrapper(results.as_slice()))
            }
            Self::Insert(stats) => {
                write!(f, "{}", stats)
            }
            Self::Replace(stats) => {
                write!(f, "{}", stats)
            }
            Self::Delete(stats) => {
                write!(f, "{}", stats)
            }
            Self::Maintain {
                drop_deleted,
                release,
            } => {
                write!(f, "{}\n\n{}", drop_deleted, release)
            }
        }
    }
}

//////////////////
// GenericStats //
//////////////////

#[derive(Debug, serde::Serialize)]
pub(crate) struct GenericStats {
    kind: Cow<'static, str>,
    total_time: MicroSeconds,
    vectors: usize,
    batches: usize,
    latencies: percentiles::Percentiles<MicroSeconds>,
}

impl GenericStats {
    pub(crate) fn new(
        kind: Cow<'static, str>,
        results: diskann_benchmark_core::build::BuildResults<()>,
    ) -> anyhow::Result<Self> {
        let total_time = results.end_to_end_latency();
        let batches = results.output().len();

        let mut latencies = Vec::new();
        let mut vectors = 0;
        results.take_output().into_iter().for_each(|r| {
            vectors += r.batchsize();
            latencies.push(r.latency);
        });

        Ok(Self {
            kind,
            total_time,
            vectors,
            batches,
            latencies: percentiles::compute_percentiles(&mut latencies)?,
        })
    }
}

impl std::fmt::Display for GenericStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} Time: {}s", self.kind, self.total_time.as_seconds())?;
        writeln!(f, "Vectors Processed: {}", self.vectors)?;
        writeln!(f, "Batches: {}", self.batches)?;
        write!(
            f,
            "{} Latencies:\n  average: {}us\n      p90: {}\n      p99: {}\n\n",
            self.kind, self.latencies.mean, self.latencies.p90, self.latencies.p99,
        )
    }
}

/////////////
// Summary //
/////////////

/// A [`Display`] helper for a collection of [`StreamStats`] results.
#[derive(Debug)]
pub(crate) struct Summary<I>(I);

impl<I> Summary<I> {
    pub(crate) fn new(iter: I) -> Self {
        Self(iter)
    }
}

impl<'a, I> std::fmt::Display for Summary<I>
where
    I: ExactSizeIterator<Item = &'a StreamStats> + Clone + 'a,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = &self.0;

        // Break the results into "search" and everything else.
        //
        // For each of the search operations, count the number of results for each search.
        let search_stages = stats.clone().filter(|stage| stage.is_search()).count();

        // Arithmetic cannot underflow because `search_stages` is strictly upper-bounded
        // by `stats.len()`.
        let other_stages = stats.len() - search_stages;
        if other_stages != 0 {
            let header = [
                "Stage",
                "Operation",
                "Mean Latency (us)",
                "P90 Latency",
                "P99 Latency",
            ];

            let mut table = Table::new(header, other_stages);
            let mut row = 0;
            let mut stage: usize = 0;
            for s in stats.clone() {
                match s {
                    // Search stats are formatter later.
                    StreamStats::Search(_) => {
                        stage += 1;
                        continue;
                    }
                    StreamStats::Insert(stats) | StreamStats::Replace(stats) => {
                        let mut r = table.row(row);
                        r.insert(stage, 0);
                        r.insert(stats.kind, 1);
                        r.insert(format!("{:.1}us", stats.insert_latencies.mean), 2);
                        r.insert(stats.insert_latencies.p90, 3);
                        r.insert(stats.insert_latencies.p99, 4);

                        row += 1;
                        stage += 1;
                    }
                    StreamStats::Delete(stats)
                    | StreamStats::Maintain {
                        drop_deleted: stats,
                        ..
                    } => {
                        let mut r = table.row(row);
                        if s.is_maintain() {
                            r.insert(format!("{}-pre", stage), 0);
                        } else {
                            r.insert(stage, 0);
                        }

                        r.insert(stats.kind.to_string(), 1);
                        r.insert(format!("{:.1}us", stats.latencies.mean), 2);
                        r.insert(stats.latencies.p90, 3);
                        r.insert(stats.latencies.p99, 4);

                        row += 1;
                        if !s.is_maintain() {
                            stage += 1;
                        }
                    }
                }
            }

            table.fmt(f)?;
            writeln!(f)?;
        }

        // Write out the search results.
        if search_stages != 0 {
            let mut stage = 0;
            for s in stats.clone() {
                if let StreamStats::Search(stats) = s {
                    writeln!(f, "Search Stage {}", stage)?;
                    writeln!(f, "{}", DisplayWrapper(&**stats))?;
                }

                if !s.is_maintain() {
                    stage += 1;
                }
            }
        }

        Ok(())
    }
}
