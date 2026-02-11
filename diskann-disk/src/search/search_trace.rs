/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Per-query search tracing and profiling for comparing PipeSearch vs UnifiedPipeSearch.
//!
//! Captures two kinds of data:
//! - **Event trace**: Ordered list of search events (submit, complete, expand, etc.)
//!   for side-by-side algorithmic comparison.
//! - **Profile counters**: Cumulative time in each phase (IO poll, IO submit, expand,
//!   PQ distance, queue ops, spin-wait) for identifying bottlenecks.
//!
//! Tracing is opt-in: create a `SearchTrace` and pass it to the search function.
//! When disabled (None), all operations are zero-cost.

use std::time::Instant;

/// A single event in the search trace.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Microseconds since the start of the search.
    pub time_us: u64,
    /// The event kind.
    pub kind: TraceEventKind,
}

/// Kinds of trace events.
#[derive(Debug, Clone)]
pub enum TraceEventKind {
    /// IO submitted for a node. `inflight` is the count AFTER submission.
    Submit { node_id: u32, inflight: usize },
    /// IO completed for a node (data loaded from disk).
    Complete { node_id: u32 },
    /// Node loaded from cache (no IO needed).
    CacheHit { node_id: u32 },
    /// Node expanded: FP distance computed, neighbors discovered.
    Expand {
        node_id: u32,
        fp_distance: f32,
        num_neighbors: u32,
        num_new_candidates: u32,
    },
    /// Node selected from priority queue for submission.
    Select {
        node_id: u32,
        pq_distance: f32,
        queue_position: u32,
    },
    /// Poll returned no completions (spin-wait iteration).
    SpinWait,
    /// Search terminated.
    Done {
        total_hops: u32,
        total_ios: u32,
        total_comparisons: u32,
    },
}

/// Cumulative profiling counters for a single query.
#[derive(Debug, Clone, Default)]
pub struct SearchProfile {
    /// Time spent polling io_uring for completions.
    pub io_poll_us: u64,
    /// Time spent submitting IO requests.
    pub io_submit_us: u64,
    /// Time spent computing full-precision distances.
    pub fp_distance_us: u64,
    /// Time spent computing PQ distances for neighbors.
    pub pq_distance_us: u64,
    /// Time spent on priority queue operations (insert, closest_notvisited).
    pub queue_ops_us: u64,
    /// Time spent in spin-wait (nothing to submit or expand).
    pub spin_wait_us: u64,
    /// Time spent parsing nodes from sector buffers.
    pub parse_node_us: u64,
    /// Number of spin-wait iterations.
    pub spin_wait_count: u64,
    /// Number of IO poll calls.
    pub poll_count: u64,
    /// Number of IO submit calls.
    pub submit_count: u64,
    /// Number of nodes expanded.
    pub expand_count: u64,
    /// Total search wall time.
    pub total_us: u64,
}

/// Per-query search trace collector.
///
/// Create one per query, pass to search functions. After search completes,
/// inspect `events` and `profile` for analysis.
pub struct SearchTrace {
    start: Instant,
    pub events: Vec<TraceEvent>,
    pub profile: SearchProfile,
    phase_start: Option<Instant>,
}

impl SearchTrace {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            events: Vec::with_capacity(256),
            profile: SearchProfile::default(),
            phase_start: None,
        }
    }

    /// Record a trace event with the current timestamp.
    #[inline]
    pub fn event(&mut self, kind: TraceEventKind) {
        let time_us = self.start.elapsed().as_micros() as u64;
        self.events.push(TraceEvent { time_us, kind });
    }

    /// Start timing a phase. Call `end_phase_*` to accumulate the duration.
    #[inline]
    pub fn begin_phase(&mut self) {
        self.phase_start = Some(Instant::now());
    }

    /// End the current phase and add elapsed time to `io_poll_us`.
    #[inline]
    pub fn end_phase_io_poll(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.io_poll_us += start.elapsed().as_micros() as u64;
            self.profile.poll_count += 1;
        }
    }

    #[inline]
    pub fn end_phase_io_submit(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.io_submit_us += start.elapsed().as_micros() as u64;
            self.profile.submit_count += 1;
        }
    }

    #[inline]
    pub fn end_phase_fp_distance(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.fp_distance_us += start.elapsed().as_micros() as u64;
        }
    }

    #[inline]
    pub fn end_phase_pq_distance(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.pq_distance_us += start.elapsed().as_micros() as u64;
        }
    }

    #[inline]
    pub fn end_phase_queue_ops(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.queue_ops_us += start.elapsed().as_micros() as u64;
        }
    }

    #[inline]
    pub fn end_phase_spin_wait(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.spin_wait_us += start.elapsed().as_micros() as u64;
            self.profile.spin_wait_count += 1;
        }
    }

    #[inline]
    pub fn end_phase_parse_node(&mut self) {
        if let Some(start) = self.phase_start.take() {
            self.profile.parse_node_us += start.elapsed().as_micros() as u64;
        }
    }

    #[inline]
    pub fn record_expand(&mut self) {
        self.profile.expand_count += 1;
    }

    /// Finalize the trace, recording total wall time.
    pub fn finish(&mut self) {
        self.profile.total_us = self.start.elapsed().as_micros() as u64;
    }

    /// Print a summary of the profile to stderr (for debugging).
    pub fn print_profile_summary(&self) {
        let p = &self.profile;
        let accounted = p.io_poll_us + p.io_submit_us + p.fp_distance_us
            + p.pq_distance_us + p.queue_ops_us + p.spin_wait_us + p.parse_node_us;
        let other = p.total_us.saturating_sub(accounted);
        eprintln!(
            "Profile: total={}us io_poll={}us({}) io_submit={}us({}) \
             fp_dist={}us pq_dist={}us queue={}us spin={}us({}) parse={}us other={}us | \
             expands={} polls={} submits={}",
            p.total_us,
            p.io_poll_us, p.poll_count,
            p.io_submit_us, p.submit_count,
            p.fp_distance_us,
            p.pq_distance_us,
            p.queue_ops_us,
            p.spin_wait_us, p.spin_wait_count,
            p.parse_node_us,
            other,
            p.expand_count, p.poll_count, p.submit_count,
        );
    }

    /// Print the first N events to stderr (for debugging).
    pub fn print_events(&self, max: usize) {
        for (i, ev) in self.events.iter().enumerate().take(max) {
            eprintln!("  [{:>4}] @{:>6}us {:?}", i, ev.time_us, ev.kind);
        }
        if self.events.len() > max {
            eprintln!("  ... ({} more events)", self.events.len() - max);
        }
    }
}

/// Optional trace wrapper — all methods are no-ops when None.
/// This avoids polluting call sites with `if let Some(trace) = ...`.
pub struct OptionalTrace<'a>(pub Option<&'a mut SearchTrace>);

impl<'a> OptionalTrace<'a> {
    #[inline]
    pub fn event(&mut self, kind: TraceEventKind) {
        if let Some(t) = self.0.as_mut() {
            t.event(kind);
        }
    }

    #[inline]
    pub fn begin_phase(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.begin_phase();
        }
    }

    #[inline]
    pub fn end_phase_io_poll(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_io_poll();
        }
    }

    #[inline]
    pub fn end_phase_io_submit(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_io_submit();
        }
    }

    #[inline]
    pub fn end_phase_fp_distance(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_fp_distance();
        }
    }

    #[inline]
    pub fn end_phase_pq_distance(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_pq_distance();
        }
    }

    #[inline]
    pub fn end_phase_queue_ops(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_queue_ops();
        }
    }

    #[inline]
    pub fn end_phase_spin_wait(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_spin_wait();
        }
    }

    #[inline]
    pub fn end_phase_parse_node(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.end_phase_parse_node();
        }
    }

    #[inline]
    pub fn record_expand(&mut self) {
        if let Some(t) = self.0.as_mut() {
            t.record_expand();
        }
    }
}
