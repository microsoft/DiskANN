/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use diskann::{
    graph::index::{QueryLabelProvider, QueryVisitDecision},
    neighbor::Neighbor,
    utils::VectorId,
};
use diskann_benchmark_runner::files::InputFile;
use diskann_label_filter::{
    kv_index::GenericIndex,
    stores::bftree_store::BfTreeStore,
    traits::{
        posting_list_trait::{PostingList, RoaringPostingList},
        query_evaluator::QueryEvaluator,
    },
    ASTExpr, DefaultKeyCodec,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;
use std::sync::Arc;

pub struct QueryBitmapEvaluator {
    pub ast_expr: ASTExpr,
    evaluated_bitmap: RoaringPostingList,
}

impl QueryBitmapEvaluator {
    /// Create a new filter and evaluate the bitmap immediately (existing behavior).
    pub fn new(
        ast_expr: ASTExpr,
        inverted_index: &GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>,
    ) -> Self {
        let evaluated_bitmap = inverted_index.evaluate_query(&ast_expr).unwrap();
        Self {
            ast_expr,
            evaluated_bitmap,
        }
    }

    /// Ensure evaluated and return a reference to the bitmap (convenience).
    fn get_bitmap(&self) -> &RoaringPostingList {
        &self.evaluated_bitmap
    }

    /// Number of matching labels in this filter's evaluated bitmap.
    pub fn count(&self) -> usize {
        self.get_bitmap().len()
    }
}

impl Debug for QueryBitmapEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapFilter")
            .field("ast_expr", &self.ast_expr)
            .field("evaluated_bitmap", &self.evaluated_bitmap)
            .finish()
    }
}

impl<T> QueryLabelProvider<T> for QueryBitmapEvaluator
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.get_bitmap().contains(vec_id.into_usize())
    }
}

#[derive(Debug)]
pub struct BitmapFilter(pub BitSet);

impl<T> QueryLabelProvider<T> for BitmapFilter
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.0.contains(vec_id.into_usize())
    }
}

/// A bitmap filter wrapper that implements the full 4-layer high selectivity handling:
///
/// **Layer 1: Exploration Queue** - Uses `RejectAndNeedExpand` to enable continued
/// graph traversal through non-matching nodes when the primary queue is exhausted.
///
/// **Layer 2: Match Rate Detection** - Automatically detects high selectivity after
/// 30 samples. If match rate < 2%, enables exploration mode.
///
/// **Layer 3: Checkpoint-Based Timeout** - Checks timeout every 1000 visits instead
/// of every visit to reduce syscall overhead by ~99%.
///
/// **Layer 4: Two-Tier Early Stop** - Soft timeout (default 10ms) triggers when
/// enough matches found; hard timeout (default 100ms) is unconditional.
pub struct HighSelectivityBitmapFilter {
    bitmap: BitSet,
    /// Number of nodes visited
    node_visited: AtomicU64,
    /// Number of nodes matched
    node_matched: AtomicU64,
    /// Cached mode: 0 = undetermined, 1 = high match rate, 2 = low match rate
    need_expand_mode: AtomicU64,
    /// Next checkpoint for timeout/mode checking (avoids checking every visit)
    next_timeout_check: AtomicU64,
    /// Start time for timeout calculations
    start_instant: Instant,
    /// Soft early stop threshold (default 10ms)
    soft_early_stop: Duration,
    /// Hard early stop threshold (default 100ms)
    hard_early_stop: Duration,
    /// Minimum matched count required for soft early stop
    min_matched_count: u64,
    /// Flag indicating if early stop was triggered
    early_stopped: AtomicBool,
}

impl HighSelectivityBitmapFilter {
    /// Minimum samples needed before we can estimate match rate reliably
    const MIN_SAMPLES_FOR_ESTIMATION: u64 = 30;

    /// Match rate threshold for enabling exploration queue (2%)
    const LOW_MATCH_RATE_THRESHOLD: f64 = 0.02;

    /// Check interval for match rate calculation and timeout enforcement
    const MATCH_RATE_CHECK_INTERVAL: u64 = 1000;

    /// Default soft early stop threshold in milliseconds
    const DEFAULT_SOFT_EARLY_STOP_MS: u64 = 10;

    /// Default hard early stop threshold in milliseconds
    const DEFAULT_HARD_EARLY_STOP_MS: u64 = 100;

    /// Default minimum matched count for soft early stop
    const DEFAULT_MIN_MATCHED_COUNT: u64 = 10;

    /// Create a new filter with default timeout settings.
    pub fn new(bitmap: BitSet) -> Self {
        Self::with_config(
            bitmap,
            Duration::from_millis(Self::DEFAULT_SOFT_EARLY_STOP_MS),
            Duration::from_millis(Self::DEFAULT_HARD_EARLY_STOP_MS),
            Self::DEFAULT_MIN_MATCHED_COUNT,
        )
    }

    /// Create a new filter with custom timeout configuration.
    ///
    /// # Arguments
    /// * `bitmap` - The bitmap for filtering
    /// * `soft_early_stop` - Soft timeout (triggers when elapsed > soft AND matched >= min_matched)
    /// * `hard_early_stop` - Hard timeout (unconditional termination)
    /// * `min_matched_count` - Minimum matches required for soft early stop
    pub fn with_config(
        bitmap: BitSet,
        soft_early_stop: Duration,
        hard_early_stop: Duration,
        min_matched_count: u64,
    ) -> Self {
        // Clamp: soft_early_stop should not exceed hard_early_stop
        let soft_early_stop = soft_early_stop.min(hard_early_stop);

        Self {
            bitmap,
            node_visited: AtomicU64::new(0),
            node_matched: AtomicU64::new(0),
            need_expand_mode: AtomicU64::new(0),
            next_timeout_check: AtomicU64::new(Self::MATCH_RATE_CHECK_INTERVAL),
            start_instant: Instant::now(),
            soft_early_stop,
            hard_early_stop,
            min_matched_count,
            early_stopped: AtomicBool::new(false),
        }
    }

    /// Check if we need expansion mode based on current match rate.
    /// Returns: 0 = not determined, 1 = high match rate (no expand), 2 = low match rate (need expand)
    fn check_need_expand_mode(&self, visited: u64, matched: u64) -> u64 {
        // Already determined
        let cached = self.need_expand_mode.load(Ordering::Relaxed);
        if cached != 0 {
            return cached;
        }

        // Need enough samples
        if visited < Self::MIN_SAMPLES_FOR_ESTIMATION {
            return 0;
        }

        let match_rate = if visited > 0 {
            matched as f64 / visited as f64
        } else {
            0.0
        };

        let mode = if match_rate < Self::LOW_MATCH_RATE_THRESHOLD {
            2 // Low match rate, need expand
        } else {
            1 // High match rate, no expand needed
        };

        // Cache the result (compare-and-swap to avoid race)
        let _ = self.need_expand_mode.compare_exchange(
            0,
            mode,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        mode
    }

    /// Check whether early stop should trigger based on two-tier logic:
    /// - Soft early stop: elapsed > soft_early_stop AND matched >= min_matched_count
    /// - Hard early stop: elapsed > hard_early_stop (unconditional)
    fn should_early_stop(&self, matched: u64) -> bool {
        let elapsed = self.start_instant.elapsed();

        // Hard early stop: unconditionally terminate to bound worst-case latency
        if elapsed > self.hard_early_stop {
            return true;
        }

        // Soft early stop: time is past threshold AND enough matched results
        // This ensures we don't terminate too early without enough results
        if elapsed > self.soft_early_stop && matched >= self.min_matched_count {
            return true;
        }

        false
    }

    /// Returns true if early stop was triggered during search.
    #[allow(dead_code)]
    pub fn was_early_stopped(&self) -> bool {
        self.early_stopped.load(Ordering::Relaxed)
    }

    /// Returns the number of nodes visited.
    #[allow(dead_code)]
    pub fn nodes_visited(&self) -> u64 {
        self.node_visited.load(Ordering::Relaxed)
    }

    /// Returns the number of nodes matched.
    #[allow(dead_code)]
    pub fn nodes_matched(&self) -> u64 {
        self.node_matched.load(Ordering::Relaxed)
    }
}

impl Debug for HighSelectivityBitmapFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HighSelectivityBitmapFilter")
            .field("bitmap_len", &self.bitmap.len())
            .field("node_visited", &self.node_visited.load(Ordering::Relaxed))
            .field("node_matched", &self.node_matched.load(Ordering::Relaxed))
            .field(
                "need_expand_mode",
                &self.need_expand_mode.load(Ordering::Relaxed),
            )
            .field("soft_early_stop", &self.soft_early_stop)
            .field("hard_early_stop", &self.hard_early_stop)
            .field("min_matched_count", &self.min_matched_count)
            .field(
                "early_stopped",
                &self.early_stopped.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl<T> QueryLabelProvider<T> for HighSelectivityBitmapFilter
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.bitmap.contains(vec_id.into_usize())
    }

    fn on_visit(&self, neighbor: Neighbor<T>) -> QueryVisitDecision<T> {
        let visited = self.node_visited.fetch_add(1, Ordering::Relaxed) + 1;
        let matched = self.node_matched.load(Ordering::Relaxed);

        // Layer 3: Checkpoint-based timeout checking
        // Only check timeout at checkpoints (every MATCH_RATE_CHECK_INTERVAL visits)
        // to reduce syscall overhead from Instant::now() calls
        let threshold = self.next_timeout_check.load(Ordering::Relaxed);
        if visited >= threshold {
            // Update next checkpoint
            self.next_timeout_check.store(
                visited + Self::MATCH_RATE_CHECK_INTERVAL,
                Ordering::Relaxed,
            );

            // Update need_expand_mode based on current match rate
            self.check_need_expand_mode(visited, matched);

            // Layer 4: Two-tier early stop
            // - Soft: elapsed > soft_early_stop AND matched >= min_matched_count
            // - Hard: elapsed > hard_early_stop (unconditional)
            if self.should_early_stop(matched) {
                self.early_stopped.store(true, Ordering::Relaxed);
                return QueryVisitDecision::Terminate;
            }
        }

        // Evaluate filter match
        if self.is_match(neighbor.id) {
            self.node_matched.fetch_add(1, Ordering::Relaxed);
            QueryVisitDecision::Accept(neighbor)
        } else {
            // Layer 2: Match rate detection for exploration mode
            let mode = self.need_expand_mode.load(Ordering::Relaxed);
            if mode == 0 {
                // Not yet determined: check every time until we have enough samples
                let current_matched = self.node_matched.load(Ordering::Relaxed);
                let updated_mode = self.check_need_expand_mode(visited, current_matched);

                if updated_mode == 2 {
                    // Just determined as low match rate: use exploration queue
                    QueryVisitDecision::RejectAndNeedExpand
                } else {
                    // Still undetermined or high match rate: simple reject
                    QueryVisitDecision::Reject
                }
            } else if mode == 2 {
                // Low match rate confirmed: use exploration queue
                QueryVisitDecision::RejectAndNeedExpand
            } else {
                // High match rate (mode=1): simple reject
                QueryVisitDecision::Reject
            }
        }
    }
}

pub(crate) fn generate_bitmaps(
    query_predicates: &InputFile,
    data_labels: &InputFile,
) -> anyhow::Result<Vec<BitSet>> {
    let bit_maps = match read_labels_and_compute_bitmap(
        data_labels.to_str().unwrap(),
        query_predicates.to_str().unwrap(),
    ) {
        Ok(bit_maps) => bit_maps,
        Err(e) => {
            return Err(e.into());
        }
    };
    Ok(bit_maps)
}

pub(crate) fn setup_filter_strategies<I, S>(
    beta: f32,
    bit_maps: I,
    search_strategy: S,
) -> Vec<BetaFilter<S, u32>>
where
    I: IntoIterator<Item = Arc<dyn QueryLabelProvider<u32>>>,
    S: Clone,
{
    bit_maps
        .into_iter()
        .map(|bit_map| BetaFilter::<S, u32>::new(search_strategy.clone(), bit_map, beta))
        .collect::<Vec<_>>()
}

pub(crate) fn as_query_label_provider(set: BitSet) -> Arc<dyn QueryLabelProvider<u32>> {
    Arc::new(BitmapFilter(set))
}

/// Convert a BitSet to a QueryLabelProvider that dynamically enables exploration mode.
///
/// This enables the exploration queue mechanism for high-selectivity scenarios.
/// The filter tracks match rate and only enables exploration mode when the match
/// rate falls below 2% (after sampling at least 30 nodes).
pub(crate) fn as_high_selectivity_query_label_provider(
    set: BitSet,
) -> Arc<dyn QueryLabelProvider<u32>> {
    Arc::new(HighSelectivityBitmapFilter::new(set))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_bitmap_filter_match() {
        let mut bitset = BitSet::new();
        bitset.insert(1);
        bitset.insert(3);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1u32));
        assert!(filter.is_match(3u32));
        assert!(!filter.is_match(2u32));
        assert!(!filter.is_match(0u32));
    }

    #[test]
    fn test_bitmap_filter_empty() {
        let bitset = BitSet::new();
        let filter = BitmapFilter(bitset);

        assert!(!filter.is_match(0u32));
        assert!(!filter.is_match(10u32));
    }

    #[test]
    fn test_bitmap_filter_large_id() {
        let mut bitset = BitSet::new();
        bitset.insert(1000);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1000u32));
        assert!(!filter.is_match(999u32));
    }

    // -----------------------------------------------------------------------
    // Layer 2: Match Rate Detection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_high_selectivity_filter_low_match_rate() {
        // Create a filter where only 1 out of 100 IDs match (1% match rate)
        let mut bitset = BitSet::new();
        bitset.insert(50); // Only ID 50 matches
        let filter = HighSelectivityBitmapFilter::new(bitset);

        // Visit 40 non-matching nodes (more than MIN_SAMPLES_FOR_ESTIMATION=30)
        for id in 0..40u32 {
            if id == 50 {
                continue;
            }
            let neighbor = Neighbor::new(id, id as f32);
            let decision = filter.on_visit(neighbor);

            // Mode is determined after 30 samples
            // Before that, we check mode each time but it stays 0 until we have enough samples
            // After we have 30 samples with 0% match rate, mode becomes 2
            if filter.node_visited.load(Ordering::Relaxed) >= 30 {
                // After 30 visits with 0% match rate: should return RejectAndNeedExpand
                assert!(
                    matches!(decision, QueryVisitDecision::RejectAndNeedExpand),
                    "After 30 samples with 0% match rate, should return RejectAndNeedExpand, got {:?}",
                    decision
                );
            }
        }

        // Mode should now be 2 (low match rate)
        assert_eq!(
            filter.need_expand_mode.load(Ordering::Relaxed),
            2,
            "Mode should be 2 (low match rate)"
        );
    }

    #[test]
    fn test_high_selectivity_filter_high_match_rate() {
        // Create a filter where half the IDs match (50% match rate, well above 2% threshold)
        let mut bitset = BitSet::new();
        for i in 0..50 {
            bitset.insert(i * 2); // Even IDs match
        }
        let filter = HighSelectivityBitmapFilter::new(bitset);

        // Visit 60 nodes alternating match/no-match
        for id in 0..60u32 {
            let neighbor = Neighbor::new(id, id as f32);
            let decision = filter.on_visit(neighbor);

            if id % 2 == 0 {
                // Even IDs match
                assert!(
                    matches!(decision, QueryVisitDecision::Accept(_)),
                    "Matching nodes should be accepted"
                );
            } else {
                // Odd IDs don't match
                // After 30 samples with 50% match rate: should return Reject (high match rate)
                if filter.node_visited.load(Ordering::Relaxed) >= 30 {
                    assert!(
                        matches!(decision, QueryVisitDecision::Reject),
                        "High match rate should return Reject, got {:?}",
                        decision
                    );
                }
            }
        }

        // Mode should be 1 (high match rate) after enough samples
        assert_eq!(
            filter.need_expand_mode.load(Ordering::Relaxed),
            1,
            "Mode should be 1 (high match rate)"
        );
    }

    #[test]
    fn test_mode_undetermined_with_insufficient_samples() {
        let bitset = BitSet::new(); // No matches
        let filter = HighSelectivityBitmapFilter::new(bitset);

        // Visit only 20 nodes (less than MIN_SAMPLES_FOR_ESTIMATION=30)
        for id in 0..20u32 {
            let neighbor = Neighbor::new(id, id as f32);
            let _ = filter.on_visit(neighbor);
        }

        // Mode should still be 0 (undetermined)
        assert_eq!(
            filter.need_expand_mode.load(Ordering::Relaxed),
            0,
            "Mode should be 0 (undetermined) with < 30 samples"
        );
    }

    // -----------------------------------------------------------------------
    // Layer 3: Checkpoint-Based Timeout Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_interval() {
        let bitset = BitSet::new();
        let filter = HighSelectivityBitmapFilter::new(bitset);

        // Initial checkpoint should be at MATCH_RATE_CHECK_INTERVAL (1000)
        assert_eq!(
            filter.next_timeout_check.load(Ordering::Relaxed),
            HighSelectivityBitmapFilter::MATCH_RATE_CHECK_INTERVAL,
            "Initial checkpoint should be at 1000"
        );

        // Visit 999 nodes - checkpoint should not be updated yet
        for id in 0..999u32 {
            let neighbor = Neighbor::new(id, id as f32);
            let _ = filter.on_visit(neighbor);
        }

        assert_eq!(
            filter.next_timeout_check.load(Ordering::Relaxed),
            HighSelectivityBitmapFilter::MATCH_RATE_CHECK_INTERVAL,
            "Checkpoint should not be updated before reaching threshold"
        );

        // Visit one more node to reach checkpoint
        let neighbor = Neighbor::new(999u32, 999.0);
        let _ = filter.on_visit(neighbor);

        // Checkpoint should now be updated to 2000
        assert_eq!(
            filter.next_timeout_check.load(Ordering::Relaxed),
            2 * HighSelectivityBitmapFilter::MATCH_RATE_CHECK_INTERVAL,
            "Checkpoint should be updated to 2000 after reaching 1000"
        );
    }

    // -----------------------------------------------------------------------
    // Layer 4: Two-Tier Early Stop Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hard_early_stop() {
        // Create filter with very short hard timeout (1ms)
        let bitset = BitSet::new();
        let filter = HighSelectivityBitmapFilter::with_config(
            bitset,
            Duration::from_millis(100), // soft (won't trigger - no matches)
            Duration::from_millis(1),   // hard (very short)
            10,
        );

        // Sleep to ensure we exceed hard timeout
        thread::sleep(Duration::from_millis(5));

        // Visit enough nodes to reach checkpoint (1000)
        let mut terminated = false;
        for id in 0..1500u32 {
            let neighbor = Neighbor::new(id, id as f32);
            let decision = filter.on_visit(neighbor);
            if matches!(decision, QueryVisitDecision::Terminate) {
                terminated = true;
                break;
            }
        }

        assert!(terminated, "Should have terminated due to hard timeout");
        assert!(
            filter.was_early_stopped(),
            "early_stopped flag should be set"
        );
    }

    #[test]
    fn test_soft_early_stop_with_enough_matches() {
        // Create filter with short soft timeout and matching IDs
        let mut bitset = BitSet::new();
        for i in 0..1000 {
            bitset.insert(i); // All IDs match
        }
        let filter = HighSelectivityBitmapFilter::with_config(
            bitset,
            Duration::from_millis(1), // soft (very short)
            Duration::from_millis(1000), // hard (long)
            5,                        // min_matched_count
        );

        // Sleep to ensure we exceed soft timeout
        thread::sleep(Duration::from_millis(5));

        // Visit enough matching nodes to exceed min_matched_count and reach checkpoint
        let mut terminated = false;
        for id in 0..1500u32 {
            let neighbor = Neighbor::new(id, id as f32);
            let decision = filter.on_visit(neighbor);
            if matches!(decision, QueryVisitDecision::Terminate) {
                terminated = true;
                break;
            }
        }

        assert!(
            terminated,
            "Should have terminated due to soft timeout (enough matches)"
        );
        assert!(
            filter.was_early_stopped(),
            "early_stopped flag should be set"
        );
        assert!(
            filter.nodes_matched() >= 5,
            "Should have at least min_matched_count matches"
        );
    }

    #[test]
    fn test_soft_early_stop_not_triggered_without_enough_matches() {
        // Create filter with short soft timeout but NO matches
        let bitset = BitSet::new(); // No matches
        let filter = HighSelectivityBitmapFilter::with_config(
            bitset,
            Duration::from_millis(1),    // soft (very short)
            Duration::from_millis(1000), // hard (long)
            10,                          // min_matched_count (won't be met)
        );

        // Sleep to ensure we exceed soft timeout
        thread::sleep(Duration::from_millis(5));

        // Visit nodes - should NOT terminate because we don't have enough matches
        let mut terminated = false;
        for id in 0..1500u32 {
            let neighbor = Neighbor::new(id, id as f32);
            let decision = filter.on_visit(neighbor);
            if matches!(decision, QueryVisitDecision::Terminate) {
                terminated = true;
                break;
            }
        }

        // Should NOT have terminated - soft timeout requires min_matched_count
        assert!(
            !terminated,
            "Should NOT terminate without enough matches (soft timeout not met)"
        );
        assert!(
            !filter.was_early_stopped(),
            "early_stopped flag should NOT be set"
        );
    }

    #[test]
    fn test_soft_timeout_clamped_to_hard_timeout() {
        // Create filter where soft > hard (should be clamped)
        let bitset = BitSet::new();
        let filter = HighSelectivityBitmapFilter::with_config(
            bitset,
            Duration::from_millis(200), // soft (greater than hard)
            Duration::from_millis(50),  // hard
            10,
        );

        // soft_early_stop should be clamped to hard_early_stop
        assert!(
            filter.soft_early_stop <= filter.hard_early_stop,
            "soft_early_stop ({:?}) should not exceed hard_early_stop ({:?})",
            filter.soft_early_stop,
            filter.hard_early_stop
        );
    }

    // -----------------------------------------------------------------------
    // Default Constants Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_constants() {
        assert_eq!(
            HighSelectivityBitmapFilter::DEFAULT_SOFT_EARLY_STOP_MS,
            10,
            "Default soft timeout should be 10ms"
        );
        assert_eq!(
            HighSelectivityBitmapFilter::DEFAULT_HARD_EARLY_STOP_MS,
            100,
            "Default hard timeout should be 100ms"
        );
        assert_eq!(
            HighSelectivityBitmapFilter::DEFAULT_MIN_MATCHED_COUNT,
            10,
            "Default min matched count should be 10"
        );
        assert_eq!(
            HighSelectivityBitmapFilter::MIN_SAMPLES_FOR_ESTIMATION,
            30,
            "Min samples for estimation should be 30"
        );
        assert!(
            (HighSelectivityBitmapFilter::LOW_MATCH_RATE_THRESHOLD - 0.02).abs() < f64::EPSILON,
            "Low match rate threshold should be 2%"
        );
        assert_eq!(
            HighSelectivityBitmapFilter::MATCH_RATE_CHECK_INTERVAL,
            1000,
            "Match rate check interval should be 1000"
        );
    }
}
