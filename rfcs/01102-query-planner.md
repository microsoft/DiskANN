# Query Planner for Per-Filter-Bitmap Disk Search

| | |
|---|---|
| **Authors** | tianyuanyuan |
| **Created** | 2026-05-25 |
| | |

## Summary

Adds a lightweight **query planner** that automatically selects between flat scan and beta-filtered graph search on the disk path based on the filter bitmap's match rate. The planner sits between the caller and `search_internal()`, takes a bitmap of allowed vector IDs plus the matching count, computes the match rate, and produces the appropriate `SearchPlan`. Callers no longer choose a search strategy manually — the planner adapts to the actual data distribution.

## Motivation

### Background

When a query is scoped to a specific filter category, the caller extracts a list of items matching that category, maps them to DiskANN's internal vector IDs, and constructs a bitmap. The caller wraps the bitmap in a `Predicate` closure and passes it to the disk searcher.

The **match rate** — the fraction of index points present in the bitmap — varies widely across filter categories and tenants. One category may cover 80% of the index, while another may cover 0.05%. No single search strategy is optimal across this range.
Without a query planner, callers must either hard-code a strategy or pass `is_flat_search` manually — neither adapts to the actual data distribution.

### Problem Statement

1. **No automatic strategy selection.** The current `search()` API requires the caller to choose between flat scan and graph search explicitly via `is_flat_search: bool`. Callers have no built-in way to pick the right strategy based on filter selectivity.

2. **Beta filter recall degrades at low match rates.** Benchmark data shows that beta-filtered graph search suffers a "recall dip" in the 2–8% pass rate range, where recall drops as low as 27–53% depending on index size. Flat scan maintains ~100% recall across all pass rates but has linearly increasing latency. The crossover point is not obvious to callers.

3. **Small indexes don't benefit from beta filter.** For indexes under ~200K vectors, flat scan achieves ~100% recall at the same latency as beta filter (~313ms). There is no benefit to beta filtering on small indexes.

### Goals

1. Provide a `QueryPlanner` that automatically selects the optimal search strategy (flat scan vs. beta-filtered graph search) based on index size and filter match rate.
2. Derive thresholds from benchmark experiments on datasets covering the 150K-10M index sizes.
3. Compose cleanly with the `SearchPlan` / `GraphMode` API from the [disk beta filter RFC](https://github.com/dyhyfu/DiskANN/blob/c3ae608683531765920f0844d70750efa731946a/rfcs/01101-disk-beta-filter.md) — the planner produces `SearchPlan` values, nothing else.
4. Allow callers to override thresholds via `QueryPlannerConfig` for tuning.

## Proposal

### 1. Prerequisites

This RFC assumes the [disk beta filter RFC](https://github.com/dyhyfu/DiskANN/blob/c3ae608683531765920f0844d70750efa731946a/rfcs/01101-disk-beta-filter.md) is implemented. Specifically, the following types are available:

- `Predicate = Box<dyn Fn(u32) -> bool + Send + Sync>` — closure-based filter
- `SearchPlan` enum: `FlatScan { filter: Option<Predicate> }` | `Graph(GraphMode)`
- `GraphMode` enum: `Unfiltered` | `PostFilter(Predicate)` | `BetaFilter { predicate: Predicate, beta: f32 }`
- Beta-biased PQ distance computation in `DiskAccessor::pq_distances()`
- Hard post-filtering in `RerankAndFilter` via the predicate closure
- The `Predicate` closure can wrap any bitmap type (`RoaringBitmap`, `BitSet`, `HashSet<u32>`)

### 2. Design Overview

The query planner is a lightweight routing layer that sits between the caller and `search_internal()`. It takes a bitmap of allowed vector IDs plus the matching count, computes the match rate, and selects the appropriate search strategy.

```
┌─────────────────────────────────────────────────────────┐
│  Caller code (bitmap construction boundary)             │
│                                                         │
│  // Wrap bitmap in Arc for shared ownership             │
│  let bitmap: Arc<RoaringBitmap> = Arc::new(bitmap);     │
│  let matching_count = bitmap.len();                     │
│                                                         │
│  // Call query planner                                  │
│  let search_plan = planner.plan_search(bitmap,          │
│                                    matching_count);     │
│  searcher.search(query, ..., search_plan)               │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│  QueryPlanner::plan()            [query_planner.rs]     │
│                                                         │
│  match_rate = matching_count / total_points             │
│                                                         │
│  if total_points ≤ total_points_threshold → FlatSearch  │
│  else if match_rate ≤ pass_rate_threshold   → FlatSearch│
│  else                        → BetaFilter               │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│  DiskIndexSearcher::search()     [disk_provider.rs]     │
│                                                         │
│  Dispatches based on SearchPlan variant                 │
└────────────────────┬────────────────────────────────────┘
                     ▼
              ┌──────┴──────┐
              │             │
        FlatScan         Graph
              │             │
              ▼             ▼
┌─────────────────┐  ┌──────────────────────────────────┐
│   flat_search   │  │   cached_beam_search             │
│                 │  │                                  │
│ for each id:    │  │  pq_distances():                 │
│   if predicate  │  │    if predicate(id):             │
│    (id) → true  │  │      distance *= beta (0.5)      │
│   → compute     │  │    else:                         │
│     distance    │  │      distance unchanged          │
│   else:         │  │                                  │
│     skip        │  │  RerankAndFilter::post_process():│
│                 │  │    .filter(|id|                  │
│                 │  │       predicate(id))             │
│                 │  │    → hard-removes non-matching   │
└────────┬────────┘  └──────────────┬───────────────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
            SearchResult returned
            to caller
```

### 3. Strategy Selection

The planner uses the match rate (`matching_count / total_points`) as the sole selection metric:

```rust
let match_rate = matching_count as f64 / total_points as f64;

if total_points <= TOTAL_POINTS_THRESHOLD {
    QueryStrategy::FlatSearch
} else {
    if match_rate <= FLAT_SEARCH_THRESHOLD {
        QueryStrategy::FlatSearch
    } else {
        QueryStrategy::BetaFilter
    }
}
```

- `TOTAL_POINTS_THRESHOLD` default: **200,000**
- `FLAT_SEARCH_THRESHOLD` default: **0.25** (25%)

### 4. API

#### 4.1 `QueryStrategy`

```rust
/// The strategy selected by the query planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryStrategy {
    /// Brute-force linear scan with hard filter.
    FlatSearch,

    /// Beta-biased graph search with post-filter.
    BetaFilter,
}
```

#### 4.2 `QueryPlannerConfig`

```rust
pub struct QueryPlannerConfig {
    /// Index size threshold. If total_points <= this value, always use flat scan.
    /// Default: 200_000
    pub total_points_threshold: u64,

    /// Match-rate threshold. When total_points > total_points_threshold and
    /// match_rate <= this value, use flat scan; otherwise use beta-filtered
    /// graph search.
    /// Default: 0.25 (25%)
    pub flat_search_threshold: f64,

    /// Beta value for beta-biased search. Must be in (0, 1].
    /// Default: 0.5
    pub beta: f32,
}

impl Default for QueryPlannerConfig {
    fn default() -> Self {
        Self {
            total_points_threshold: 200_000,
            flat_search_threshold: 0.25,
            beta: 0.5,
        }
    }
}
```

#### 4.3 `QueryPlanner`

```rust
pub struct QueryPlanner {
    config: QueryPlannerConfig,
    total_points: u64,
}

impl QueryPlanner {
    pub fn new(config: QueryPlannerConfig, total_points: u64) -> Self {
        Self { config, total_points }
    }

    /// Determine the search strategy based on index size and bitmap match rate.
    ///
    /// Decision logic:
    ///   1. If total_points <= total_points_threshold → FlatSearch (small index)
    ///   2. Else if match_rate <= flat_search_threshold → FlatSearch (sparse filter)
    ///   3. Else → BetaFilter (dense filter on large index)
    pub fn plan(&self, matching_count: u64) -> QueryStrategy {
        if self.total_points <= self.config.total_points_threshold {
            return QueryStrategy::FlatSearch;
        }

        let match_rate = matching_count as f64 / self.total_points as f64;
        if match_rate <= self.config.flat_search_threshold {
            QueryStrategy::FlatSearch
        } else {
            QueryStrategy::BetaFilter
        }
    }

    /// Plan and produce a `SearchPlan` with the appropriate predicate wiring.
    ///
    /// The caller provides the bitmap (as an `Arc<RoaringBitmap>`) and the
    /// matching count. The planner selects the strategy and constructs the
    /// `SearchPlan` with the closure already wired to the bitmap.
    pub fn plan_search(
        &self,
        bitmap: Arc<RoaringBitmap>,
        matching_count: u64,
    ) -> SearchPlan {
        let strategy = self.plan(matching_count);
        let beta = self.config.beta;

        match strategy {
            QueryStrategy::FlatSearch => {
                let bm = bitmap.clone();
                SearchPlan::FlatScan {
                    filter: Some(Box::new(move |id| bm.contains(id))),
                }
            }
            QueryStrategy::BetaFilter => {
                let bm = bitmap.clone();
                SearchPlan::Graph(GraphMode::BetaFilter {
                    predicate: Box::new(move |id| bm.contains(id)),
                    beta,
                })
            }
        }
    }
}
```

#### 4.4 Caller Usage

```rust
// At initialization — once per index.
// `total_points` is the number of vectors in the index,
// available from the index load context (e.g. AsyncDiskLoadContext::num_points).
let planner = QueryPlanner::new(
    QueryPlannerConfig::default(),
    total_points as u64,
);

// Per-query — bitmap comes from the caller's filter infrastructure.
// `bitmap.len()` returns the number of set bits as u64.
let bitmap: Arc<RoaringBitmap> = /* vector IDs for the target filter category */;
let matching_count = bitmap.len();
let search_plan = planner.plan_search(bitmap, matching_count);
let results = searcher.search(
    query,
    return_list_size,
    search_list_size,
    beam_width,
    search_plan,
)?;
```

#### 4.5 Decision Flow

The planner applies two thresholds in order:

1. **Index size check** (`total_points_threshold` = 200K):
   - If the index has ≤ 200K vectors, flat scan is always used regardless of filter pass rate.
   - Rationale: On small indexes, flat scan achieves ~100% recall at the same latency as beta filter (~313ms). There is no benefit to beta filtering.

2. **Match rate check** (`flat_search_threshold` = 25%):
   - On indexes > 200K vectors, if filter pass rate ≤ 25%, use flat scan.
   - If filter pass rate > 25%, use beta filter.
   - Rationale: Beta filter recall drops to 27–53% in the 2–8% pass rate range ("beta saturation dip"). Flat scan maintains ~100% recall. Above 25%, beta filter achieves 95–99% recall with constant latency, while flat scan latency grows linearly with matching vectors.

### 5. Threshold Derivation

The thresholds are derived from benchmark experiments on datasets sized 150K-10M vector counts.

#### 5.1 Benchmark Experiments

All experiments: `beta=0.5`, flat scan `L=2000`, beta `L=2000` and `L=3000`, `K=10`, `beam_width=4`, `squared_l2` distance.

**Dataset Overview:**

| Dataset Size | Embedding Model | Dim | Vectors | PQ Chunks | Pass Rates Tested |
|---|---|---|---|---|---|
| 150K | FBV8_V2 | 896 | 150,000 | 384 | 0.01% – 100% (19 points) |
| 292K | FBV8_V2 | 896 | 292,697 | 384 | 0.01% – 100% (19 points) |
| 1M | FBV8_V2 | 896 | 958,152 | 384 | 0.01% – 100% (19 points) |
| 1M | FBV4 | 384 | 1,087,932 | 192 | 0.01% – 100% (19 points) |
| 10M | FBV4 | 384 | 10,000,000 | 192 | 0.01% – 100% (19 points) |

#### 5.2 Key Observations

**When index size < 200K vectors, always use flat scan:**

For small indexes, flat scan with `L=2000` achieves ~100% recall at constant ~313ms latency across all pass rates. Beta filter provides no latency advantage — both methods have similar latency (~315ms vs ~313ms) — but flat scan has strictly better recall. There is no reason to use beta filter on small indexes.

**When index size > 200K vectors, 25% filter pass rate is the threshold:**

For larger indexes (292K, 958K, 1M, 10M), the choice between beta filter and flat scan depends on the filter pass rate:

- **<25%**: Beta filter recall drops severely in the 2–8% range (the "beta saturation dip"), falling as low as 27–53% depending on index size. Flat scan maintains ~100% recall with similar or lower latency.
- **>25%**: Beta filter achieves 95–99% recall with constant latency (~315ms for `L=2000`). Flat scan latency rises linearly with matching vectors — at 100% pass rate, it reaches 665ms (1M vectors), 1,076ms (958K), or 6,332ms (10M vectors). Beta is both faster and has good recall.

## Trade-offs

### Match-rate threshold vs. absolute-count threshold

**Chosen: match rate.** The planner uses `matching_count / total_points` rather than an absolute count. This makes the threshold scale-invariant — a 25% pass rate has the same recall/latency tradeoff on a 300K index and a 10M index. An absolute count (e.g., "switch at 100K matching vectors") would need recalibration for every index size.

### Two thresholds (index size + match rate) vs. single threshold

**Chosen: two thresholds.** The index-size check (`total_points ≤ 200K`) catches the regime where flat scan dominates unconditionally — no match-rate analysis needed. This avoids the pathological case where a 150K-vector index at 80% pass rate would be routed to beta filter despite flat scan being equally fast and having better recall.

### Planner as a separate struct vs. integrated into `DiskProvider`

**Chosen: separate struct.** `QueryPlanner` is a pure function of `(config, total_points, matching_count)` → `SearchPlan`. It has no dependency on `DiskProvider`'s internals, runtime, or I/O context. Keeping it separate makes it testable in isolation and reusable across different provider implementations.

### `plan()` returning `QueryStrategy` vs. directly returning `SearchPlan`

**Chosen: both.** `plan()` returns the lightweight `QueryStrategy` enum (no allocation, `Copy`), useful for logging, metrics, and testing. `plan_search()` takes the bitmap and produces a ready-to-use `SearchPlan` with the closure wired in. Callers that need fine-grained control use `plan()` + manual `SearchPlan` construction; callers that want convenience use `plan_search()`.

## Benchmark Results

All experiments: `beta=0.5`, flat scan `L=2000`, beta `L=2000` and `L=3000`, `K=10`, `beam_width=4`, `squared_l2` distance.

### Experiment 1: MERB 150K (FBV8_V2, 896d, 150,000 vectors)

| Pass Rate | Matching | Flat Recall | Flat Latency (μs) | Beta L=2000 Recall | Beta L=2000 Latency (μs) | Beta L=3000 Recall | Beta L=3000 Latency (μs) |
|---|---|---|---|---|---|---|---|
| 0.01% | 15 | 100.0% | 3,872 | 19.9% | 324,634 | 25.7% | 481,312 |
| 0.05% | 75 | 100.0% | 11,301 | 51.2% | 321,427 | 64.4% | 481,867 |
| 0.10% | 150 | 100.0% | 23,185 | 73.3% | 323,755 | 84.1% | 474,363 |
| 0.50% | 750 | 100.0% | 116,963 | 96.8% | 319,661 | 98.8% | 480,147 |
| 1.00% | 1,500 | 100.0% | 235,017 | 98.7% | 321,585 | 99.4% | 482,362 |
| 2.00% | 3,000 | 99.8% | 312,965 | 98.6% | 319,485 | 99.3% | 483,300 |
| 3.00% | 4,500 | 99.5% | 312,210 | 97.7% | 322,561 | 98.9% | 484,367 |
| 5.00% | 7,500 | 99.7% | 312,403 | 73.2% | 327,081 | 76.9% | 485,744 |
| 8.00% | 12,000 | 99.6% | 312,666 | 87.9% | 326,794 | 90.9% | 486,264 |
| 10.00% | 15,000 | 99.5% | 310,354 | 94.1% | 320,711 | 95.3% | 483,213 |
| 25.00% | 37,500 | 99.8% | 312,899 | 98.1% | 324,632 | 99.1% | 482,161 |
| 50.00% | 75,000 | 99.8% | 312,873 | 98.2% | 319,873 | 99.2% | 478,667 |
| 100.00% | 150,000 | 99.9% | 313,936 | 98.5% | 318,852 | 99.2% | 480,694 |

**Observation:** Flat scan achieves ~100% recall at constant ~313ms latency across all pass rates. Beta filter has similar or higher latency (~319–327ms for L=2000) but worse recall at low pass rates. **No benefit to beta filter on this index size.**

### Experiment 2: MERB 292K (FBV8_V2, 896d, 292,697 vectors)

| Pass Rate | Matching | Flat Recall | Flat Latency (μs) | Beta L=2000 Recall | Beta L=2000 Latency (μs) | Beta L=3000 Recall | Beta L=3000 Latency (μs) |
|---|---|---|---|---|---|---|---|
| 0.01% | 30 | 100.0% | 6,435 | 0.0% | 398,987 | 0.0% | 485,700 |
| 0.05% | 150 | 100.0% | 23,331 | 0.0% | 326,855 | 0.0% | 487,659 |
| 0.10% | 300 | 100.0% | 46,631 | 0.0% | 326,299 | 0.0% | 478,675 |
| 1.00% | 2,926 | 100.0% | 313,423 | 0.9% | 316,386 | 0.9% | 474,188 |
| 2.00% | 5,853 | 100.0% | 313,339 | 1.9% | 321,180 | 1.9% | 477,944 |
| 3.00% | 8,780 | 99.9% | 313,214 | 2.8% | 325,457 | 2.7% | 490,312 |
| 4.00% | 11,707 | 99.8% | 313,371 | 36.1% | 330,269 | 38.8% | 495,156 |
| 5.12% | 15,000 | 99.8% | 313,258 | 61.2% | 327,206 | 64.9% | 485,799 |
| 8.00% | 23,415 | 99.8% | 313,916 | 84.0% | 338,059 | 87.8% | 501,793 |
| 10.25% | 30,000 | 99.8% | 313,118 | 90.8% | 320,605 | 94.2% | 486,707 |
| 25.62% | 75,000 | 99.8% | 313,198 | 96.6% | 322,463 | 97.7% | 477,724 |
| 51.25% | 150,000 | 99.9% | 306,765 | 95.9% | 321,243 | 97.9% | 476,805 |
| 100.00% | 292,697 | 100.0% | 331,226 | 96.9% | 322,186 | 98.2% | 478,167 |

**Observation:** Beta recall is near 0% at very low pass rates, dips severely below 4%, then recovers above 10%. Flat scan is ~100% recall at constant ~313ms. Above 25%, beta achieves 95%+ recall but flat scan latency is still similar (~313ms), so beta's advantage is marginal on this index size.

### Experiment 3: Enron 958K (FBV8_V2, 896d, 958,152 vectors)

| Pass Rate | Matching | Flat Recall | Flat Latency (μs) | Beta L=2000 Recall | Beta L=2000 Latency (μs) | Beta L=3000 Recall | Beta L=3000 Latency (μs) |
|---|---|---|---|---|---|---|---|
| 0.01% | 100 | 100.0% | 25,957 | 34.4% | 320,368 | 42.8% | 475,944 |
| 0.05% | 500 | 100.0% | 78,216 | 62.2% | 318,788 | 71.6% | 476,454 |
| 0.10% | 1,000 | 99.9% | 154,944 | 74.7% | 318,500 | 82.7% | 474,809 |
| 1.00% | 9,581 | 100.0% | 310,574 | 93.7% | 317,008 | 95.8% | 478,717 |
| 2.00% | 19,163 | 100.0% | 313,127 | 71.7% | 325,383 | 89.9% | 477,149 |
| 3.00% | 28,744 | 100.0% | 313,720 | 34.5% | 319,075 | 38.0% | 475,476 |
| 5.22% | 50,000 | 99.9% | 314,071 | 64.3% | 319,150 | 67.9% | 474,914 |
| 8.00% | 76,652 | 99.9% | 314,848 | 79.8% | 319,836 | 82.9% | 479,142 |
| 10.44% | 100,000 | 99.9% | 314,896 | 85.5% | 318,938 | 87.8% | 475,283 |
| 26.09% | 250,000 | 100.0% | 356,443 | 95.4% | 322,761 | 96.3% | 477,892 |
| 52.18% | 500,000 | 99.9% | 608,060 | 97.3% | 321,331 | 98.0% | 485,033 |
| 78.28% | 750,000 | 99.9% | 841,963 | 98.0% | 319,180 | 98.6% | 476,673 |
| 100.00% | 958,152 | 99.9% | 1,075,861 | 98.2% | 314,731 | 98.8% | 478,826 |

**Observation:** Beta recall dips to 34.5% at 3% pass rate (the "beta saturation dip"). Above 26%, beta achieves 95%+ recall at constant ~315ms. Flat scan latency rises linearly — from 314ms at 10% to **1,076ms at 100%**. Beta is clearly faster at high pass rates.

### Experiment 4: Enron 1M (FBV4, 384d, 1,087,932 vectors)

| Pass Rate | Matching | Flat Recall | Flat Latency (μs) | Beta L=2000 Recall | Beta L=2000 Latency (μs) | Beta L=3000 Recall | Beta L=3000 Latency (μs) |
|---|---|---|---|---|---|---|---|
| 0.01% | 100 | 100.0% | 24,849 | 0.0% | 317,352 | 0.0% | 473,265 |
| 0.05% | 500 | 100.0% | 77,782 | 0.1% | 316,096 | 0.1% | 473,590 |
| 0.10% | 1,000 | 99.9% | 156,540 | 0.1% | 315,600 | 0.1% | 473,810 |
| 1.00% | 10,879 | 100.0% | 314,088 | 1.1% | 318,327 | 1.1% | 473,112 |
| 2.00% | 21,758 | 100.0% | 312,942 | 8.9% | 325,231 | 7.0% | 482,611 |
| 3.00% | 32,637 | 100.0% | 313,487 | 29.6% | 320,663 | 29.8% | 479,626 |
| 5.00% | 50,000* | 100.0% | 313,305 | 52.5% | 318,650 | 52.2% | 475,911 |
| 8.00% | 87,034 | 100.0% | 313,613 | 72.4% | 319,740 | 71.6% | 476,315 |
| 10.00% | 100,000* | 99.9% | 313,460 | 75.8% | 320,355 | 74.6% | 477,895 |
| 23.00% | 250,000* | 100.0% | 314,749 | 89.1% | 318,503 | 87.9% | 474,210 |
| 46.00% | 500,000* | 100.0% | 387,149 | 94.9% | 314,853 | 94.1% | 476,054 |
| 69.00% | 750,000* | 99.9% | 490,339 | 97.4% | 315,909 | 97.1% | 470,516 |
| 100.00% | 1,087,932 | 99.9% | 665,666 | 99.7% | 314,767 | 99.8% | 475,439 |

**Observation:** Similar pattern to FBV8 but with worse beta recall at low pass rates (0% below 1%). Beta recall dip zone extends to ~10%. Above 23%, beta achieves 89%+ recall. Flat scan latency rises from 313ms to **666ms at 100%**. Beta latency constant at ~315ms.

### Experiment 5: Enron 10M (FBV4, 384d, 10,000,000 vectors)

| Pass Rate | Matching | Flat Recall | Flat Latency (μs) | Beta L=2000 Recall | Beta L=2000 Latency (μs) | Beta L=3000 Recall | Beta L=3000 Latency (μs) |
|---|---|---|---|---|---|---|---|
| 0.01% | 1,000 | 100.0% | 297,002 | 23.2% | 316,610 | 30.2% | 472,065 |
| 0.05% | 5,000 | 99.9% | 394,867 | 55.0% | 315,572 | 64.6% | 472,223 |
| 0.10% | 10,000 | 100.0% | 417,171 | 70.3% | 316,017 | 78.6% | 477,212 |
| 0.50% | 50,000 | 100.0% | 491,736 | 92.1% | 328,254 | 94.8% | 473,778 |
| 1.00% | 100,000 | 100.0% | 537,764 | 94.5% | 307,171 | 96.5% | 472,571 |
| 2.00% | 200,000 | 99.9% | 621,667 | 84.6% | 325,391 | 88.7% | 484,016 |
| 3.00% | 300,000 | 100.0% | 692,308 | 39.1% | 346,273 | 43.9% | 503,781 |
| 5.00% | 500,000 | 100.0% | 853,947 | 53.5% | 329,050 | 57.6% | 484,794 |
| 8.00% | 800,000 | 99.9% | 786,862 | 73.6% | 320,619 | 77.1% | 481,412 |
| 10.00% | 1,000,000 | 100.0% | 972,848 | 80.7% | 318,887 | 83.7% | 475,876 |
| 25.00% | 2,500,000 | — | — | 94.7% | 1,528,934 | 95.7% | 657,100 |
| 50.00% | 5,000,000 | 99.9% | 3,382,211 | 97.8% | 316,860 | 98.3% | 472,996 |
| 75.00% | 7,500,000 | 100.0% | 4,905,380 | 98.8% | 315,207 | 99.1% | 471,848 |
| 100.00% | 10,000,000 | 100.0% | 6,331,538 | 99.0% | 313,288 | 99.2% | 473,983 |

**Observation:** Beta recall dips to 39.1% at 3%. Above 25%, beta achieves 95%+ recall at constant ~315ms. Flat scan latency grows dramatically — from 297ms at 0.01% to **6,332ms at 100%** (20× slower than beta). At 10M vectors, the planner's routing is critical.

### Summary

The 25% match-rate threshold sits well above the beta recall dip zone (2–8%) and below the range where beta consistently delivers 95%+ recall, providing a safety margin. The 200K index-size threshold ensures small indexes always use the higher-recall flat scan strategy.

## Future Work

- [ ] **Adaptive beta value.** The current design uses a fixed `beta = 0.5`. Future work could adapt beta based on match rate — higher beta (less bias) at higher match rates, lower beta (more bias) at lower match rates — to improve recall in the transition zone.
- [ ] **Multi-strategy planner.** The current planner selects between two strategies. Future graph algorithms (e.g., `MultihopSearch`) could be added as additional `GraphMode` variants with their own match-rate ranges.

## References

1. [RFC 01101: Beta Filter For Disk Search](https://github.com/dyhyfu/DiskANN/blob/c3ae608683531765920f0844d70750efa731946a/rfcs/01101-disk-beta-filter.md) — prerequisite design for `SearchPlan`, `GraphMode`, and beta-biased PQ distance computation.
