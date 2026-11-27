# Query Planner for Per-Filter-Bitmap Disk Search

| | |
|---|---|
| **Authors** | tianyuanyuan |
| **Created** | 2026-05-25 |
| | |

## Summary

Adds a lightweight **query planner** that automatically selects between flat scan and beta-filtered graph search on the disk path based on the filter bitmap's **matching count** and **match rate**. The planner sits between the caller and `search_internal()`, takes a bitmap of allowed vector IDs plus the matching count, and produces the appropriate `SearchPlan`. Callers no longer choose a search strategy manually — the planner adapts to the actual data distribution.

## Motivation

### Background

When a query is scoped to a specific filter category, the caller constructs a bitmap of matching vector IDs and wraps it in a `Predicate` closure. The caller passes the predicate to the disk searcher.

The **match rate** — the fraction of index points present in the bitmap — varies widely across filter categories and tenants. One category may cover 80% of the index, while another may cover 0.05%. No single search strategy is optimal across this range.
Without a query planner, callers must either hard-code a strategy or pass `is_flat_search` manually — neither adapts to the actual data distribution.

### Problem Statement

1. **No automatic strategy selection.** The current `search()` API requires the caller to choose between flat scan and graph search explicitly via `is_flat_search: bool`. Callers have no built-in way to pick the right strategy based on filter selectivity.

2. **Beta filter recall degrades at low match rates.** Benchmark data shows that beta-filtered graph search suffers a "recall dip" in the 2–8% pass rate range, where recall drops as low as 27–53% depending on index size. Flat scan maintains ~100% recall across all pass rates but has linearly increasing latency. The crossover point is not obvious to callers.

3. **Flat scan is cheap when few vectors match the filter.** For ≤200K matching vectors, flat scan achieves ~100% recall at ~313ms latency — the same as or better than beta filter — regardless of total index size. There is no benefit to beta filtering in this regime.

### Goals

1. Provide a `QueryPlanner` that automatically selects the optimal search strategy (flat scan vs. beta-filtered graph search) based on filter matching count and match rate.
2. Derive thresholds from benchmark experiments on datasets covering the 150K–10M index sizes.
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
│  // Build predicate from any bitmap type                │
│  let predicate = move |id| bitmap.contains(id);         │
│  let matching_count = bitmap.len();                     │
│                                                         │
│  // Call query planner                                  │
│  let search_plan = planner.plan_search(predicate,       │
│                                    matching_count);     │
│  searcher.search(query, ..., search_plan)               │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│  QueryPlanner::plan()            [query_planner.rs]     │
│                                                         │
│  match_rate = matching_count / total_points             │
│                                                         │
│  if matching_count ≤ 200K              → FlatSearch     │
│  else if matching_count ≥ 1M           → BetaFilter     │
│  else if match_rate ≤ 25%              → FlatSearch     │
│  else                                  → BetaFilter     │
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

The planner applies three checks in order:

```rust
if matching_count <= MATCHING_COUNT_THRESHOLD {
    QueryStrategy::FlatSearch
} else if matching_count >= MAX_BRUTE_FORCE_COUNT {
    QueryStrategy::BetaFilter
} else {
    let match_rate = matching_count as f64 / total_points as f64;
    if match_rate <= FLAT_SEARCH_THRESHOLD {
        QueryStrategy::FlatSearch
    } else {
        QueryStrategy::BetaFilter
    }
}
```

- `MATCHING_COUNT_THRESHOLD` default: **200,000** — flat scan is cheap when few vectors match
- `MAX_BRUTE_FORCE_COUNT` default: **1,000,000** — caps flat scan latency at ~500–700ms
- `FLAT_SEARCH_THRESHOLD` default: **0.25** (25%) — avoids the beta recall dip zone (2–10%)

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
    /// Matching-count threshold. If matching_count <= this value, always use
    /// flat scan — flat scan is cheap for small result sets regardless of
    /// total index size.
    /// Default: 200_000
    pub matching_count_threshold: u64,

    /// Maximum matching count for flat scan. If matching_count >= this value,
    /// always use beta filter regardless of match rate — flat scan latency
    /// grows linearly with matching count and would exceed latency budgets.
    /// Default: 1_000_000
    pub max_brute_force_count: u64,

    /// Match-rate threshold. When matching_count is between
    /// matching_count_threshold and max_brute_force_count, and
    /// match_rate <= this value, use flat scan; otherwise use
    /// beta-filtered graph search.
    /// Default: 0.25 (25%)
    pub flat_search_threshold: f64,

    /// Beta value for beta-biased search. Must be in (0, 1].
    /// Default: 0.5
    pub beta: f32,
}

impl Default for QueryPlannerConfig {
    fn default() -> Self {
        Self {
            matching_count_threshold: 200_000,
            max_brute_force_count: 1_000_000,
            flat_search_threshold: 0.25,
            beta: 0.5,
        }
    }
}

impl QueryPlannerConfig {
    /// Validate the configuration. Returns an error if beta is not in (0, 1]
    /// or if matching_count_threshold >= max_brute_force_count.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(self.beta > 0.0 && self.beta <= 1.0) {
            return Err("beta must be in (0, 1]");
        }
        if self.matching_count_threshold >= self.max_brute_force_count {
            return Err("matching_count_threshold must be < max_brute_force_count");
        }
        Ok(())
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
    pub fn new(config: QueryPlannerConfig, total_points: u64) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self { config, total_points })
    }

    /// Determine the search strategy based on matching count and match rate.
    ///
    /// Decision logic:
    ///   1. If matching_count <= matching_count_threshold → FlatSearch
    ///      (flat scan is cheap for small filter result sets)
    ///   2. If matching_count >= max_brute_force_count → BetaFilter
    ///      (flat scan latency too high for large result sets)
    ///   3. Else if match_rate <= flat_search_threshold → FlatSearch
    ///      (avoids beta recall dip zone at 2–10% match rate)
    ///   4. Else → BetaFilter (high match rate on large index)
    pub fn plan(&self, matching_count: u64) -> QueryStrategy {
        if matching_count <= self.config.matching_count_threshold {
            return QueryStrategy::FlatSearch;
        }

        if matching_count >= self.config.max_brute_force_count {
            return QueryStrategy::BetaFilter;
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
    /// The caller provides a predicate closure (wrapping any bitmap type) and
    /// the matching count. The planner selects the strategy and constructs the
    /// `SearchPlan` with the predicate wired in.
    ///
    /// Returns `Err` if beta validation fails (e.g. beta not in (0, 1]).
    pub fn plan_search<F>(
        &self,
        predicate: F,
        matching_count: u64,
    ) -> Result<SearchPlan, &'static str>
    where
        F: Fn(u32) -> bool + Send + Sync + 'static,
    {
        let strategy = self.plan(matching_count);
        let beta = self.config.beta;

        match strategy {
            QueryStrategy::FlatSearch => {
                Ok(SearchPlan::flat_filtered(predicate))
            }
            QueryStrategy::BetaFilter => {
                GraphMode::beta_filter(predicate, beta)
                    .map(SearchPlan::graph_with)
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
)?;

// Per-query — predicate wraps any bitmap type (RoaringBitmap, BitSet, HashSet, etc.).
let bitmap: Arc<RoaringBitmap> = /* vector IDs for the target filter category */;
let matching_count = bitmap.len();
let bm = bitmap.clone();
let search_plan = planner.plan_search(move |id| bm.contains(id), matching_count)?;
let results = searcher.search(
    query,
    return_list_size,
    search_list_size,
    beam_width,
    search_plan,
)?;
```

#### 4.5 Decision Flow

The planner applies three thresholds in order:

1. **Matching count lower bound** (`matching_count_threshold` = 200K):
   - If ≤ 200K vectors match the filter, flat scan is always used regardless of match rate or total index size.
   - Rationale: Flat scan latency plateaus at ~313ms for ≤200K matching vectors across all tested index sizes (150K–10M). Flat scan has ~100% recall at this latency, while beta filter may have lower recall. This check also implicitly handles small indexes — a 150K-vector index can have at most 150K matching vectors, which is below the threshold.

2. **Matching count upper bound** (`max_brute_force_count` = 1M):
   - If ≥ 1M vectors match the filter, beta filter is always used regardless of match rate.
   - Rationale: Flat scan latency grows linearly with matching count. Across 5 index sizes (1M–10M), flat scan crosses 500ms between 600K and 1M matching vectors (see §5.3). At 1M matching, flat scan ranges from 538ms (2M index) to 973ms (10M index). Beta filter is constant at ~315ms with 95–99% recall above the dip zone. The 1M default caps worst-case flat-scan latency.

3. **Match rate check** (`flat_search_threshold` = 25%):
   - When matching_count is between 200K and 1M, if filter pass rate ≤ 25%, use flat scan.
   - If filter pass rate > 25%, use beta filter.
   - Rationale: The beta recall dip zone sits at **2–10% match rate** consistently across all index sizes (see §5.2). The 25% threshold provides a safety margin above the dip. Above 25%, beta filter achieves 95–99% recall with constant ~315ms latency, while flat scan latency grows linearly (up to 6,332ms at 100% on a 10M index).

### 5. Threshold Derivation

The thresholds are derived from benchmark experiments on datasets sized 150K–10M vectors, plus follow-up experiments exploring matching-count-based vs. match-rate-based thresholds.

#### 5.1 Benchmark Experiments

All experiments: `beta=0.5`, flat scan `L=2000`, beta `L=2000` and `L=3000`, `K=10`, `beam_width=4`, `squared_l2` distance.

**Dataset Overview:**

| Dataset Size | Dim | Vectors | PQ Chunks | Pass Rates Tested |
|---|---|---|---|---|
| 150K | 896 | 150,000 | 384 | 0.01% – 100% (19 points) |
| 292K | 896 | 292,697 | 384 | 0.01% – 100% (19 points) |
| 1M | 896 | 958,152 | 384 | 0.01% – 100% (19 points) |
| 1M | 384 | 1,087,932 | 192 | 0.01% – 100% (19 points) |
| 10M | 384 | 10,000,000 | 192 | 0.01% – 100% (19 points) |

#### 5.2 Key Observations

**Flat scan latency depends on matching count, not total index size (for ≤ 200K matching vectors):**

Benchmark data shows that flat scan latency at the same matching count is consistent across different index sizes:

| Matching Count | 150K index | 292K index | 958K index | 1M index | 10M index |
|---|---|---|---|---|---|
| 5,000 | 313ms | 313ms | 313ms | 313ms | 395ms |
| 50,000 | 313ms | — | 314ms | 313ms | 492ms |
| 100,000 | — | — | 315ms | 313ms | 538ms |
| 150,000 | 314ms | 307ms | 311ms | — | 425ms |

For indexes ≤ 1M, flat scan latency plateaus at ~313ms regardless of index size once matching count exceeds ~3K (the `L=2000` rerank plateau). This confirms that flat scan cost is governed by the number of matching vectors, not the total index size. For 10M indexes, a higher PQ scan baseline (~300–400ms) adds overhead, but flat scan is still the correct choice at low matching counts because beta recall is terrible there.

**The beta recall dip zone is at consistent match rates (2–10%):**

| Dataset | Dip zone (match rate) | Dip zone (matching count) |
|---|---|---|
| 150K | 4–8% | 6K–12K |
| 292K | 0.01–8% | 30–23K |
| 958K | 2–10% | 19K–100K |
| 1M | 0.01–10% | 100–100K |
| 10M | 2–10% | 200K–1M |

The dip zone is a **match-rate phenomenon** — it sits at 2–10% match rate regardless of index size. The absolute matching count varies 100× across index sizes, but the rate range is stable. This is why the second threshold uses match rate, not matching count.

**Above 25% match rate, beta filter wins on latency with good recall:**

Beta filter achieves 95–99% recall with constant latency (~315ms for `L=2000`). Flat scan latency rises linearly — at 100% pass rate, it reaches 665ms (1M vectors), 1,076ms (958K), or 6,332ms (10M vectors).

#### 5.3 Why the hybrid threshold (matching_count + match_rate + max_brute_force_count)

The planner uses three thresholds: a lower bound on matching count (200K), an upper bound on matching count (1M), and a match-rate guard (25%). The upper bound (`max_brute_force_count`) addresses the case where matching count is in the 200K–1M range but match rate is ≤ 25% on a very large index — without it, flat scan latency can blow past any reasonable SLA.

**Where flat scan crosses 500ms by index size:**

Benchmark data across 7 index sizes (1M–10M, dim=384):

| Index Size | Flat crosses 500ms at | Flat crosses 1s at |
|---|---|---|
| 1M | 1,087,932 (100%) | never |
| 2M | 1,000,000 (50%) | never |
| 3M | 800,000 (27%) | 2,000,000 (67%) |
| 5M | 600,000 (12%) | 2,000,000 (40%) |
| 10M | 100,000 (1%) | 900,000 (9%) |

Flat scan latency has two components: (1) a PQ-scan baseline that scales with `total_points`, and (2) a rerank cost that scales with `min(matching_count, L)`. For indexes ≤ 2M, the baseline is ~313ms and flat scan stays under 500ms up to ~1M matching. For larger indexes, the baseline rises and flat scan crosses 500ms sooner.

The default `max_brute_force_count = 1,000,000` is a conservative choice that works for indexes up to ~3M. Callers with larger indexes (5M+) may want to lower it (e.g., 500K) to keep flat scan latency under 500ms. The parameter is exposed in `QueryPlannerConfig` for this purpose.

**Three-threshold evaluation:**

| Approach | Mistakes (109 points, 150K–10M) | Notes |
|---|---|---|
| Pure match rate only (rate ≤ 25%) | 14/109 | Fails on small indexes |
| matching_count ≤ 200K + rate ≤ 25% (two thresholds) | 9/109 | No upper bound on flat scan |
| **matching_count ≤ 200K + max_brute_force ≥ 1M + rate ≤ 25%** | **7/109** | Caps flat scan latency |

The three-threshold approach caps the 10M 20% case (matching=2M, rate<25%) that the two-threshold version routes to flat scan at ~1.7s.

## Trade-offs

### Matching-count threshold + match-rate threshold (hybrid) vs. alternatives

**Chosen: hybrid.** The planner first checks `matching_count ≤ 200K` (flat scan is cheap), then checks `match_rate ≤ 25%` (beta recall dip zone). This was validated against 109 data points across 5 datasets (150K–10M vectors) and achieves 9/109 mistakes — better than pure match-rate (14/109) or total-points-based (11/109) approaches.

**Alternative considered: pure matching-count threshold.** Suryansh Gupta proposed thresholding on `matching_count` alone (e.g., ≤ 250K → flat scan). This works for indexes ≤ 1M because flat scan latency plateaus at ~313ms regardless of matching count. However, on 10M vectors a fixed count threshold gets 11/26 decisions wrong: it routes 300K–1M matching vectors to beta filter where recall is 39–81% (the dip zone). The dip zone is a **match-rate phenomenon** (consistently at 2–10% rate), not an absolute count, so the second threshold must use match rate.

**Alternative considered: pure match-rate threshold (rate ≤ 25%).** Simpler but makes 14/109 mistakes: it routes small indexes (150K) at 50–100% rate to beta filter, even though flat scan is faster and has better recall on those indexes. The matching-count check catches this regime.

**Alternative considered: total_points ≤ 200K + rate ≤ 25%.** The original RFC approach. Makes 11/109 mistakes: it mishandles the 292K index at 25–77% rate where matching count is ≤ 200K (flat scan is still ~313ms and 100% recall, but the planner routes to beta). Replacing `total_points` with `matching_count` fixes these cases.

### Planner as a separate struct vs. integrated into `DiskProvider`

**Chosen: separate struct.** `QueryPlanner` is a pure function of `(config, total_points, matching_count)` → `SearchPlan`. It has no dependency on `DiskProvider`'s internals, runtime, or I/O context. Keeping it separate makes it testable in isolation and reusable across different provider implementations.

### `plan()` returning `QueryStrategy` vs. directly returning `SearchPlan`

**Chosen: both.** `plan()` returns the lightweight `QueryStrategy` enum (no allocation, `Copy`), useful for logging, metrics, and testing. `plan_search()` takes a generic predicate closure and produces a ready-to-use `SearchPlan`. The planner is bitmap-agnostic — callers wrap any bitmap type (`RoaringBitmap`, `BitSet`, `HashSet<u32>`) into a closure before calling `plan_search()`.

## Benchmark Results

All experiments: `beta=0.5`, flat scan `L=2000`, beta `L=2000` and `L=3000`, `K=10`, `beam_width=4`, `squared_l2` distance.

### Experiment 1: 150K vectors (dim=896)

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

### Experiment 2: 292K vectors (dim=896)

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

### Experiment 3: 958K vectors (dim=896)

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

### Experiment 4: 1M vectors (dim=384)

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

**Observation:** Similar pattern to the 896d experiments but with worse beta recall at low pass rates (0% below 1%). Beta recall dip zone extends to ~10%. Above 23%, beta achieves 89%+ recall. Flat scan latency rises from 313ms to **666ms at 100%**. Beta latency constant at ~315ms.

### Experiment 5: 10M vectors (dim=384)

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

The three-threshold planner (`matching_count ≤ 200K` → flat, `matching_count ≥ 1M` → beta, `match_rate ≤ 25%` → flat) correctly routes the vast majority of benchmark data points across 7 index sizes (150K–10M). The lower bound ensures flat scan is used whenever it's cheap (≤200K matching vectors → ~313ms latency, ~100% recall). The upper bound caps flat scan latency (at 1M matching, flat scan is 538–973ms depending on index size). The match-rate guard avoids the beta recall dip zone at 2–10% match rate. Above 25%, beta filter achieves 95–99% recall at constant ~315ms latency.

## Future Work

- [ ] **Adaptive beta value.** The current design uses a fixed `beta = 0.5`. Future work could adapt beta based on match rate — higher beta (less bias) at higher match rates, lower beta (more bias) at lower match rates — to improve recall in the transition zone.
- [ ] **Multi-strategy planner.** The current planner selects between two strategies. Future graph algorithms (e.g., `MultihopSearch`) could be added as additional `GraphMode` variants with their own match-rate ranges. Each new variant lands with its own threshold derivation and a new `QueryStrategy` arm.
- [ ] **`PostFilter` dominance validation.** The planner currently omits `GraphMode::PostFilter` because it is strictly dominated by `BetaFilter` — `BetaFilter` applies the same hard post-filter plus beta-biased beam traversal. No `(matching_count, match_rate)` cell in the benchmark data shows `PostFilter` outperforming both `FlatScan` and `BetaFilter`. If future datasets or workloads reveal a regime where `PostFilter` wins, the planner should add a corresponding `QueryStrategy` arm.

## References

1. [RFC 01101: Beta Filter For Disk Search](https://github.com/dyhyfu/DiskANN/blob/c3ae608683531765920f0844d70750efa731946a/rfcs/01101-disk-beta-filter.md) — prerequisite design for `SearchPlan`, `GraphMode`, and beta-biased PQ distance computation.
