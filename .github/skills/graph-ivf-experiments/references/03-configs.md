# Stage 3 — Benchmark configs

Experiment configs live in the untracked `_results/configs/` tree, grouped by campaign:

| Directory | Contents |
|---|---|
| `_results/configs/sweeps/<dataset>/graph-ivf/` | build and search sweeps for one dataset |
| `_results/configs/sweeps/<dataset>/disk-index/` | matching disk-index baselines |
| `_results/configs/churn/` | insert/delete churn runs |
| `_results/configs/msturing/` | streaming BigANN runbook replays |

`diskann-benchmark/example/graph-ivf-*.json` holds only the three small, portable,
version-controlled examples (static build, online build, search) — keep dataset-specific
sweep configs out of it. The full schema is documented in
[`diskann-benchmark/README.md`](../../../../diskann-benchmark/README.md); this covers the
online graph-IVF conventions.

Naming: `graph-ivf-<dataset>-<static|online|load|runbook>-<identity tags>.json`,
e.g. `graph-ivf-nq-online-t800-s32-b4096-iters6.json`,
`graph-ivf-nq-load-t120-r1000.json`.

## Source types

| Source | Purpose | Search timing |
|---|---|---|
| `Static` | Batch k-means baseline with an explicit cluster count | optional final `search_phase` |
| `Online` | One corpus-order insert pass with split-driven cluster growth | optional final `search_phase` |
| `Load` | Re-sweep an existing static or online index | required `search_phase` |
| `OnlineRunbook` | Replay BigANN insert/delete/search stages against a live partition | nested `search` at each runbook search stage; optional final `search_phase` |

For immutable-index experiments, use one build config per index and one or more `Load`
configs for search bands. Prefer measuring every recall depth in the same config. Use
separate legacy-style depth configs only when extending existing results that analysis
already joins by effective `nlist`.

### Build-only (`Online`)

Constructs the index and saves it. No `search_phase`, so it produces only build statistics.

```jsonc
{
  "search_directories": ["<data-root>/mteb-nq-full"],
  "jobs": [{
    "type": "graph-ivf",
    "content": {
      "source": {
        "graph-ivf-source": "Online",
        "data_type": "minmax8",
        "data": "corpus_nozero_minmax8.bin",   // resolved via search_directories
        "distance": "squared_l2",
        "dim": 1536,

        "split_threshold": 800,     // T — the knob that sets cluster count
        "batch_size": 4096,
        "warmup_centroids": 100,    // seed clustering …
        "warmup_points": 10000,     // … over the first N points
        "warmup_iters": 15,
        "assign_l": 64,
        "two_means_iters": 6,       // split quality
        "reassign_neighbors": 32,   // "s" in the run names
        "reassign_l": 256,
        "capacity_mult": 3,
        "normalize": true,

        "graph_degree": 32,         // centroid navigation graph
        "graph_slack": 1.2,
        "graph_l_build": 64,
        "graph_alpha": 1.2,

        "num_threads": 16,
        "seed": 0,
        "save_path": ".../_tmp/graphivf_nq_online_s32_t800_b4096_iters6_nozero_minmax8",
        "telemetry_csv": ".../..._nozero_minmax8.splits.csv"
      }
    }
  }]
}
```

`split_threshold` is the primary independent variable. It is a soft equilibrium target, not
a cap: cluster count emerges as roughly `N / (0.7 · T)`, so `T ≈ 1.43 · N / target`. Finer
partitions need a larger coefficient (~2.3 at 45K clusters on Caselaw). Leave
`max_clusters` unset for uncapped runs.

`save_path` doubles as the identity of the run for analysis. Put every varied parameter in
it.

### Sweep (`Load`)

Loads the saved index and sweeps fractions of its clusters. Never rebuilds.

```jsonc
{
  "search_directories": ["<data-root>/mteb-nq-full"],
  "jobs": [{
    "type": "graph-ivf",
    "content": {
      "source": {
        "graph-ivf-source": "Load",
        "data_type": "minmax8",
        "load_path": ".../_tmp/graphivf_nq_online_s32_t120_b4096_iters6_nozero_minmax8"
      },
      "search_phase": {
        "queries": "queries_sample1000_minmax8.bin",
        "groundtruth": "groundtruth_nozero_recall_1000_query_1000.bin",
        "num_threads": 1,
        "cluster_fractions": [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.0125, 0.0175, 0.0225, 0.03, 0.04, 0.05, 0.06],
        "centroid_search_alpha": 4.0,
        "recall_at": [50, 1000],
        "distance": "squared_l2"
      }
    }
  }]
}
```

### Streaming churn (`OnlineRunbook`)

An `OnlineRunbook` nests the normal online fields under `build`, identifies the BigANN
dataset and groundtruth directory under `runbook`, and defines the live search under
`search`. Its mutation ranges are sub-batched by `build.batch_size`; searches happen only
at explicit runbook search stages. See [stage 7](./07-online-runbooks.md) for the complete
schema, input audit, query/truthset subsetting, and long-run validation procedure.

## Rules

**Fractions are in `(0.0, 1.0]`.** For `C` clusters the runner probes
`ceil(cluster_fraction * C)` lists and records that effective `nlist`. The centroid beam
is derived from it as `max(128, ceil(centroid_search_alpha * nlist))`, so it tracks the
sweep automatically. `centroid_search_alpha` defaults to 4.0, measured to retrieve ~98% of
the exactly-nearest clusters; 1.5 retrieves only ~63% and costs ~7 points of recall@50.

**List every depth in one config.** `"recall_at": [50, 1000]` searches once per cluster
fraction, to the deepest value, and scores both from that result set — one run instead of
two. A scalar is accepted for compatibility. Validation sorts and deduplicates a list.
Results predating this were measured as an `-r50` and an `-r1000` run per index, and
analysis still joins those on effective `nlist`; a value present in only one is dropped
rather than half-plotted.

**`num_threads: 1` for sweeps.** Latency, QPS and per-query I/O are only comparable
single-threaded. Builds use 16.

**Groundtruth `K` must be ≥ the largest `recall_at`.** Checked before the sweep runs.
All these studies use `recall_1000` groundtruth and measure at 50 and 1000.

**The build config's `save_path` and the sweep config's `load_path` must match exactly.**
A typo produces a "no run matching" gap in the analysis, not an error at run time.

## Choosing the cluster-fraction band

Pick a geometric-ish ladder that brackets 90% recall at both depths, ~14 points. Starting
points that worked:

| Clusters | Fraction range | Approximate `nlist` range |
|---|---|---|
| ~4.5K (T≈800) | 0.002 → 0.135 | 9 → 608 |
| ~36K (T≈120) | 0.002 → 0.06 | 72 → 2160 |
| ~115K (T≈50) | 0.002 → 0.08 | 230 → 9200 |

Recall@1000 needs roughly 1.7–2.3× the effective `nlist` of recall@50 for the same recall,
so size the top fraction off the deeper measurement.

### Banding

The centroid beam is charged to every query in a sweep. It used to be a constant, which
forced a choice between inflating the cheap end of the curve and silently widening only at
large fractions — the reason these sweeps were split into bands spliced by
`benchlib.concat_bands`. `centroid_search_alpha` removes that reason: the beam is
`max(128, ceil(alpha * nlist))`, so it is proportional at every point of the ladder and one
config can span the whole range. Banding is still useful for splitting a long sweep into
resumable pieces, and `concat_bands` still splices them (narrowest first; duplicate
effective `nlist` keeps its first appearance).

If the band tops out below the target recall, add an `-ext` config extending the fraction
upward rather than re-running the whole sweep — and delete it once folded in, so it cannot
be picked up as a duplicate later.
