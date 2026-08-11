# Stage 3 — Benchmark configs

Configs live in `diskann-benchmark/example/` and are the one part of an experiment that is
version-controlled. The full schema is documented in
[`diskann-benchmark/README.md`](../../../../diskann-benchmark/README.md); this covers the
online graph-IVF conventions.

Naming: `graph-ivf-<dataset>-<online|load>-t<threshold>[-<variant tags>][-r<depth>].json`,
e.g. `graph-ivf-nq-online-t800-s32-b4096-iters6.json`,
`graph-ivf-nq-load-t120-r1000.json`.

## Two kinds of config

An experiment needs **one build config per index** and **two sweep configs per index** (one
per recall depth).

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

Loads the saved index and sweeps `nlist`. Never rebuilds.

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
        "nlist": [60, 100, 140, 180, 220, 280, 350, 450, 600, 800, 1000, 1300, 1700, 2000],
        "centroid_search_l": 2048,
        "recall_at": [50, 1000],
        "distance": "squared_l2"
      }
    }
  }]
}
```

## Rules

**`nlist <= centroid_search_l`.** Hard constraint. Size the beam to the largest `nlist` in
the band.

**List every depth in one config.** `"recall_at": [50, 1000]` searches once per `nlist`,
to the deepest value, and scores both from that result set — one run instead of two.
Results predating this were measured as an `-r50` and an `-r1000` run per index, and
analysis still joins those on `nlist`; an `nlist` present in only one is dropped rather
than half-plotted.

**`num_threads: 1` for sweeps.** Latency, QPS and per-query I/O are only comparable
single-threaded. Builds use 16.

**Groundtruth `K` must be ≥ the largest `recall_at`.** Checked before the sweep runs.
All these studies use `recall_1000` groundtruth and measure at 50 and 1000.

**The build config's `save_path` and the sweep config's `load_path` must match exactly.**
A typo produces a "no run matching" gap in the analysis, not an error at run time.

## Choosing the `nlist` band

Pick a geometric-ish ladder that brackets 90% recall at both depths, ~14 points. Starting
points that worked:

| Clusters | `nlist` range | `centroid_search_l` |
|---|---|---|
| ~4.5K (T≈800) | 10 → 600 | 1024 |
| ~36K (T≈120) | 60 → 2000 | 2048 |
| ~115K (T≈50) | 200 → 9000 | 16384 |

recall@1000 needs roughly 1.7–2.3× the `nlist` of recall@50 for the same recall, so size the
top of the band off the deeper measurement.

### Banding

The centroid beam is charged to every point in a sweep, so sweeping a wide range under one
large `centroid_search_l` inflates the cheap end of the curve — real cost at the top,
pure overhead at the bottom. Split wide ranges into bands, each with the narrowest beam that
admits its largest `nlist`, and let `benchlib.concat_bands` splice them (narrowest first;
duplicate `nlist` keeps its first, cheaper appearance). GloVe-200 is the worked example.

If the band tops out below the target recall, add an `-ext` config extending `nlist`
upward rather than re-running the whole sweep — and delete it once folded in, so it cannot
be picked up as a duplicate later.
