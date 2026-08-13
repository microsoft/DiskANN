# Stage 7 — Streaming online runbooks

Use this workflow for a BigANN runbook that interleaves insert, delete, and search stages.
Unlike an immutable build/load sweep, every search observes the current in-memory partition.

## Audit the inputs first

Before building, verify all of the following:

1. Read the corpus and query matrix headers and confirm their element type and dimension.
2. Parse the selected dataset section of the runbook. Reject unsupported replace stages.
3. Replay insert/delete ranges as a set of live corpus row ids. Check bounds, duplicate
   inserts, deletes of absent ids, non-empty search stages, maximum live points, and final
   live points.
4. Resolve exactly one `step<stage>.gt<depth>` file for every search stage.
5. Check each truthset header, required recall depth, id bounds, and that every sampled
   groundtruth id is live at that stage.

A runbook stage is an experimental boundary. `build.batch_size` may split one insert or
remove range into smaller internal mutation batches, but it must not create extra searches.

## Subset queries and truthsets safely

For an expensive run, it is often sufficient to use the first 1,000 queries. Preserve row
alignment across the query matrix and every search-stage truthset:

- write a query matrix with header `(1000, dim)` and the first `1000 * dim` values;
- write every truthset with header `(1000, depth)`;
- copy the first `1000 * depth` ids;
- if distances are present, seek past the original file's complete ids block and copy the
  first `1000 * depth` distances separately;
- replay the live-id audit against the subset files.

Do not truncate a truthset as one byte prefix. IDs and distances are separate complete
row-major blocks.

## Author the config

```jsonc
{
  "jobs": [{
    "type": "graph-ivf",
    "content": {
      "source": {
        "graph-ivf-source": "OnlineRunbook",
        "build": {
          "data_type": "float32",
          "data": "corpus.fbin",
          "distance": "squared_l2",
          "dim": 100,
          "split_threshold": 120,
          "merge_threshold": 40,
          "batch_size": 4096,
          "reassign_neighbors": 10,
          "reassign_l": 64,
          "warmup_centroids": 100,
          "warmup_points": 10000,
          "warmup_iters": 15,
          "assign_l": 64,
          "two_means_iters": 12,
          "capacity_mult": 3,
          "normalize": false,
          "graph_degree": 32,
          "graph_slack": 1.2,
          "graph_l_build": 64,
          "graph_alpha": 1.2,
          "num_threads": 16,
          "seed": 0,
          "save_path": "<absolute-output-prefix>",
          "telemetry_csv": "<absolute-split-csv>"
        },
        "runbook": {
          "runbook_path": "final_runbook.yaml",
          "dataset_name": "dataset-key",
          "gt_directory": "<groundtruth-directory>"
        },
        "search": {
          "queries": "queries_first1000.fbin",
          "cluster_fractions": [0.01, 0.02, 0.03, 0.04, 0.05],
          "centroid_search_l": 4096,
          "recall_at": [50]
        }
      }
    }
  }]
}
```

Set `distance` to the dataset's actual metric. `normalize` controls warmup/split centroid
normalization only; it does not normalize corpus or query rows. Maintain merge hysteresis:
`2 * merge_threshold <= split_threshold`. Estimate the maximum live cluster count and make
`centroid_search_l` intentionally large enough for the maximum effective
`nlist = ceil(max_fraction * live_clusters)`; the search implementation otherwise widens
its beam to `max(centroid_search_l, nlist)`.

## Run and monitor

Build only the required backend when disk-index comparison is not part of the run:

```powershell
cargo build --release -p diskann-benchmark --features graph-ivf
```

Run exactly one long benchmark at a time and preserve both JSON and console log. Runbook
result and split/merge telemetry files are finalized only after replay and final flush, so
absence of output during the run is expected. Monitor process responsiveness and cumulative
CPU without polling aggressively or cleaning up its terminal.

## Validate and analyze

After a successful exit, validate before plotting:

- result stage count and operation sequence match the runbook;
- every search stage has one result per requested cluster fraction;
- each effective `nlist` equals `ceil(fraction * that stage's live clusters)`, clamped to
  the live cluster count;
- recall entries have exactly the configured depths;
- live point counts match the independent runbook replay;
- final build statistics agree with the final stage state;
- split and merge telemetry parse, have valid stage/batch ordering, and contain no events
  after the final mutation.

Export separate build, stage, search, split, and merge tables. Plot recall and scan
percentage against search-stage ordinal, and annotate insert/delete regions so churn effects
are not confused with static search-effort curves.
