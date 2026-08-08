# Stage 4 — Running builds and sweeps

## Build the runner

```powershell
cargo build --release -p diskann-benchmark --features graph-ivf,disk-index
```

Both features are needed for the graph-IVF vs disk-index comparison. Rebuild after any
change under `diskann-benchmark/` or `diskann-graphivf/`.

## Run one job

```powershell
$b = git rev-parse --show-toplevel   # repo root, from anywhere in the working tree
& "$b\target\release\diskann-benchmark.exe" run `
    --input-file  "$b\diskann-benchmark\example\graph-ivf-nq-load-t120-r1000.json" `
    --output-file "$b\_results\logs\nq\graphivf_nq_t120_s32_b4096_iters6_nz_r1000.json" `
    2>&1 | Tee-Object -FilePath "$b\_results\logs\nq\graphivf_nq_t120_s32_b4096_iters6_nz_r1000.log" | Out-Null
Write-Output "DONE exit=$LASTEXITCODE"
```

- `Tee-Object … | Out-Null` keeps the console readable while preserving the full log.
  `Out-File` mangles encoding (UTF-16 BOM); `Tee-Object` does not.
- The trailing `Write-Output` makes completion and exit status unambiguous.
- The `.json` is what analysis reads; the `.log` is for diagnosing a failure. Keep both,
  same stem, in `_results/logs/<dataset>/`.

**Run one at a time.** A `foreach` loop over several benchmark invocations silently
executes nothing. Long sweeps should use async mode and be left alone until they report
completion.

**Never clean up a terminal while a run is in flight.** Doing so kills the child process —
you get no log, no JSON, and no error. One t120 sweep was lost exactly this way and had to
be relaunched from scratch.

## Order of operations

```
build t=<A>  →  build t=<B>  →  sweep A@50  →  sweep A@1000  →  sweep B@50  →  sweep B@1000
```

Builds are the expensive, serial part (mostly per-insert). Sweeps are cheap by comparison
and reuse the saved index, so get both builds done first.

## Naming

The run name is the experiment's identity — analysis selects on the embedded `save_path` /
`load_path`, and `benchlib.select_one` raises on ambiguity rather than guessing.

```
graphivf_<dataset>_<t-tag>_<variant tags>_<correction marker>_<phase>.json
graphivf_nq_online_t800_s32_b4096_iters6_nz_build.json
graphivf_nq_t800_s32_b4096_iters6_nz_r50.json
```

When results are superseded (corrected groundtruth, rebuilt corpus), **rename the new ones**
rather than deleting the old. Keeping the invalid runs on disk is fine as long as the
selector cannot match them — but verify that it cannot.

## Reading build output

`results.build` in a build JSON carries:

| Field | Note |
|---|---|
| `final_clusters`, `total_splits`, `total_reassigned` | partition outcome |
| `min/mean/max_cluster_size` | mean should land near `0.7 · T` |
| `residual` | clustering quality; lower is better |
| `insert`, `routing`, `split`, `flush` | **microseconds** |
| `centroid_capacity`, `seed`, `num_points`, `dim` | provenance |

`routing` and `split` are sub-phases of `insert`, so wall time is `insert + flush` —
summing all four double-counts. `corpus_load` and `decompress` are harness I/O.

Sanity check `mean_cluster_size ≈ 0.7 · split_threshold` and `final_clusters ≈ N / mean`.
A large deviation means the run did not reach equilibrium.

## Reading sweep output

`results.search.search_results_per_nlist[]` gives per-`nlist`: `qps`, `mean_latency`,
`p95_latency`, `p999_latency`, `recall`, and a `breakdown` of `io_count`, `bytes_read`,
`preprocess_ns`, `centroid_search_ns`, `plan_io_ns`, `disk_read_ns`, `score_ns`, `topk_ns`.

Expected shapes — a deviation is a bug, not a finding:

- **I/O count ≈ `nlist`** (sometimes `nlist - 1`). Each probed list is one read.
- **Bytes read grows linearly in `nlist`**, and is identical between the recall@50 and
  recall@1000 runs of the same index — only the scoring depth differs.
- **Centroid search time is flat** across `nlist` (it depends on `centroid_search_l`, not
  `nlist`): ~2.6 ms at L=1024, ~9 ms at L=2048 with 36K centroids.
- **Disk read dominates** at high `nlist`.

If recall does not climb with `nlist`, stop and re-read [stage 2](./02-groundtruth.md)
before adding sweep points.

## Rough costs

Reference points on 16 threads, 2.68M × 1536 minmax8:

| | T=800 (4.6K clusters) | T=120 (36K clusters) |
|---|---|---|
| build wall | ~474 s | ~1325 s |
| sweep, 14 points, 1000 queries, single-thread | tens of minutes | ~1 hour |

Finer partitions cost disproportionately more to build (splitting dominates) and less to
search.
