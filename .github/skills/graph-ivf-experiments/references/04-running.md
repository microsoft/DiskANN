# Stage 4 — Running builds and sweeps

## Build the runner

```powershell
cargo build --release -p diskann-benchmark --features graph-ivf
```

Add `disk-index` only when the same run needs that backend; unrelated feature sets can
require additional generated tooling. Rebuild after any change under `diskann-benchmark/`
or `diskann-graphivf/`.

## Run one job

```powershell
$b = "<absolute-repo-root>"
& "$b\target\release\diskann-benchmark.exe" run `
  --input-file  "$b\diskann-benchmark\example\graph-ivf-nq-load-t120.json" `
  --output-file "$b\_results\logs\nq\graphivf_nq_t120_s32_b4096_iters6_nz_search.json" `
  2>&1 | Tee-Object -FilePath "$b\_results\logs\nq\graphivf_nq_t120_s32_b4096_iters6_nz_search.log" | Out-Null
Write-Output "DONE exit=$LASTEXITCODE"
```

- `Tee-Object … | Out-Null` keeps the console readable while preserving the full log.
  `Out-File` mangles encoding (UTF-16 BOM); `Tee-Object` does not.
- The trailing `Write-Output` makes completion and exit status unambiguous.
- The `.json` is what analysis reads; the `.log` is for diagnosing a failure. Keep both,
  same stem, in `_results/logs/<dataset>/`.

**Run one benchmark at a time.** Invoke it directly rather than through a PowerShell loop.
The terminal integration may move a quiet long-running command into the background; retain
that execution and leave it alone until completion notification. Do not launch another
benchmark beside it.

**Never clean up a terminal while a run is in flight.** Doing so kills the child process —
you get no log, no JSON, and no error. One t120 sweep was lost exactly this way and had to
be relaunched from scratch.

## Order of operations

```
build t=<A>  →  build t=<B>  →  sweep A@[50,1000]  →  sweep B@[50,1000]
```

Builds are the expensive, serial part (mostly per-insert). Sweeps reuse the saved index, so
get the builds done first. Split a sweep into multiple centroid-beam bands only when one
beam would distort the low-effort end; each band can still measure both recall depths.

An `OnlineRunbook` does not follow this order: it intentionally interleaves mutations and
live searches in one replay. Runbook JSON, split telemetry, and merge telemetry are written
only after replay and final flush complete, so a quiet log and absent output files are not
stage-level progress indicators. Check process liveness and cumulative CPU without killing
or restarting the terminal.

## Naming

The run name is the experiment's identity — analysis selects on the embedded `save_path` /
`load_path`, and `benchlib.select_one` raises on ambiguity rather than guessing.

```
graphivf_<dataset>_<t-tag>_<variant tags>_<correction marker>_<phase>.json
graphivf_nq_online_t800_s32_b4096_iters6_nz_build.json
graphivf_nq_t800_s32_b4096_iters6_nz_search.json
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

`results.search.search_results_per_nlist[]` gives the requested `cluster_fraction`, its
effective concrete `nlist`, `qps`, `mean_latency`, `p95_latency`, `p999_latency`, recall,
and a `breakdown` of `io_count`, `bytes_read`, `preprocess_ns`, `centroid_search_ns`,
`plan_io_ns`, `disk_read_ns`, `score_ns`, and `topk_ns`.

Expected shapes — a deviation is a bug, not a finding:

- **I/O count ≈ `nlist`** (sometimes `nlist - 1`). Each probed list is one read.
- **Bytes read grows linearly in `nlist`.** Multiple recall depths scored from one result
  buffer share the same I/O measurements.
- **Centroid search time is flat** across `nlist` only while
  `centroid_search_l >= nlist`; the effective beam is `max(centroid_search_l, nlist)`.
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
