# Stage 5 — Analysis: plots and the workbook

All analysis lives in `_results/scripts/` and shares [`benchlib.py`](../../../../_results/scripts/benchlib.py).

**Run from `_results/scripts/`** (the scripts `import benchlib` by bare name) **using conda
python, not the venv** — the venv has numpy and openpyxl but no matplotlib. See
[environment](./06-environment.md).

## benchlib

Two eras of artifacts coexist and it reads both: legacy `sweep_*.txt` logs (filename-keyed,
both recall columns in one file) and runner `*.json` (parameter-keyed). **Prefer JSON for
anything new** — a filename only the author can decode goes stale silently.

| Function | Purpose |
|---|---|
| `resolve(name)` | find an existing artifact anywhere under `_results/`, else route a new one by extension and dataset |
| `load_runs()` | every runner JSON under `_results/logs/` as `Run` objects |
| `select(runs, **criteria)` / `select_one(...)` | pick runs by parameter; values may be callables used as predicates |
| `online_rows(runs, **criteria)` | one index's rows with both recall columns, joined on `nlist` |
| `variant_rows(spec, runs)` | rows for one plot variant; accepts all spec shapes below |
| `concat_bands(specs, runs)` | splice sweep bands into one curve |
| `ios_and_bytes_at(rows, key)` | interpolate to the 90% recall target, clamped if unattained |

`Run.params` flattens the job source plus `recall_at`, `search_num_threads` and
`search_centroid_search_l`, so runs are selected by *what they were*, not what they were
named.

**Spec shapes** accepted by `variant_rows`:

| Shape | Meaning |
|---|---|
| `"sweep_x.txt"` | legacy log |
| `{...}` | `online_rows` criteria matching one index's @50 and @1000 runs |
| `(spec50, spec1000)` | the two depths selected separately |
| `[spec, ...]` | bands concatenated into one curve |

Row columns: `nlist, qps, mean_us, p95_us, p999_us, recall, recall50, recall1000, bytes_q,
ios_q, reqbytes, preproc_us, centroid_us, planio_us, diskread_us, score_us, topk_us`.
Legacy rows carry `p99_us`; JSON rows carry `p999_us`. Neither has both.

## The dataset registry

[`registry.py`](../../../../_results/scripts/registry.py) is the **single source of dataset
truth**. The four online plot scripts and the workbook builder all read it; none of them
contains a dataset name. Keep it that way — a dataset-specific script is a defect.

(The module is `registry.py`, not `datasets.py`, so it cannot be shadowed by the PyPI
`datasets` package.)

Each entry is keyed by display name and carries identity (`slug`, `title`, `sheet`,
`caption`, `overview_label`, `npts`, `dim`, `row_bytes`), prose (`meta`, `search_note`),
optional axis control (`max_scan_pct`), and a list of `variants`.

Variants are built by `variant(threshold, clusters, search, build=..., qualifier=...,
clusters_label=..., workbook_label=...)`, which derives the per-consumer labels so the same
numbers cannot drift apart between a figure and a spreadsheet:

| Derived key | Consumed by | Shape |
|---|---|---|
| `plot_label` | bytes-recall, diskread-recall | `"16,384 centroids (T=106)"` |
| `cost_label` | cost | `"106"` |
| `latency_label` | latency breakdown | `"16,384"` |
| `workbook_label` | workbook | explicit override |

Selector helpers live there too: `ends(suffix)`, `build_of(suffix)`, `split_depths(suffix)`
(the save-path@50 / load-path@1000 pair), and `beam_bands(suffix, beams)` for multi-band
curves.

`ORDER` drives the figures (8 datasets); `WORKBOOK_ORDER` drives the workbook (7 — it
excludes the plot-only `s=32` ablation, which has `sheet: None`).

## Registering a new dataset

**Two edits**, and only the first is easy to forget:

1. **`benchlib.resolve`** — add the dataset token to the routing tuple:

   ```python
   for d in ("caselaw", "merb", "msmarco", "lotteall", "lotte", "glove200", "nq", "enron")
   ```

   Order matters: it is a substring test, so more specific names come first (`lotteall`
   before `lotte`). Miss this and figures render but land in the wrong directory.

2. **One entry in `registry.py`**, appended to `ORDER` (and `WORKBOOK_ORDER` if it should
   get sheets). Its `slug` must be the token from step 1, since `figure_path` builds
   `online_<figure>_<slug>.png` from it.

   Make each variant's selector suffix specific enough to exclude superseded runs still on
   disk. When building selectors in a loop, bind the loop variable as a default argument
   (`lambda p, s=suffix: ...`) — otherwise every closure captures the last value.

Then run the four plot scripts and the workbook builder and confirm each prints `wrote …`.
A `skip (missing): …` line means the selector matched zero runs, or `select_one` found
several — check with `load_runs()` + `select()` that exactly two runs match (one per depth).

## The four figures

| Script | Figure |
|---|---|
| `plot_online_bytes_recall.py` | bytes read/query as % of dataset vs recall |
| `plot_online_diskread_recall.py` | disk-read time vs recall |
| `plot_online_latency_breakdown.py` | stacked latency at ~90% recall |
| `plot_online_cost.py` | IOs/query and avg bytes/IO at ~90% recall |

Each writes one PNG per dataset into `_results/plots/<dataset>/`.

**House style** (tuned for slides): `figsize=(8, 8)`, `dpi=150`, base font 19, title 20,
legends 16–17, data labels 13–14, dashed grid at `alpha=0.45` with minor ticks on both axes.
Solid line / solid fill = recall@50; dotted / faded = recall@1000.

At this size the bar charts need care: long titles must wrap to a second line, legends need
~1.65–1.85× headroom above the tallest bar, and paired bar labels must be staggered
vertically or near-equal values collide.

## The workbook

`build_online_workbook.py` → `_results/workbooks/graphivf_online_results.xlsx`: an Overview
sheet plus `<Dataset> Search` and `<Dataset> Build` per dataset. Both the sheets and the
Overview's dataset table are generated from `WORKBOOK_ORDER`, so adding a registry entry is
the whole job.

A variant is `(label, search_spec, build_spec)`, where each spec is a legacy log filename
**or** runner-selection criteria — `build_of(...)` for a build run, `split_depths(...)` for
a search pair. Build stats come from `parse_build` (legacy text) or `build_from_run` (JSON,
microseconds → seconds). Specs may be bare filenames; every consumer calls `resolve` itself.

Record in `meta` the metric, query count, groundtruth, and any corpus caveat.

Columns a source does not record stay blank rather than being faked: legacy rows have no
`batch size` / `reassign s` / `reassign_l`, JSON rows have no `p99`.

## Cross-checks before reporting

- I/O volume identical across recall depths for the same index.
- Cluster count and mean size consistent between the build sheet and the plot labels.
- The 90%-recall interpolation is not clamped (i.e. the sweep actually reached 90%);
  `plot_online_cost.py` hatches bars where it did not, and those numbers are lower bounds.
