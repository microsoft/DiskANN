---
name: graph-ivf-experiments
description: 'Run a graph-IVF or disk-index benchmark experiment end to end: prepare a dataset, compute and audit groundtruth, author benchmark configs, build indexes, sweep recall@50 / recall@1000, then turn the results into plots and the Excel workbook. Use when asked to run a sweep, add a new dataset, build an online graph-IVF index, measure or improve recall, compare split thresholds, regenerate the online plots, or update the results workbook.'
argument-hint: 'dataset name and what you want measured, e.g. "sweep recall@50 and @1000 on NQ at T=800 and T=120"'
---

# Graph-IVF Benchmark Experiments

End-to-end procedure for the comparative online graph-IVF study (Enron, MERB, Caselaw,
MSMARCO, LoTTE, GloVe-200, MTEB-NQ). Read the stage reference before doing that stage.

## Where things live

| What | Where | Tracked? |
|---|---|---|
| Corpora, queries, groundtruth, saved indexes | `<data-root>/<dataset>/` | No — outside the repo |
| Dataset prep tool (`vecprep.py`, `compress_minmax`) | `diskann-graphivf/scripts/dataprep/` | Yes |
| Benchmark configs | `diskann-benchmark/example/*.json` | Yes |
| Runner binary | `target/release/diskann-benchmark.exe` | No (build output) |
| Run logs + results | `_results/logs/<dataset>/*.{json,log}` | **No — untracked** |
| Analysis scripts + dataset registry | `_results/scripts/`, `_results/scripts/registry.py` | **No — untracked** |
| Figures / workbooks | `_results/plots/<dataset>/`, `_results/workbooks/` | **No — untracked** |

`<data-root>` is a directory of your choosing outside the clone — the datasets are far too
large to live in the repository, and nothing in the tooling assumes a particular location.
The study used a `data/` directory sitting alongside the clone. Wherever you put it, the
path appears only in benchmark configs and on `vecprep.py` command lines.

`_results/` is not gitignored, it has simply never been committed. Treat it as local
working state and do not assume a teammate has it. Superseded one-off scripts are parked in
`_results/scripts/_archive/` rather than deleted, for the same reason — there is no git
history to recover them from.

**The tooling is dataset-agnostic and must stay that way.** `vecprep.py` handles any corpus
via flags, and every plot/workbook script reads `registry.py` instead of naming datasets.
Adding a dataset-specific script is a regression, not a shortcut.

## Pipeline

```mermaid
flowchart LR
  A[1. Dataset prep] --> B[2. Groundtruth + audit]
  B --> C[3. Configs]
  C --> D[4. Build + sweep]
  D --> E[5. Plots + workbook]
```

1. **[Dataset preparation](./references/01-dataset-prep.md)** — acquire vectors, convert to
   the `fp16` / `minmax8` binary formats, subsample queries.
2. **[Groundtruth](./references/02-groundtruth.md)** — compute exact top-1000 **and audit it**.
   Skipping the audit is the single most expensive mistake available here.
3. **[Benchmark configs](./references/03-configs.md)** — one build-only config per index,
   one sweep config per (index, recall depth).
4. **[Running](./references/04-running.md)** — build the runner, execute builds and sweeps,
   capture logs.
5. **[Analysis](./references/05-analysis.md)** — add the dataset to `registry.py`, render the
   four standard figures, refresh the workbook.

[Environment and tooling gotchas](./references/06-environment.md) apply throughout; read it
first if anything behaves strangely.

## Invariants

Violating any of these produces results that look plausible and are wrong, or costs hours.

**Measurement**

- **Audit the groundtruth before trusting any recall number.** Degenerate rows (all-zero
  vectors from empty documents) land *inside* the retrieval band under squared L2 and
  silently cap recall. See [stage 2](./references/02-groundtruth.md).
- **`nlist` must be `<= centroid_search_l`.** The runner rejects configs that violate it.
- **A run measures one `recall_at`.** recall@50 and recall@1000 are separate runs of the
  same index, joined on `nlist` during analysis.
- **Sweep single-threaded** (`num_threads: 1`) so latency and per-query I/O are comparable.

**Process**

- **Build once, save, then sweep from `Load` configs.** Never rebuild an index to re-sweep it.
- **Encode every experimental variable in the index and log name.** `_nozero_`, `_s32_`,
  `_b4096_`, `_iters6_`, `_t800_`. Analysis selects runs by these paths, and a superseded
  run left on disk with an ambiguous name will be silently picked up.
- **Run one benchmark command at a time.** A PowerShell `foreach` over several runs
  silently no-ops. See [environment](./references/06-environment.md).
- **Never clean up a terminal with a run in flight** — it kills the child process and you
  lose the sweep with no log and no error.

## Quick reference

```powershell
# Build the runner (from repo root)
cargo build --release -p diskann-benchmark --features graph-ivf,disk-index

# One run: config in, JSON out, console tee'd to a log
$b = git rev-parse --show-toplevel   # repo root, from anywhere in the working tree
& "$b\target\release\diskann-benchmark.exe" run `
    --input-file  "$b\diskann-benchmark\example\<config>.json" `
    --output-file "$b\_results\logs\<dataset>\<run>.json" `
    2>&1 | Tee-Object -FilePath "$b\_results\logs\<dataset>\<run>.log" | Out-Null
Write-Output "DONE exit=$LASTEXITCODE"

# Regenerate all figures and the workbook (conda python — the venv has no matplotlib)
Push-Location "$b\_results\scripts"
python plot_online_bytes_recall.py
python plot_online_diskread_recall.py
python plot_online_latency_breakdown.py
python plot_online_cost.py
python build_online_workbook.py
Pop-Location
```
