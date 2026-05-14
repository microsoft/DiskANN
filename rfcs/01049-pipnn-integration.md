# Integrate PiPNN as an Alternative Graph-Index Build Algorithm

| | |
|---|---|
| **Authors** | Weiyao Luo |
| **Contributors** | DiskANN team |
| **Created** | 2026-05-11 |
| **Updated** | 2026-05-14 |

## Summary

Add **PiPNN** ([Pick-in-Partitions Nearest Neighbors](https://arxiv.org/abs/2602.21247)) as a second algorithm for **DiskANN's disk-index full-build path** — both initial builds and full rebuilds. PiPNN writes a graph byte-compatible with Vamana's disk format and search API, at **up to 6.3× lower build time** on measured workloads. Vamana stays the default; PiPNN is opt-in behind a Cargo feature. In-memory PiPNN build is intentionally out of scope — DiskANN's in-mem path is for streaming per-point construction, which PiPNN's batch algorithm cannot do.

## Motivation

### How DiskANN builds today

DiskANN uses one algorithm — **Vamana** — which inserts points one-by-one with greedy search + `RobustPrune`. Clients use it in three modes:

| Mode | Description |
|---|---|
| Incremental | Per-point inserts on an in-memory graph (Vamana). Disk index not mutated. |
| Full rebuild | Rebuild the entire graph from a snapshot; produces an immutable disk index. |
| Partitioned full rebuild | Shard, build, merge — bounds peak RAM (`build_merged_vamana_index`). |

PiPNN is a faster substitute for modes 2 and 3. Mode 1 stays with Vamana — PiPNN's batch design has no efficient per-point insert (see "Batch-only" below).

### What PiPNN does

A four-phase **batch** builder:

1. **Partition (RBC).** Randomized Ball Carving recursively splits the dataset into small *overlapping* leaf clusters; each point lands in `fanout` of its nearest leaders at every level. Recursion stops at a leaf-size cap (`c_max`, ~256–1024).
2. **Local k-NN per leaf (GEMM).** For each leaf, compute the full pairwise distance matrix as one batched GEMM (intra-leaf N×N where N ≈ `c_max`), then extract per-point top-`leaf_k`. This is structurally different from a 1×N flat scan ([#1036](https://github.com/microsoft/DiskANN/issues/1036)) — the batching across `c_max²` evaluations is where PiPNN's wall-clock advantage comes from.
3. **HashPrune merge.** Merge edges from all leaves into a per-point reservoir of bounded size (`l_max` ~64–128), keyed by an LSH angular bucket for diversity. Naturally streamable — see memory mitigation.
4. **Optional final RobustPrune.** Same algorithm Vamana uses, applied as a single pass when the workload wants more geometric diversification.

Output: `Vec<Vec<u32>>` adjacency lists, handed to the existing disk writer. PQ training and search are unchanged.

### Problem statement

Vamana's per-point cost scales linearly with point count, making 10M+ full builds the bottleneck:

| Dataset | Vamana build |
|---|---:|
| Enron 1M (384d) | 70s |
| BigANN 10M (128d) | 358s |
| Enron 10M (384d) | 844s |

PiPNN completes the same builds **up to 6.3× faster** at matching recall (numbers below).

#### Trade-off hypothesis

> Given a fixed worker (CPU/RAM/SSD), PiPNN delivers higher build throughput than Vamana at matching recall *when its working set fits*. Below that threshold, the three-tier dispatch (one-shot → disk-edges → merged-shards) keeps PiPNN at or under Vamana's RAM footprint.

On BigANN 10M, 16 threads:

| RAM budget | PiPNN strategy | PiPNN build | Vamana build |
|---:|---|---:|---:|
| ≥ 12 GB | one-shot | 80–133s | 358s |
| 6–12 GB | disk-edges | ~126s | 358s |
| 3–6 GB | merged-shards | ~332s | ~358s (partitioned) |

**Neither algorithm uses surplus RAM to build faster.** PiPNN's wall-clock is bottlenecked by HashPrune + GEMM; extra RAM headroom past the working set doesn't help (more memory *channels* / bandwidth does, but that's a hardware property, not a budget knob). The honest framing: PiPNN trades a higher *minimum* RAM budget for a substantially faster build at that budget. Validation: see **M6**.

### Hybrid update model (Stage 2 direction)

Both algorithms write the same disk format, so a graph built by either can be loaded and (once in memory) extended by Vamana. Production update story:

- **Full rebuild → PiPNN.** Several times faster than Vamana.
- **Incremental insert → in-memory Vamana.** Unchanged — applies to the in-memory graph; the disk index file is not mutated in place.
- **Rebuild triggers.** Embedding rotation, schema/parameter retuning, large batch inserts, or periodic safety rebuilds — not just gradual recall drift. DiskANN's claim that incremental updates keep recall healthy still stands; PiPNN just makes the eventual rebuild cheaper.

This is why we don't need PiPNN to support `insert(point)` — the disk format is the integration point between batch and incremental.

### Two-stage rollout

- **Stage 1 (this RFC).** Land PiPNN as an alternative builder for the disk-index full-build path (initial builds *and* rebuilds), behind a `pipnn` Cargo feature. Vamana stays default.
- **Stage 2 (separate proposal, gated by Stage-1 milestones).** Retire Vamana's full-build path; keep Vamana for in-memory incremental inserts.

**In-memory PiPNN build is not in any stage.** The in-mem path exists for streaming construction — exactly what PiPNN's batch design cannot do. See "Out of scope" below.

### Goals (Stage 1)

1. **Pluggable disk builder** — a selector that routes Vamana (default) vs. PiPNN (opt-in). No behavior change at existing call sites.
2. **Disk-format compatibility** — byte-identical to Vamana's output; search/PQ/storage layouts unchanged.
3. **API backward compatibility** — `DiskIndexBuilder`, `IndexConfiguration`, JSON schema all stay additive.
4. **Feature parity for the full-build role** — deliver the Vamana disk-build capabilities PiPNN still lacks (quantization, label filters).
5. **Documented memory mitigation** — a three-tier build path that brings PiPNN's peak RSS to or under Vamana's at a documented build-time cost.

## Proposal

### Workspace structure

Add a crate `diskann-pipnn` depending on `diskann`, `diskann-linalg`, `diskann-vector`, `diskann-quantization`, `diskann-utils`. **It does NOT depend on `diskann-disk`** — that would form a cycle with the consumer-side feature gate. Data flows one-way: PiPNN produces `Vec<Vec<u32>>`, `diskann-disk` consumes it behind its `pipnn` feature.

```text
diskann, diskann-linalg, diskann-quantization, diskann-vector, diskann-utils
                            │  (shared deps, no edges to builders)
                ┌───────────┴────────────┐
        diskann-pipnn              diskann-disk
                │                         ↑ feature "pipnn"
                └───→  Vec<Vec<u32>>  ────┘
```

### `BuildAlgorithm` enum

```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
pub enum BuildAlgorithm {
    #[default]
    Vamana,

    #[cfg(feature = "pipnn")]
    PiPNN {
        c_max: usize, c_min: usize, p_samp: f64,
        fanout: Vec<usize>, leaf_k: usize, replicas: usize,
        l_max: usize, num_hash_planes: usize,
        final_prune: bool, leader_cap: usize,
        saturate_after_prune: bool,
        num_threads: usize,  // 0 = all logical CPUs
    },
}
```

`DiskIndexBuildParameters` gains a `build_algorithm` field defaulting to `Vamana`. The JSON config gains an optional `build_algorithm` block.

**`num_threads` is not a RAM knob.** Per-thread overhead is small (~16–20 MB/thread: stripe buffers + leaf-build scratch). Use `build_ram_limit_gb` to bound RAM.

**Feature-flag deserialization applies to JSON configs only, not index files.** An index file built by either algorithm loads with or without the `pipnn` feature. A JSON config with `"algorithm": "PiPNN"` fed to a binary built without the feature fails fast with `unknown variant 'PiPNN'`. Configs that omit `build_algorithm` parse identically across feature builds — not a backward-compat regression.

### Builder dispatch

```rust
match build_parameters.build_algorithm() {
    BuildAlgorithm::Vamana => self.build_inmem_vamana_index().await,
    #[cfg(feature = "pipnn")]
    BuildAlgorithm::PiPNN { .. } => self.build_inmem_pipnn_index().await,
}
```

The PiPNN branch produces `Vec<Vec<u32>>` via `diskann_pipnn::builder::build_typed`, then hands it to the existing `DiskIndexWriter`. PQ training and disk-sector layout are reused.

### Compatibility

| Surface | Status |
|---|---|
| On-disk graph format | unchanged |
| PQ/SQ codes on disk | unchanged |
| Search API | unchanged |
| Public Rust types | additive only (new field with default) |
| Benchmark JSON config | additive only (new optional field) |

A search-only consumer cannot tell which builder produced the index.

### Feature gating

- `diskann-disk` gains a `pipnn` Cargo feature. Default features do **not** include it.
- With the feature off: `BuildAlgorithm::PiPNN` does not exist at the type level. No runtime branch, no extra binary size, no `diskann-pipnn` dependency.

## Trade-offs

### Batch-only (algorithmic, not implementation)

The PiPNN paper "eliminates search from graph-building" — partition first, then one batched GEMM per leaf. The batching advantage **requires knowing leaf membership before computing distances**; at batch size 1, PiPNN reduces to per-point distance work no faster than Vamana's greedy insert. Two phases (partition, leaf k-NN) are batch-by-design; HashPrune and final RobustPrune happen to be online, but you've already paid for the batch phases by the time you reach them. This is why Vamana keeps the incremental role.

### Memory vs. build speed

PiPNN holds more working memory than Vamana — dominated by the **HashPrune reservoir** (`l_max × 8 B` per point). On BigANN 10M (`c_max=256, fanout=[10,3], leaf_k=3, l_max=64`):

| | PiPNN one-shot | Vamana |
|---|---:|---:|
| Peak RSS | 10.8 GB | 6.3 GB |

The +4.5 GB delta is the working set the algorithm needs, not a bug. Mitigation via the three-tier build (dispatched by the existing `build_ram_limit_gb` knob):

| Strategy | Peak RSS | Build | Recall@10 L=50 | Trigger |
|---|---:|---:|---:|---|
| One-shot | 10.8 GB | 133s | 95.00% | RAM ≥ ~32 GB |
| Disk-edges | 6.4 GB | 126s | 95.00% | RAM 8–32 GB |
| Merged shards | 3.3 GB | 332s | 95.31% | RAM 4–8 GB |

Disk-edges matches Vamana's RAM at ~3× the build speed. Merged-shards uses *less* RAM than Vamana (3.3 vs. 6.3 GB) at a 2.5× build-time cost.

### Alternatives considered

| | Choice | Rejected because |
|---|---|---|
| A | (Chosen) Add PiPNN behind a feature flag | — |
| B | Replace Vamana immediately | PiPNN lacks checkpoint / full quantization / label-filter parity; production-validation gap |
| C | Separate top-level crate/binary | Duplicates PQ training, disk writer, search pipeline — maintenance burden, no compatibility benefit |

### Algorithm risks

Recall depends on partition overlap (`fanout`) and reservoir size (`l_max`). Parameter space is larger than Vamana's `R`/`L_build`. Stage 1 mitigates by keeping Vamana as default and shipping reference parameter sets per workload class.

## Benchmark Results

Azure `Standard_L16s_v3`, 16 threads, NVMe, `RUSTFLAGS=-C target-cpu=native`.

### Build time

| Dataset | Vamana | PiPNN | Speedup |
|---|---:|---:|---:|
| Enron 1M (384d) | 70s | 13s | 5.4× |
| BigANN 10M (128d) | 358s | 80s | 4.5× |
| Enron 10M (384d) | 844s | 133s | 6.3× |

### BigANN 10M — recall × QPS

Default PiPNN (`c_max=256, fanout=[10,3], leaf_k=3, l_max=64, pq_chunks=64`) vs. Vamana (`R=64, L=64, pq_chunks=64`):

| L | PiPNN R@10 | PiPNN QPS | Vamana R@10 | Vamana QPS |
|---|---:|---:|---:|---:|
| 10 | 77.76% | 10,670 | 79.23% | 11,618 |
| 50 | 96.31% | 5,574 | 97.10% | 5,940 |
| 100 | 98.61% | 3,430 | 99.01% | 3,568 |

With a higher-recall config (`c_max=512, fanout=[10,4], l_max=128, final_prune`), PiPNN matches/exceeds Vamana at L=50 (97.22%) and L=100 (99.21%) at 143s (still 2.5× faster).

### Enron 10M (384d) — recall × QPS

PiPNN (`c_max=256, fanout=[8,3], leaf_k=2, l_max=64, pq_chunks=192`) vs. Vamana (`R=64, L=72, pq_chunks=192`):

| L | PiPNN R@1000 | PiPNN QPS | Vamana R@1000 | Vamana QPS |
|---|---:|---:|---:|---:|
| 1000 | 89.99% | 378 | 89.33% | 384 |
| 2000 | 96.46% | 192 | 95.36% | 195 |
| 3000 | 97.74% | 129 | 96.68% | 130 |

PiPNN beats Vamana on recall at every L at parity QPS — and 6.3× faster build.

## Future Work — Stage-1 Milestones

Stage 1 covers build-from-scratch and full rebuilds with PiPNN. M0 ships in this RFC; M1–M6 are follow-on work, parallelizable where dependencies allow.

### M0 — Skeleton integration (this RFC)
Crate, `BuildAlgorithm` enum, dispatch behind `pipnn` Cargo feature. JSON config gains optional `build_algorithm`. CI smoke test (SIFT-1M) with `--features pipnn`.

### M1 — Quantization parity
Extend PiPNN beyond `SQ1` to `SQ_2/4/8`, reusing the trained `ScalarQuantizer`. **Pass:** SQ_8 recall within 0.5% of FP on BigANN 10M and Enron 10M.

### M2 — Label-filtered indexes
Run filter benchmark configs with `BuildAlgorithm::PiPNN`; confirm filter-recall within ±1% of Vamana. Partition may need label-aware leaf assignment for high-cardinality labels.

### M3 — Three-tier memory dispatch
Implement and validate the disk-edges + merged-shards paths selected by `build_ram_limit_gb`. **Pass:** at `build_ram_limit_gb=4`, PiPNN-merged on BigANN 10M has peak RSS ≤ 4 GB and recall within 1% of one-shot.

Two disk-edges variants are on the table: (i) materialize all leaf edges then stream HashPrune (current prototype), or (ii) interleave leaf-build + HashPrune in chunks. The second avoids full edge-set materialization at the cost of a second partition pass.

### M4 — Fixed-resource trade-off validation
Validates the **trade-off hypothesis** from the Problem Statement.

- **Setup.** Lock CPU/SSD on a fixed worker; enforce RAM via cgroups. Sweep RAM `{3, 6, 8, 12, 16, 24, 32}` GB on BigANN 10M; include rows for Enron 10M and a 100M-scale dataset.
- **Cells.** Vamana one-shot, Vamana partitioned, PiPNN one-shot, PiPNN disk-edges, PiPNN merged-shards. OOM cells are valid results.
- **Metrics.** Wall-clock, peak RSS, CPU util, SSD bytes, recall@K, QPS — reported as **vectors/min/worker** for cross-shape comparison.
- **Pass.** Documented matrix with a clearly-better algorithm (or tie) per budget at matching recall. Surprises are Stage-1 blockers.

### M5 — Production validation: recall × QPS × dimensionality
End-to-end on the full workload mix. Datasets: BigANN, Enron, plus one production-representative. Scales 10M and 100M (billion if hardware permits). Metrics `squared_l2` and `cosine_normalized`. **Pass:** per cell, PiPNN recall@K within ±1% of Vamana's at matching QPS, or higher QPS at matching recall.

### M6 — Operational readiness
Telemetry (per-phase timing + RSS via existing OTel tracer), permanent docs replacing experimental `CLAUDE.md` notes, runbook (OOM, partition timeout, `l_max` saturation), default parameter recommendations per workload class.

### Deferred to Stage 2

- **Hybrid update model validation.** End-to-end validation of the Stage-2 loop — PiPNN build → incremental Vamana inserts → recall-decay curve → PiPNN rebuild — belongs with the Stage-2 proposal that actually adopts the hybrid model. Stage 1 exercises only the full-build path. The disk-format-compatibility check (Vamana's in-mem insert path reading a PiPNN-produced graph) is a one-shot sanity test that can run at Stage 2 entry.

- **Checkpoint / resume.** Vamana's streaming checkpoint design doesn't fit PiPNN's batch phases. Useful boundaries (partition output, post-extract) would need a different scheme, and operational value is lower (PiPNN's BigANN-10M build is ~80s). Defer until Stage 2 reveals the production rebuild cadence.

  *Determinism note:* PiPNN is rayon-parallel — byte-identical output across runs is not free (would need fixed thread schedule, deterministic reductions, seeded LSH). The right validation criterion for any future resumed-build test is **recall parity**, not byte-identity.

### Out of scope: not part of any stage

- **In-memory PiPNN build.** The in-mem `DiskANNIndex` exists for streaming construction — exactly what PiPNN can't do efficiently. Building one from PiPNN adjacency lists is mechanically possible but offers no incremental capability and would force `diskann-pipnn` to depend on the in-mem graph crate. If a non-streaming in-mem consumer ever needs PiPNN's speed: build to disk, then load.
- **Build-time PQ distance kernel.** Not used by Vamana in production today.
- **PiPNN incremental insert/delete API.** The hybrid update model (Vamana inserts on the in-memory graph, PiPNN for full rebuilds) removes the need.
- **Frozen-point semantics.** PiPNN writes the medoid as the single frozen start point — already byte-compatible with Vamana's default.
- **Multi-vector index support.** Revisit only if a production workload requires it.

## References

1. [PiPNN: Pick-in-Partitions Nearest Neighbors (arXiv:2602.21247)](https://arxiv.org/abs/2602.21247)
2. [Vamana / DiskANN (NeurIPS 2019)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
3. Existing disk index layout: `diskann-disk/src/storage/`
4. Existing Vamana builder: `diskann-disk/src/build/builder/build.rs`
