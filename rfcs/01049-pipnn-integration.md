# Integrate PiPNN as an Alternative Graph-Index Build Algorithm

| | |
|---|---|
| **Authors** | Weiyao Luo |
| **Contributors** | DiskANN team |
| **Created** | 2026-05-11 |
| **Updated** | 2026-05-11 |

## Summary

Add **PiPNN** (Pick-in-Partitions Nearest Neighbors, arXiv:2602.21247) as a second graph-construction algorithm for DiskANN's disk index. PiPNN produces a graph byte-compatible with Vamana's disk format and search API, at **up to 6.3× lower build time** on the workloads we have measured. Vamana remains the default and the only algorithm supported for incremental inserts; PiPNN is the proposed faster path for full rebuilds.

## Motivation

### Background

DiskANN currently builds the disk index with a single algorithm — **Vamana** (`diskann-disk/src/build/builder/`). Vamana incrementally inserts each point into a graph, running a greedy search + `RobustPrune` for each insertion, producing the on-disk format documented in `diskann-disk/src/storage/`.

**PiPNN** (Pick-in-Partitions Nearest Neighbors, arXiv:2602.21247) is a partition-based **batch** graph builder, in contrast to Vamana's **incremental** insert + prune. The construction has four phases:

1. **Partition** — Randomized Ball Carving (RBC) recursively splits the dataset into small *overlapping* leaf clusters. Each point lands in `fanout` of its nearest cluster leaders at every recursion level, so every point appears in multiple leaves. Recursion stops when a cluster fits a configured leaf-size cap (`c_max`, typically 256–1024 points).
2. **Local k-NN per leaf** — For each leaf, compute the full pairwise distance matrix in one batched GEMM call, then extract each point's `leaf_k` nearest neighbors inside the leaf. GEMM batching is the source of most of PiPNN's wall-clock advantage over per-point greedy search.
3. **HashPrune merge** — Edges from all leaves are merged into a per-point reservoir of bounded size (`l_max`, ~64–128). The pruner is keyed by an LSH **angular bucket** of each candidate neighbor: at most one candidate per bucket is retained, and on collision the closer candidate wins. This produces a diverse short-list per point using O(`l_max`) memory per node and O(1) amortized insert work.
4. **Optional final prune** — A single RobustPrune-style pass (same algorithm Vamana uses, with a configurable `alpha`) applies geometric occlusion to the HashPrune candidates. Used when the workload benefits from explicit graph diversification.

The output is `Vec<Vec<u32>>` adjacency lists in the same shape Vamana produces, then handed to the existing disk-layout writer. PQ training and search-side data structures are unchanged.

The structural trade-off: Vamana is sequential per insert with fine-grained parallelism and memory-efficient; PiPNN is batch-parallel across leaves with higher peak working memory in exchange for far shorter wall-clock builds.

### Problem Statement

Vamana's incremental design scales linearly in points × per-insert search cost, which makes full rebuilds expensive at the scales we operate. Measured baselines:

| Dataset | Vamana build time |
|---|---:|
| Enron 1M (1.087M × 384, fp16, cosine_normalized) | 70s |
| BigANN 10M (10M × 128, fp16, squared_l2) | 358s |
| Enron 10M (10M × 384, fp16, cosine_normalized) | 844s |

Frequent rebuilds (driven by data churn or parameter sweeps) and full rebuilds at 10M-scale and above are the bottleneck. PiPNN's offline benchmarks at matching recall budgets complete the same builds **up to 6.3× faster** while writing the same disk format (full numbers in the Benchmark Results section). This RFC proposes landing PiPNN so teams can opt into faster builds and so we can collect production-relevant signal on whether PiPNN can eventually replace Vamana's full-rebuild path.

#### Hybrid update model (Stage 2 direction)

Vamana and PiPNN write the same on-disk graph format, so a graph built by either algorithm can be *read* (and incrementally edited) by either. We exploit this for the production update story:

- **Bulk / full rebuild → PiPNN.** When data churn is large enough to justify a full rebuild, PiPNN is used because it is several times faster than Vamana at this job.
- **Incremental insert → Vamana.** Between full rebuilds, individual inserts use Vamana's existing greedy-search + RobustPrune insert path. PiPNN's batch design has no natural single-point-insert API and we do not plan to build one.
- **Quality decay → trigger PiPNN rebuild.** When recall on the live graph degrades past a configured threshold (driven by accumulated incremental inserts), the system schedules a PiPNN full rebuild from the current dataset snapshot.

Because both algorithms produce the same disk format, switching between "fresh PiPNN build" and "Vamana-edited delta" is transparent to search-side consumers. This answers "should PiPNN implement incremental inserts?" — no, we keep Vamana for that, and use the disk index format as the integration point.

#### Two-stage rollout

- **Stage 1 (this RFC):** Land PiPNN behind a build-algorithm selector. Vamana stays default; PiPNN is opt-in. Stage 1 has explicit milestones (in Future Work) that gate readiness for Stage 2.
- **Stage 2 (separate proposal, conditional on Stage 1 milestones):** Retire the Vamana **full-rebuild** path. Vamana remains the implementation for incremental inserts via the hybrid model above.

### Goals

1. **Algorithm-level pluggability**: introduce a build-algorithm selector to the build pipeline that routes between Vamana (existing) and PiPNN (new). Existing build sites continue to default to Vamana with no behavior change.
2. **Disk format compatibility**: the PiPNN-built index is byte-compatible with Vamana-built indexes on disk — search, PQ, and storage layouts are unchanged. This is the foundation for the hybrid update model.
3. **Public API compatibility**: the disk-index public API surface (`DiskIndexBuilder::new`, `IndexConfiguration`, `DiskIndexWriter`, JSON config schema) remains backward-compatible. PiPNN configuration is added under a new tagged enum variant.
4. **Feature-parity milestones**: deliver the Vamana capabilities PiPNN needs for a full-rebuild role in production (see Future Work below).
5. **Documented memory mitigation**: provide a configuration knob (three-tier build) that brings PiPNN's peak RSS to or below Vamana's at the cost of build time.

## Proposal

### Workspace structure

Add a new crate, `diskann-pipnn`, that depends on the existing `diskann`, `diskann-disk`, `diskann-linalg`, `diskann-vector`, `diskann-quantization`, and `diskann-utils` crates. PiPNN lives outside `diskann-disk` so the core disk path has no compile-time dependency on PiPNN; the disk builder takes a typed `BuildAlgorithm` and only depends on PiPNN behind a feature flag.

```text
diskann/                    # core types, traits, search
diskann-disk/               # disk index layout, builder, search
  └── feature "pipnn"       # opt-in dependency on diskann-pipnn
diskann-pipnn/              # new: PiPNN builder
diskann-linalg/             # GEMM/SVD (used by both Vamana and PiPNN)
diskann-quantization/       # PQ/SQ training (used by both)
```

### `BuildAlgorithm` enum

Introduce a tagged enum in `diskann-disk/src/build/configuration/build_algorithm.rs`:

```rust
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "algorithm")]
pub enum BuildAlgorithm {
    /// Default Vamana graph construction.
    #[default]
    Vamana,

    /// PiPNN: Pick-in-Partitions Nearest Neighbors.
    #[cfg(feature = "pipnn")]
    PiPNN {
        c_max: usize,                // maximum leaf partition size
        c_min: usize,                // minimum cluster size before merging
        p_samp: f64,                 // RBC leader sampling fraction
        fanout: Vec<usize>,          // per-level fanout
        leaf_k: usize,               // k-NN within each leaf
        replicas: usize,             // independent partitioning passes
        l_max: usize,                // HashPrune reservoir cap
        num_hash_planes: usize,      // LSH hyperplane count
        final_prune: bool,           // optional RobustPrune final pass
        leader_cap: usize,           // hard cap on leaders per level
        saturate_after_prune: bool,
    },
}
```

`Vamana` is the `Default` so every existing call site that constructs `DiskIndexBuildParameters` without specifying an algorithm keeps the existing behavior.

`DiskIndexBuildParameters` gains a `build_algorithm: BuildAlgorithm` field and a constructor pair: `new` (defaults to Vamana, no PiPNN dep) and `new_with_algorithm` (explicit). The JSON schema for benchmark configs gains an optional `build_algorithm` block that, when present, deserializes via `#[serde(tag = "algorithm")]` into one of the variants above.

### Builder dispatch

In `DiskIndexBuilder::build()` (or the new equivalent), dispatch on `BuildAlgorithm`:

```rust
match build_parameters.build_algorithm() {
    BuildAlgorithm::Vamana =>
        self.build_inmem_vamana_index().await,
    #[cfg(feature = "pipnn")]
    BuildAlgorithm::PiPNN { .. } =>
        self.build_inmem_pipnn_index().await,
}
```

The PiPNN path produces a `Vec<Vec<u32>>` adjacency list using `diskann_pipnn::builder::build_typed`, then hands it to the existing disk-layout writer (`DiskIndexWriter`) which emits the same format Vamana does (header, per-node adjacency, frozen start-point block). PQ training and disk-sector layout are reused unchanged.

### Compatibility surface

| Surface | Status |
|---|---|
| On-disk graph format (header + adjacency + frozen start point) | unchanged |
| PQ codes / SQ codes on disk | unchanged (trained the same way) |
| Search API (`DiskANNIndex::search`, beam_width, search_list, recall_at, num_nodes_to_cache, search_io_limit, filters API) | unchanged |
| Public Rust types (`IndexConfiguration`, `DiskIndexWriter`, `DiskIndexBuildParameters`) | additive only (new field with default) |
| Benchmark JSON config | additive only (new optional `build_algorithm` field) |
| C/C++ FFI (if any) | unchanged |

Since the produced graph and PQ/SQ artifacts are byte-identical in format, a search-only consumer cannot tell which builder wrote the index.

### Feature gating

- The `diskann-disk` crate gains a `pipnn` Cargo feature. With it disabled, `BuildAlgorithm::PiPNN` does not exist at the type level — no runtime branch, no extra binary size, no dependency on `diskann-pipnn`.
- The benchmark binary and any production binary that wants PiPNN must enable the `pipnn` feature on `diskann-disk` (or transitively).
- The default features set continues to not include `pipnn`, matching the principle that the existing Vamana path is what ships unchanged.

### What this RFC does *not* change

- Distance metrics, vector representations, storage layouts.
- The greedy-search / RobustPrune logic used by Vamana — both stay as-is for the Vamana path. PiPNN brings its own equivalents internally (HashPrune + optional final RobustPrune).
- PQ training, search-time decoders, and the disk layout.
- Public traits, types, or method signatures outside the new optional fields/variants described above.

## Trade-offs

### PiPNN is algorithmically batch-only

This is a property of the algorithm, not of our implementation. The PiPNN paper (arXiv:2602.21247) is explicit that the design departs from incremental methods by "eliminating search from the graph-building process altogether": instead of running a greedy search for each new point's neighbors, PiPNN partitions the dataset, then computes neighbors for all points within each leaf as a single batched operation. The paper describes no per-point insertion algorithm and reports no streaming results. The framing throughout is "fast one-shot construction on a static dataset."

Where this batch assumption is load-bearing:

- **Partition (RBC)** samples leaders from the global dataset distribution and recursively splits into overlapping leaves. Leader quality depends on representativeness of the full data. Adding new points to an existing partition works mechanically (assign to fanout nearest existing leaders), but the *partition itself* is a one-shot decision — the cluster structure can drift as the data distribution shifts.
- **Leaf k-NN via GEMM** is where PiPNN gets its speed. A leaf's pairwise distance matrix is computed in one batched matrix multiplication and amortizes per-leaf overhead across `c_max²` distance evaluations. **This is the algorithm's central optimization, and it requires knowing the leaf membership before computing distances.** Inserting one point against an existing leaf reduces to `c_max` individual distance computations, which is no faster than what Vamana already does per insert — the batching advantage evaporates at batch size 1.
- **HashPrune** is the one PiPNN component that *is* online — it accepts an arbitrary stream of `(point, neighbor, distance)` edges and maintains a bounded reservoir per point. So the merge stage doesn't structurally object to incremental updates. But by the time you have edges to feed it, you've already paid for the partition assignment and the per-leaf distance work.
- **Final RobustPrune** is per-point and naturally re-runnable.

In other words: of PiPNN's four phases, two (partition, leaf k-NN) are batch-by-design and would need to be replaced for true incremental construction. Replacing them defeats the purpose — the algorithm degenerates into something more like Vamana but without Vamana's online-friendly graph-search structure.

The realistic alternatives for "PiPNN-like incremental" are all mini-batch variants (accumulate N new points → run a partial partition + leaf-build), which works fine but isn't really an incremental algorithm. Vamana already does per-point online inserts correctly; we keep it for that role.

This is why the Motivation section's hybrid update model exists: **PiPNN for full rebuilds, Vamana for inserts**, with the disk format as the integration point. PiPNN is not a drop-in replacement for code paths that rely on `insert(point)` semantics — and the limitation is the algorithm, not just our crate's API surface.

### Memory vs build speed

PiPNN's batch design holds more working memory during build than Vamana's incremental design. The dominant overhead is the **HashPrune reservoir** — a bounded per-point candidate list (`l_max × 8 bytes` per point) that PiPNN needs to merge edges from overlapping leaves. Vamana has no equivalent: it writes neighbors directly into the final adjacency list as it inserts each point.

For example, on BigANN 10M (10M × 128 fp16, `c_max=256, fanout=[10,3], leaf_k=3, l_max=64`):

| | PiPNN one-shot | Vamana |
|---|---:|---:|
| Peak RSS | 10.8 GB | 6.3 GB |

That delta — roughly **+4.5 GB**, dominated by HashPrune (`10M × 64 × 8 ≈ 5 GB`) plus smaller PiPNN-only working buffers (LSH sketches, partition leaf indices) — is the cost of the batch design and not a bug. It is the working set the algorithm explicitly needs. The next subsection describes the mitigation.

### Memory mitigation: three-tier build

For deployments that need PiPNN's build speed but cannot afford its working memory, we reuse the same **`MemoryBudget`** parameter Vamana already uses for sharded builds. When `build_ram_limit_gb` is below a threshold, PiPNN switches to a chunked path that spills HashPrune reservoirs to disk between leaf batches. Measurements on the same dataset as the table above (BigANN 10M):

| Strategy | Peak RSS | Build time | Recall@10 L=50 | Trigger |
|---|---:|---:|---:|---|
| **One-shot** (in-memory) | 10.8 GB | 133s | 95.00% | RAM ≥ ~32 GB |
| **Disk-edges** (per-batch reservoir flush) | 6.4 GB | 126s | 95.00% | RAM 8-32 GB |
| **Merged shards** (per-shard graph, then merge) | 3.3 GB | 332s | 95.31% | RAM 4-8 GB |

The merged-shards path **uses less peak RSS than Vamana** (3.3 GB vs Vamana's 6.3 GB on this same dataset) at a 2.5× build-time cost. The disk-edges path matches Vamana on RAM at 3× the build speed.

The control knob is the existing `build_ram_limit_gb` config; no new parameter is introduced. The dispatch happens inside `build_inmem_pipnn_index()`.

### Stage-1 separate path vs immediate-replace

We considered three options:

**A. (Chosen) Add PiPNN as an alternative behind a feature flag.** Default is Vamana, opt-in for PiPNN. Existing users see no change. Lets us collect production validation signal without risk.

**B. Replace Vamana with PiPNN immediately.** Cleaner code, smaller binary. Rejected because: (1) PiPNN lacks checkpoint, full quantization, and label-filtered search support today — replacing now is a regression; (2) we have not validated PiPNN under the full production workload mix; (3) recall behavior on edge-case datasets is not yet characterized at production scale.

**C. Maintain PiPNN as a fully separate top-level binary/crate.** Rejected because it would duplicate the PQ training, disk-layout writer, search pipeline, and benchmark harness — adding maintenance burden with no compatibility benefit.

### Algorithm risks

PiPNN's recall depends on partition overlap (controlled by `fanout`) and reservoir size (`l_max`). On the workloads in the benchmark section recall matches or beats Vamana at the chosen settings, but the parameter space is larger than Vamana's `R`/`L_build`. Stage-1 mitigates by keeping Vamana as the default and providing reference parameter sets in code comments and benchmark configs.

## Benchmark Results

All benchmarks run on Azure `Standard_L16s_v3` (Intel Xeon Platinum 8370C, 16 threads, NVMe), with `RUSTFLAGS=-C target-cpu=native`.

### Build time

| Dataset | Vamana | PiPNN (one-shot) | Speedup |
|---|---:|---:|---:|
| Enron 1M (1.087M × 384, fp16, cosine_normalized) | 70s | 13s | 5.4× |
| BigANN 10M (10M × 128, fp16, squared_l2) | 358s | 80.2s | 4.5× |
| Enron 10M (10M × 384, fp16, cosine_normalized) | 844s | 133s | 6.3× |

### Recall / QPS — BigANN 10M

Config: PiPNN `c_max=256, fanout=[10,3], leaf_k=3, l_max=64, hp=12, pq_chunks=64, no final_prune`. Vamana `R=64, L=64, pq_chunks=64`.

| L | PiPNN Recall@10 | PiPNN QPS | Vamana Recall@10 | Vamana QPS |
|---|---:|---:|---:|---:|
| 10 | 77.76% | 10,670 | 79.23% | 11,618 |
| 50 | 96.31% | 5,574 | 97.10% | 5,940 |
| 100 | 98.61% | 3,430 | 99.01% | 3,568 |

With higher-recall PiPNN config (`c_max=512, fanout=[10,4], leaf_k=3, l_max=128, final_prune`), PiPNN exceeds Vamana on recall at L=50 (97.22% vs 97.10%) and L=100 (99.21% vs 99.01%) at the cost of 143s build time (still 2.5× faster than Vamana's 358s).

### Recall / QPS — Enron 10M (384d)

Config: PiPNN `c_max=256, fanout=[8,3], leaf_k=2, l_max=64, hp=14, pq_chunks=192`. Vamana `R=64, L=72, pq_chunks=192`.

| L | PiPNN Recall@1000 | PiPNN QPS | Vamana Recall@1000 | Vamana QPS |
|---|---:|---:|---:|---:|
| 1000 | 89.99% | 378 | 89.33% | 384 |
| 1500 | 95.19% | 255 | 94.12% | 258 |
| 2000 | 96.46% | 192 | 95.36% | 195 |
| 2500 | 97.23% | 154 | 96.15% | 155 |
| 3000 | 97.74% | 129 | 96.68% | 130 |

PiPNN beats Vamana on recall at every L on the 384d Enron 10M workload, at parity QPS and 6.3× faster build.

## Future Work

The Stage 1 milestones below are gating items for Stage 2 (retiring Vamana's full-rebuild path). Each must be addressed before that proposal is credible. M0 is the foundation shipped by this RFC; M1–M8 are deferred to follow-on work and ordered by dependency, not strict calendar sequence — some can run in parallel.

### M0 — Skeleton integration

The foundation that ships first.

- **Scope:** introduce the `diskann-pipnn` crate, the `BuildAlgorithm` enum, and the dispatch in `DiskIndexBuilder` behind a `pipnn` Cargo feature.
- **Config surface:** JSON config gains an optional `build_algorithm` block; default behavior unchanged.
- **Compatibility:** PiPNN-built indexes are read by the existing search pipeline unchanged (the on-disk format is identical) and produce recall numbers within the tolerances the existing disk-index test suite enforces.
- **CI:** benchmark binary runs with `--features pipnn` on a small smoke test (SIFT-1M).

M1–M5 close the feature-parity gaps; M6–M8 are validation and operational readiness.

### M1 — Feature parity: in-memory build / search

Vamana supports both a **disk-resident** build/search path (via `diskann-disk`) and an **in-memory only** path (via `diskann::graph::index::DiskANNIndex`). PiPNN today only produces graphs handed to `DiskIndexWriter`; an in-mem-only consumer that wants PiPNN's speed has no entry point.

- **Scope:** expose `diskann_pipnn::build_typed` output (`Vec<Vec<u32>>`) as a populated in-memory `DiskANNIndex` so callers can build + search without touching disk.
- **API:** add `diskann_pipnn::build_into_inmem_index(...)` returning an in-memory index that is read by the existing `DiskANNIndex::search` path unchanged.
- **Validation:** in-mem search recall on Enron 1M with PiPNN-built graph matches the disk-build + load round-trip recall within noise.

### M2 — Feature parity: checkpoint / resume

- **Scope:** add checkpoint/resume to the PiPNN build pipeline using the existing `CheckpointManager` / `ChunkingConfig` infrastructure in `diskann-disk/src/build/chunking/`.
- **Boundaries:** natural checkpoint points are partition output (`Vec<Leaf>`), per-leaf HashPrune flush, post-extract graph.
- **Behavior:** matches Vamana's — a killed build resumes from the last checkpoint instead of starting over.
- **Validation:** kill-and-resume test on BigANN 10M at three different checkpoint phases; final graph byte-identical to a non-interrupted build given the same seeds.

### M3 — Feature parity: quantized vector support

PiPNN currently has only a `SQ1` (1-bit) build path.

- **Scope:** extend the build to accept `QuantizationType::SQ { nbits, standard_deviation }` for the same `nbits` values Vamana supports (`SQ_2`, `SQ_4`, `SQ_8`).
- **Reuse:** trained `ScalarQuantizer` from `diskann-quantization`; do not duplicate quantizer training.
- **Implementation:** the leaf-build distance kernel needs an `nbits`-aware path. Today the kernel is either FP (GEMM) or 1-bit Hamming.
- **Validation:** PiPNN at `SQ_8` produces recall within 0.5% of FP for BigANN 10M and Enron 10M, matching the Vamana SQ_8 baseline.

*Note: build-time Product Quantization (PQ-distance during graph construction) is not currently used by Vamana in any production path and is out of scope.*

### M4 — Feature parity: label-filtered indexes

PiPNN-built graphs already work with the existing search-time filter pipeline (`diskann-label-filter`) because the disk format is the same. The build-time flow for filter-aware indexes has not been exercised end-to-end.

- **Scope:** run the filter benchmark JSON configs with `BuildAlgorithm::PiPNN`; confirm filter-recall numbers match Vamana's.
- **Risk:** the partition phase may need label-aware leaf assignment for high-cardinality labels.
- **Validation:** filter-recall on a representative labeled dataset within ±1% of Vamana's filter-recall.

### M5 — Memory mitigation: three-tier dispatch

Implement two memory-constrained PiPNN paths and select among them via the existing `build_ram_limit_gb` knob.

- **Disk-edges:** HashPrune reservoirs spill to disk between leaf batches when `MemoryBudget` is below a threshold (currently ~8 GB for 10M-scale workloads).
- **Merged-shards:** per-shard graphs built independently then merged, mirroring Vamana's `build_merged_vamana_index` pipeline at `diskann-disk/src/build/builder/build.rs:327`. The existing shard merger is reused.
- **Dispatch:** inside `build_inmem_pipnn_index()` — no new public parameter.
- **Validation:** at `build_ram_limit_gb=4`, the PiPNN-merged path on BigANN 10M produces peak RSS ≤ 4 GB and recall within 1% of one-shot PiPNN.

### M6 — Production validation: recall × QPS × dimensionality matrix

End-to-end validation on the full production workload mix.

- **Datasets:** at minimum three families (BigANN, Enron, plus one production-representative).
- **Scales:** 10M and 100M; one billion-scale sample if hardware permits.
- **Metrics:** `squared_l2` and `cosine_normalized`.
- **Pass criterion:** for each (dataset, scale, metric) cell, PiPNN recall@K is within Vamana's recall ±1% at matching QPS, *or* higher QPS at matching recall.
- **Out-of-band cells** are documented as "PiPNN not yet recommended for X" rather than blocking Stage 2 entirely.

### M7 — Production validation: hybrid update model

Validate the Stage-2 hybrid loop end-to-end.

- **Sequence:** PiPNN build → N incremental Vamana inserts representing production churn → measure recall decay vs. graph age → trigger PiPNN rebuild from snapshot → confirm post-rebuild recall restored.
- **Output:** a recommended "quality decay threshold" for production rebuild triggers, derived from the measured decay curve.
- **Disk-format compatibility test:** confirm Vamana's incremental-insert path reads PiPNN-produced graphs correctly. This is the load-bearing compatibility check for the hybrid model.

### M8 — Operational readiness

- **Telemetry:** emit per-phase timing and peak RSS via the existing OpenTelemetry tracer, comparable to Vamana's spans.
- **Documentation:** replace experimental notes in `CLAUDE.md` with a permanent doc covering recommended parameters per workload class (dim × scale × metric).
- **Runbook:** failure modes (OOM under one-shot, partition timeout, `l_max` saturation), diagnosis, recovery.
- **Defaults:** parameter recommendations baked into the JSON config builder so users don't hand-tune for common cases.

### Out of scope (intentionally not on this list)

- **Build-time PQ distance kernel.** Not used by Vamana in production paths today; deferred indefinitely.
- **PiPNN incremental insert API.** The hybrid model (PiPNN rebuild + Vamana inserts) removes the need.
- **PiPNN incremental delete API.** Same reason.
- **Frozen-point semantics differences.** PiPNN writes the dataset medoid as the single frozen start point, same as Vamana's default. Already byte-compatible; no work required.
- **Multi-vector index support.** Out of scope for Stage 1; revisit only if a production workload requires it.

## References

1. [PiPNN: Pick-in-Partitions Nearest Neighbors (arXiv:2602.21247)](https://arxiv.org/abs/2602.21247)
2. [Vamana / DiskANN (NeurIPS 2019)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
3. Existing disk index layout: `diskann-disk/src/storage/`
4. Existing Vamana builder: `diskann-disk/src/build/builder/build.rs`
