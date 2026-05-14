# Integrate PiPNN as an Alternative Graph-Index Build Algorithm

| | |
|---|---|
| **Authors** | Weiyao Luo |
| **Contributors** | DiskANN team |
| **Created** | 2026-05-11 |
| **Updated** | 2026-05-14 |

## Summary

Add **PiPNN** (Pick-in-Partitions Nearest Neighbors, [arXiv:2602.21247](https://arxiv.org/abs/2602.21247)) as a second graph-construction algorithm for **DiskANN's disk-index full-build path** — both initial builds from a fresh dataset and full rebuilds that replace an existing index. PiPNN produces a graph byte-compatible with Vamana's disk format and search API, at **up to 6.3× lower build time** on the workloads we have measured. Vamana remains the default for disk builds and the only algorithm supported for in-memory incremental inserts. In-memory PiPNN build is explicitly out of scope: DiskANN's in-mem path exists to support streaming construction, which PiPNN's batch algorithm cannot do efficiently.

## Motivation

### Background

DiskANN currently builds the disk index with a single algorithm — **Vamana** (`diskann-disk/src/build/builder/`). Vamana incrementally inserts each point into a graph, running a greedy search + `RobustPrune` for each insertion, producing the on-disk format documented in `diskann-disk/src/storage/`.

Clients today update indexes in three main ways:

1. **Incremental** — continuously insert and delete vectors in an existing in-memory graph (Vamana's per-point greedy-search + `RobustPrune` path). The disk index itself is not mutated in place.
2. **Full rebuild** — rebuild the entire graph from scratch on a static snapshot, producing an immutable disk index.
3. **Partitioned full rebuild** — split points into N clusters, build N separate graphs in parallel, then stitch them together with a lightweight merge step to bound peak build-time memory (Vamana's `build_merged_vamana_index` path).

PiPNN, as proposed here, is a faster substitute for paths (2) and (3). Path (1) remains Vamana's responsibility for the foreseeable future (see "PiPNN is algorithmically batch-only" below).

**PiPNN** (Pick-in-Partitions Nearest Neighbors, [arXiv:2602.21247](https://arxiv.org/abs/2602.21247)) is a partition-based **batch** graph builder, in contrast to Vamana's **incremental** insert + prune. The construction has four phases:

1. **Partition** — Randomized Ball Carving (RBC) recursively splits the dataset into small *overlapping* leaf clusters. Each point lands in `fanout` of its nearest cluster leaders at every recursion level, so every point appears in multiple leaves. Recursion stops when a cluster fits a configured leaf-size cap (`c_max`, typically 256–1024 points).
2. **Local k-NN per leaf** — For each leaf, compute the full pairwise distance matrix as a single batched GEMM (an N×N intra-leaf computation, where N ≈ `c_max`), then extract each point's `leaf_k` nearest neighbors inside the leaf. This is structurally different from a flat scan (1×N query against the whole dataset, e.g. work item [#1036](https://github.com/microsoft/DiskANN/issues/1036)) — every column of the GEMM contributes to every row's top-k, so the cost is amortized across `c_max²` distance evaluations. GEMM batching is the source of most of PiPNN's wall-clock advantage over per-point greedy search.
3. **HashPrune merge** — Edges from all leaves are merged into a per-point reservoir of bounded size (`l_max`, ~64–128). The pruner is keyed by an LSH **angular bucket** of each candidate neighbor: at most one candidate per bucket is retained, and on collision the closer candidate wins. This produces a diverse short-list per point using O(`l_max`) memory per node and O(1) amortized insert work. The merge stage is naturally streamable — edges can be fed in chunks (either generated all at once and replayed from disk, or generated leaf-batch-by-leaf-batch interleaved with HashPrune inserts) to bound peak RAM; see M5 below.
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

Initial builds at 10M-scale and above, and the frequent full rebuilds that follow them (driven by data churn or parameter sweeps), are the bottleneck. PiPNN's offline benchmarks at matching recall budgets complete the same builds **up to 6.3× faster** while writing the same disk format (full numbers in the Benchmark Results section). This RFC proposes landing PiPNN so teams can opt into faster builds and so we can collect production-relevant signal on whether PiPNN can eventually replace Vamana's full-build path (initial builds and rebuilds alike).

#### Concrete trade-off hypothesis

To make the comparison precise rather than headline-only, we frame Stage-1 validation around a fixed-resource hypothesis:

> Given a worker with fixed CPU cores, RAM budget, and SSD throughput, PiPNN delivers higher index-build throughput (vectors per minute per worker) than Vamana at matching recall, *provided* its working set fits within the RAM budget. When the RAM budget is below PiPNN's one-shot working set, the three-tier dispatch (disk-edges, then merged-shards) keeps PiPNN within or below Vamana's RAM footprint at a documented build-time cost.

Concretely, on BigANN 10M with the same 16-thread / NVMe worker:

| RAM budget | PiPNN strategy | PiPNN build | Vamana build |
|---:|---|---:|---:|
| ≥ ~12 GB | one-shot | 80–133s | 358s |
| 6–12 GB | disk-edges | ~126s | 358s |
| 3–6 GB | merged-shards | ~332s | 358s (partitioned: similar) |
| < 3 GB | merged-shards w/ smaller shards | further degrades | further degrades |

Two important things this table is **not** claiming:

- **PiPNN does not auto-scale build time downward when given more RAM than its working set needs.** PiPNN's wall-clock is dominated by HashPrune inserts + leaf-build GEMM. Once the dataset, HashPrune reservoir, and per-thread buffers fit comfortably in RAM, additional RAM headroom does not buy faster builds. (More memory *channels* / higher bandwidth do help, but that is a hardware property, not a budget knob.)
- **Vamana also does not have a "use more RAM to build faster" mode** — its peak RSS is largely set by the dataset + working graph, and giving it more RAM headroom past that does not accelerate the per-insert greedy search.

So the honest framing is: PiPNN trades a higher minimum RAM budget for a substantially faster build at that budget. Neither algorithm currently converts surplus RAM into faster builds; both convert surplus RAM into "no pressure to use the chunked/shard fallbacks."

The numbers above are from initial benchmarks on a single workload and configuration. A dedicated experiment to validate this hypothesis across RAM budgets and worker shapes is part of Stage 1 — see **M6 — Fixed-resource trade-off validation** in Future Work below.

#### Hybrid update model (Stage 2 direction)

Vamana and PiPNN write the same on-disk graph format, so a graph built by either algorithm can be loaded by the same search code and, once loaded into memory, can be incrementally edited by Vamana. We exploit this for the production update story:

- **Bulk / full rebuild → PiPNN.** When a full rebuild is needed, PiPNN is used because it is several times faster than Vamana at this job.
- **Incremental insert → in-memory Vamana.** Between full rebuilds, individual inserts use Vamana's existing greedy-search + RobustPrune insert path **on the in-memory graph** (`diskann::graph::index::DiskANNIndex`). The on-disk index file is not mutated in place — the standing convention is that a refreshed disk index is produced by a full rebuild from the current dataset snapshot. PiPNN's batch design has no natural single-point-insert API and we do not plan to build one.
- **Triggers for a full PiPNN rebuild.** A rebuild is scheduled in response to operationally meaningful events, not just gradual recall drift. The expected triggers include: (a) embedding-model rotation (vectors are no longer comparable to existing ones), (b) schema/parameter retuning (`R`, `L`, `pq_chunks`, distance metric, quantization), (c) large batch inserts that exceed what the in-memory incremental path is sized for, and (d) periodic safety rebuilds on a cadence that depends on observed graph health. DiskANN's existing claim that incremental updates keep recall healthy still holds; PiPNN does not change that, it just makes the eventual rebuild cheaper.

Because both algorithms produce the same disk format, switching between "fresh PiPNN build" and "Vamana-edited in-mem graph reloaded from a fresh disk build" is transparent to search-side consumers. This answers "should PiPNN implement incremental inserts?" — no, we keep Vamana's in-memory insert path for that, and use the disk index format as the integration point between rebuilds.

#### Two-stage rollout

- **Stage 1 (this RFC):** Land PiPNN as an alternative builder for the **disk-index full-build path** — covering both initial builds (no prior index) and full rebuilds (replacing an existing index) — behind a build-algorithm selector. Vamana stays default; PiPNN is opt-in. Stage 1 has explicit milestones (in Future Work) that gate readiness for Stage 2.
- **Stage 2 (separate proposal, conditional on Stage 1 milestones):** Retire the Vamana **disk-index full-build** path (initial builds and rebuilds). Vamana remains the implementation for incremental inserts on the in-memory graph via the hybrid model above.

In-memory PiPNN build/search is **not part of any stage**. DiskANN's in-memory `DiskANNIndex` path exists primarily to support streaming (per-point) index construction, which is exactly the use case PiPNN's batch algorithm does not address (see "PiPNN is algorithmically batch-only"). Replacing or extending the in-memory builder with PiPNN would offer no incremental capability and duplicate the disk path's value. We therefore list it under unstaged future work rather than as a Stage 1 / Stage 2 milestone.

### Goals (Stage 1)

Stage 1 is scoped to the **disk-index full-build path** — both initial builds (no prior index) and full rebuilds (replacing an existing index). In-memory index construction is explicitly out of scope.

1. **Algorithm-level pluggability for the disk builder**: introduce a build-algorithm selector to the disk-index build pipeline that routes between Vamana (existing) and PiPNN (new). Existing build sites continue to default to Vamana with no behavior change.
2. **Disk format compatibility**: the PiPNN-built index is byte-compatible with Vamana-built indexes on disk — search, PQ, and storage layouts are unchanged. This is the foundation for the hybrid update model.
3. **Public API compatibility**: the disk-index public API surface (`DiskIndexBuilder::new`, `IndexConfiguration`, `DiskIndexWriter`, JSON config schema) remains backward-compatible. PiPNN configuration is added under a new tagged enum variant.
4. **Feature-parity milestones (disk path only)**: deliver the Vamana disk-build capabilities PiPNN needs to take over both initial builds and full rebuilds in production (see Future Work below).
5. **Documented memory mitigation**: provide a configuration knob (three-tier build) that brings PiPNN's peak RSS to or below Vamana's at the cost of build time.

## Proposal

### Workspace structure

Add a new crate, `diskann-pipnn`, that depends on the existing `diskann`, `diskann-linalg`, `diskann-vector`, `diskann-quantization`, and `diskann-utils` crates. **`diskann-pipnn` does not depend on `diskann-disk`.** The PiPNN builder produces a plain `Vec<Vec<u32>>` adjacency list (defined in terms of core types from `diskann`), and `diskann-disk` consumes that output behind its own `pipnn` Cargo feature. This is intentional: a `diskann-pipnn → diskann-disk → [feature] diskann-pipnn` edge would form a dependency cycle. Keeping the data-flow direction one-way (PiPNN produces, disk consumes) means PiPNN never imports any disk-layout symbols and the feature gate sits cleanly on the consumer side.

```text
diskann/                    # core types, traits, search    ←┐ used by both
diskann-linalg/             # GEMM/SVD                       ├─ shared deps
diskann-quantization/       # PQ/SQ training                 ├─ (no edges
diskann-vector/             # vector representations         ├─  to either
diskann-utils/              # threading, file I/O            ←┘  builder)

diskann-pipnn/              # new: PiPNN builder
   ↑ produces Vec<Vec<u32>>
   │
diskann-disk/               # disk index layout, builder, search
  └── feature "pipnn"       # opt-in: takes Vec<Vec<u32>> from diskann-pipnn
                            # and hands it to DiskIndexWriter
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
        num_threads: usize,          // 0 = all logical CPUs (matches Vamana)
    },
}
```

`Vamana` is the `Default` so every existing call site that constructs `DiskIndexBuildParameters` without specifying an algorithm keeps the existing behavior.

`DiskIndexBuildParameters` gains a `build_algorithm: BuildAlgorithm` field and a constructor pair: `new` (defaults to Vamana, no PiPNN dep) and `new_with_algorithm` (explicit). The JSON schema for benchmark configs gains an optional `build_algorithm` block that, when present, deserializes via `#[serde(tag = "algorithm")]` into one of the variants above.

**`num_threads`.** Like Vamana, PiPNN accepts `num_threads` as a build-time parameter (default `0` = all logical CPUs). Thread count has a small, bounded effect on peak RSS: each worker holds thread-local stripe buffers in the partition phase (~`stripe_kb`, typically 16 MB) and thread-local leaf-build scratch (~`c_max² × 4 B`, ≈ 256 KB at `c_max=256`). Total per-thread overhead is ~16–20 MB; at 48 threads this is ~960 MB of incremental resident set on top of the dataset and HashPrune reservoir, which dominate the peak. We do not consider `num_threads` a memory-budget knob — to bound RAM, use `build_ram_limit_gb` (see Memory mitigation).

**Deserialization behavior when the `pipnn` feature is disabled — scope:** this affects only **JSON configs**, not the index files themselves. Because PiPNN and Vamana write byte-identical disk formats, an index *file* built by either algorithm is loaded by the same search code and does not require the `pipnn` feature at load time. The restriction below applies to the *build-time configuration* that selects which algorithm to invoke. Because `BuildAlgorithm::PiPNN` is gated by `#[cfg(feature = "pipnn")]`, a binary built without the feature does not see that variant. A JSON config containing `"algorithm": "PiPNN"` fed to such a binary fails at parse time with a serde error along the lines of `unknown variant 'PiPNN', expected 'Vamana'`. This is a clear, fail-fast diagnostic — not a backward-compatibility regression. Configs that omit `build_algorithm` (or set `"algorithm": "Vamana"`) parse identically across feature combinations. Documentation alongside the config schema will call this out so users know that PiPNN configs require a PiPNN-enabled build.

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

This is a property of the algorithm, not of our implementation. The PiPNN paper ([arXiv:2602.21247](https://arxiv.org/abs/2602.21247)) is explicit that the design departs from incremental methods by "eliminating search from the graph-building process altogether": instead of running a greedy search for each new point's neighbors, PiPNN partitions the dataset, then computes neighbors for all points within each leaf as a single batched operation. The paper describes no per-point insertion algorithm and reports no streaming results. The framing throughout is "fast one-shot construction on a static dataset."

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

Note on disk-edges build time (~126s vs one-shot's ~133s): the disk-edges path is not slower despite the extra I/O. The smaller resident working set means HashPrune inserts touch fewer cache lines per operation, and the spill to disk is sequential append-only and overlaps with leaf-build compute. Net: roughly the same wall-clock as one-shot in this benchmark, with significantly lower peak RSS.

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

The Stage 1 milestones below are gating items for Stage 2 (retiring Vamana's disk-index full-build path — initial builds and rebuilds). Each must be addressed before that proposal is credible. M0 is the foundation shipped by this RFC; M3–M9 are deferred to follow-on work and ordered by dependency, not strict calendar sequence — some can run in parallel. M1 (in-memory build/search) and M2 (checkpoint/resume) are intentionally absent — see "Out of scope: not part of any stage" and "Deferred to Stage 2" below.

### M0 — Skeleton integration

The foundation that ships first.

- **Scope:** introduce the `diskann-pipnn` crate, the `BuildAlgorithm` enum, and the dispatch in `DiskIndexBuilder` behind a `pipnn` Cargo feature.
- **Config surface:** JSON config gains an optional `build_algorithm` block; default behavior unchanged.
- **Compatibility:** PiPNN-built indexes are read by the existing search pipeline unchanged (the on-disk format is identical) and produce recall numbers within the tolerances the existing disk-index test suite enforces.
- **CI:** benchmark binary runs with `--features pipnn` on a small smoke test (SIFT-1M).

M1, M3–M5 close the feature-parity gaps in Stage 1; M6–M9 are validation and operational readiness. Checkpoint/resume (previously M2) is deferred to Stage 2 — see "Deferred to Stage 2" below for the rationale.

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

- **Disk-edges:** today's prototype generates all leaf edges first, spills them to disk, then streams chunks back into HashPrune. An alternative we plan to evaluate is to interleave the two — write partition metadata to disk and run leaf-build + HashPrune in chunks (build edges for the first N leaves' points, flush their adjacency lists, then move on). Both variants bound the resident HashPrune reservoir; the second avoids the full edge-set materialization at the cost of a second pass over the partition.
- **Merged-shards:** per-shard graphs built independently then merged, mirroring Vamana's `build_merged_vamana_index` pipeline at `diskann-disk/src/build/builder/build.rs:327`. The existing shard merger is reused.
- **Dispatch:** inside `build_inmem_pipnn_index()` — no new public parameter.
- **Validation:** at `build_ram_limit_gb=4`, the PiPNN-merged path on BigANN 10M produces peak RSS ≤ 4 GB and recall within 1% of one-shot PiPNN.

### M6 — Fixed-resource trade-off validation

This milestone validates the **concrete trade-off hypothesis** stated in the Problem Statement: under a fixed worker shape (CPU cores, RAM budget, SSD throughput), PiPNN delivers higher build throughput than Vamana at matching recall when its working set fits, and remains competitive (via the three-tier dispatch in M5) when it does not. The output of this milestone is the evidence behind the per-budget recommendation in the Stage-1 deployment guide.

- **Fixed worker shape per run.** Lock CPU cores (e.g. 16), SSD model/throughput, and a RAM ceiling enforced via cgroups (`memory.max`) so the build *cannot* exceed it. RAM-budget sweep on BigANN 10M: `{3, 6, 8, 12, 16, 24, 32}` GB at minimum. Include at least one row each for Enron 10M (higher dim, larger reservoir) and a 100M-scale dataset (one budget per algorithm sufficient to fit).
- **Algorithm × strategy cells.** For each RAM budget, run: Vamana one-shot, Vamana partitioned, PiPNN one-shot (if fits), PiPNN disk-edges, PiPNN merged-shards. Skip cells whose minimum working set exceeds the budget — those count as "OOM, not supported at this budget" and are part of the result, not a gap.
- **Metrics captured per cell.** Wall-clock build time, peak RSS (via heaptrack or `/usr/bin/time -v`), CPU utilization (`pidstat`), SSD bytes read/written, recall@K at L=50/100/L_target, and queries-per-second at matching recall. Throughput reported as **vectors per minute per worker** so different worker shapes compare directly.
- **Hypotheses to confirm or falsify.**
  1. PiPNN's wall-clock advantage over Vamana persists across all RAM budgets where its working set fits (one-shot or disk-edges variant).
  2. PiPNN's merged-shards path matches or beats Vamana's partitioned-rebuild at the same RAM ceiling on build time *and* recall.
  3. Neither algorithm reduces build time when given RAM headroom past its working-set requirement (validates the "surplus RAM doesn't buy speed" claim).
  4. PiPNN's per-thread overhead is bounded as stated (~16–20 MB/thread) and `num_threads` is not a hidden RAM knob.
- **Out-of-budget behavior.** Each (algorithm × budget) cell that cannot complete is recorded as such — explicit "PiPNN one-shot not supported at 6 GB on BigANN 10M" is a valid result, not a failed experiment.
- **Pass criterion for Stage 2 readiness.** A documented matrix where each budget has a clearly-better algorithm (or "tie") at matching recall, with no surprise cells that contradict the Problem Statement's hypothesis. Surprises must be either reproduced and explained, or treated as Stage-1 blockers.

### M7 — Production validation: recall × QPS × dimensionality matrix

End-to-end validation on the full production workload mix (independent of the resource matrix in M6).

- **Datasets:** at minimum three families (BigANN, Enron, plus one production-representative).
- **Scales:** 10M and 100M; one billion-scale sample if hardware permits.
- **Metrics:** `squared_l2` and `cosine_normalized`.
- **Pass criterion:** for each (dataset, scale, metric) cell, PiPNN recall@K is within Vamana's recall ±1% at matching QPS, *or* higher QPS at matching recall.
- **Out-of-band cells** are documented as "PiPNN not yet recommended for X" rather than blocking Stage 2 entirely.

### M8 — Production validation: hybrid update model

Validate the Stage-2 hybrid loop end-to-end.

- **Sequence:** PiPNN build → N incremental Vamana inserts representing production churn → measure recall decay vs. graph age → trigger PiPNN rebuild from snapshot → confirm post-rebuild recall restored.
- **Output:** a recommended "quality decay threshold" for production rebuild triggers, derived from the measured decay curve.
- **Disk-format compatibility test:** confirm Vamana's incremental-insert path reads PiPNN-produced graphs correctly. This is the load-bearing compatibility check for the hybrid model.

### M9 — Operational readiness

- **Telemetry:** emit per-phase timing and peak RSS via the existing OpenTelemetry tracer, comparable to Vamana's spans.
- **Documentation:** replace experimental notes in `CLAUDE.md` with a permanent doc covering recommended parameters per workload class (dim × scale × metric).
- **Runbook:** failure modes (OOM under one-shot, partition timeout, `l_max` saturation), diagnosis, recovery.
- **Defaults:** parameter recommendations baked into the JSON config builder so users don't hand-tune for common cases.

### Deferred to Stage 2

- **Checkpoint / resume (was M2).** Vamana's checkpoint/resume is a *streaming* mechanism — it relies on the per-point incremental insert order to define natural checkpoint boundaries. PiPNN's batch design has no equivalent monotonic insertion sequence: partition output, per-leaf GEMM, and HashPrune merge are all coarse-grained whole-phase artifacts rather than fine-grained incremental progress. A useful PiPNN checkpoint scheme would therefore *not* mirror Vamana's; it would need new design choices about which phase boundaries to materialize, at what granularity, and whether the cost-benefit justifies the extra disk I/O. Empirically, PiPNN's full BigANN-10M build runs in ~80 s, so the operational value of resuming a partially completed build is materially lower than for Vamana's multi-hour rebuilds. We defer checkpoint design until Stage 2, when the production rebuild cadence and observed failure modes will tell us whether it is needed and what shape it should take.

  *Note on determinism for any future checkpoint validation:* PiPNN is a parallel algorithm (rayon-parallel partition, leaf-build GEMM, and HashPrune merge), so byte-identical output across runs — and therefore across "resumed vs. never-interrupted" runs — is **not** a free property. It would require extra determinism work (fixed thread schedule, deterministic reduction order in the HashPrune reservoir, seeded LSH hyperplanes). The right validation criterion for a resumed build is **recall parity with a non-resumed build**, not byte-identical adjacency lists.

### Out of scope: not part of any stage

These are explicitly *not* on a Stage 1 or Stage 2 roadmap. They may be revisited if a future workload demands them, but they are not gating items for either stage.

- **In-memory PiPNN build / in-memory index population (was M1).** DiskANN's in-memory `DiskANNIndex` exists primarily to support streaming per-point construction and online inserts — which is exactly what PiPNN's batch design cannot do efficiently (see "PiPNN is algorithmically batch-only"). Building a `DiskANNIndex` from PiPNN-produced adjacency lists is mechanically possible (the data structures are compatible) but offers no incremental capability, duplicates the disk path's value, and would force `diskann-pipnn` to take a runtime dependency on the in-mem graph crate. We defer indefinitely; if a non-streaming in-mem consumer ever needs PiPNN's build speed, the simpler answer is "build to disk, then load."
- **Build-time PQ distance kernel.** Not used by Vamana in production paths today; deferred indefinitely.
- **PiPNN incremental insert API.** The hybrid model (PiPNN rebuild + Vamana inserts) removes the need.
- **PiPNN incremental delete API.** Same reason.
- **Frozen-point semantics differences.** PiPNN writes the dataset medoid as the single frozen start point, same as Vamana's default. Already byte-compatible; no work required.
- **Multi-vector index support.** Revisit only if a production workload requires it.

## References

1. [PiPNN: Pick-in-Partitions Nearest Neighbors (arXiv:2602.21247)](https://arxiv.org/abs/2602.21247)
2. [Vamana / DiskANN (NeurIPS 2019)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
3. Existing disk index layout: `diskann-disk/src/storage/`
4. Existing Vamana builder: `diskann-disk/src/build/builder/build.rs`
