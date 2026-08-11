<!--
 Copyright (c) Microsoft Corporation.
 Licensed under the MIT license.
-->

# Graph-IVF (online) — storage read-path cost & latency model

**Scope**: query path only. Azure East US, USD, pay-as-you-go, LRS.
**Purpose**: size the *"many indexes in storage, minimal RAM"* architecture — what
latency, throughput, and $ to expect from local NVMe vs. Azure Managed Disks vs. Azure
Blob, as a function of the clustering granularity knob (`split_threshold`).

Companion docs: [`ONLINE.md`](ONLINE.md) (algorithm),
[`INVESTIGATION_RESULTS.md`](INVESTIGATION_RESULTS.md) (measurements),
[`PERFORMANCE.md`](PERFORMANCE.md) (read-path design),
[`../VM_DISK_INVESTIGATION.md`](../VM_DISK_INVESTIGATION.md) (this VM's disk).

---

## TL;DR

1. **A query is `nlist` independent reads of `≈ 0.5–0.75 · split_threshold` vectors each.**
   `split_threshold` (`T`) sets the **size** of each read; `nlist` sets the **count**.
   Every storage medium prices exactly one of those two axes.
2. **The two axes trade against each other at iso-recall, and the trade is sub-linear.**
   Measured at 2.82M × 1536: going **15.8× coarser** (`T` 50 → 790) bought **6.0× fewer
   IOs** but cost **3.5× more bytes**. Empirically `IOs ∝ T^-0.65`, `bytes ∝ T^+0.46`.
3. **Bandwidth-billed media (NVMe, Premium SSD v2) want small `T`.**
   **Operation-billed media (Blob) want large `T`.** There is no `T` that is good for both.
4. **`T` is also the RAM knob**, and it opposes (3) for disk. The centroid graph is
   in-RAM and always `f32`: 10M × 768 at `T`=64 needs **929 MiB**; at `T`=4096, **10 MiB**.
   Below `T ≈ 512` the centroid graph alone defeats the low-RAM premise.
5. **Blob has a hard wall unrelated to cost**: 3,000 requests/sec **per blob**, and one
   index = one `.graphivf_lists` blob → **12 QPS/index** at `nlist`=256. Must be sharded.
6. ⚠️ **Existing benchmark numbers in this repo do not measure the P40 they appear to.**
   They sustain 0.9–2.1 GB/s — above **both** the P40's 250 MB/s media cap **and** the VM's
   1,000 MB/s cached-path ceiling. They are memory-served. Good proxy for local NVMe /
   PSSDv2; **~5–8× optimistic** for any real managed disk.
7. **A query issues its `R` reads in `⌈R/128⌉` serial waves, each with a hard barrier**
   (`MAX_IO_CONCURRENCY = 128`), so it samples the **p99.2 tail once per wave** — median
   TTFB is nearly irrelevant. This disqualifies **standard hot blob** outright
   (p99 = 287 ms → 283–1,127 ms/query).
8. **Net recommendation**: local NVMe (Lsv4/Lasv3) or Premium SSD v2, `T` in the
   **512–2048** band, `minmax8` lists. Blob only behind a cache, sharded ≥8 ways.

---

## 1. The read model

From [`src/storage.rs`](src/storage.rs) and [`src/index.rs`](src/index.rs)
(`Searcher::search_profiled`), one query does:

| Stage | Cost | Touches storage? |
|---|---|---|
| preprocess | ~3 µs | no |
| centroid search (Vamana over `k` centroids) | 2.2–4.5 ms, **RAM-resident, f32** | no |
| `plan_io` | one 512-aligned window per non-empty probed list | no |
| **`disk_read`** | **`⌈nlist/128⌉` serial waves of ≤128 concurrent unbuffered reads** | **yes** |
| score + topk | ~5–10% of total | no |

Reads are unbuffered (`FILE_FLAG_NO_BUFFERING` / `O_DIRECT`) and issued as a single
batch via IOCP / io_uring. There is no page cache and no LRU in the read path.

### Formulas

Let `N` = points, `d` = dim, `T` = `split_threshold`, `e` = element size,
`h` = per-vector quant header, `γ` = probe-selection bias.

```
record_bytes  = 4 (u32 id) + d·e + h
mean list  c̄  = ρ(T) · T          ρ ≈ 0.126 + 0.093·ln(T), capped at 0.75   ← measured
clusters   k  = N / c̄
req_bytes     = γ · (c̄ · record_bytes + 512)
ios / query   = nlist − (empty probed lists)      ≈ nlist − 1 in practice
bytes / query = ios · req_bytes
centroid RAM  = k · (4d + 4·graph_degree)         ← f32 centroids + adjacency
```

**The `+ 512`.** `cluster_window` rounds outward to a 512 boundary at both ends. For a
list of `L` bytes at a uniformly random 512-phase start, `E[aligned_len] = L + 512` exactly.

**`ρ(T)` — why not `T/2`?** The README's equilibrium `k ≈ 2N/T` implies `c̄ = T/2`.
That only holds at *fine* granularity. Splitting a list at `>T` into halves yields sizes
uniform on `[T/2, T]`, i.e. `c̄ → 0.75·T`; reassignment and empty clusters pull the ratio
down when `T` is small. Measured `ρ` rises **0.49 → 0.61 → 0.75** as `T` goes 50 → 120 → 790.
Using `c̄ = T/2` under-predicts bytes by **21% at `T`=120 and 49% at `T`=790**.

**`γ` — probe-selection bias.** Queries preferentially probe denser (larger) lists, so the
realized mean read exceeds the corpus mean. Measured `γ` = **1.11–1.19** (`T`=50),
**1.04–1.09** (`T`=120), **1.00–1.01** (`T`=790). Use `γ`=1.10 for `T`<100, 1.05 for
`T`<400, 1.00 above.

### Validation — real online build, 2.82M × 1536, `minmax8`, `reassign_neighbors`=32

Source: `_results/logs/lotteall/graphivf_online_t{50,120,790}_b4096.log`.
`record_bytes` = 4 + 1536 + 20 = 1,560 B.

| `split_thr` | clusters `k` | `c̄` = N/k | `ρ` = c̄/T | model req KiB | **measured req KiB** | `γ` |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 115,251 | 24.5 | 0.49 | 37.8 | **41.9 – 44.8** | 1.11–1.19× |
| 120 | 38,706 | 72.8 | 0.61 | 111.4 | **116.1 – 121.8** | 1.04–1.09× |
| 790 | 4,779 | 589.9 | 0.75 | 899.2 | **902.3 – 909.5** | 1.00–1.01× |

The model is exact to **≤1%** at coarse granularity and within the `γ` band at fine
granularity. `Reads` measured = `nlist − 1` at every point (one empty probed cluster).
Note `max` list size can exceed `T` (272 observed at `T`=120) because splits are deferred
to batch boundaries.

---

## 2. ⚠️ The existing measurements are not measuring the disk

Every `DiskRead` figure in `INVESTIGATION_RESULTS.md` and the lotte-all logs was taken on
`D16as_v5` + **PSSDv1 P40**, whose ceiling is **250 MB/s**. The realized bandwidth:

| `split_thr` | `nlist` | MB read/q | `DiskRead` ms | **effective MB/s** | × P40 cap |
|---:|---:|---:|---:|---:|---:|
| 50 | 200 | 9.0 | 8.1 | **1,104** | 4.4× |
| 50 | 450 | 19.7 | 16.9 | **1,160** | 4.6× |
| 120 | 100 | 12.2 | 7.1 | **1,709** | 6.8× |
| 120 | 450 | 53.4 | 30.8 | **1,731** | 6.9× |
| 790 | 40 | 37.1 | 20.8 | **1,787** | 7.1× |
| 790 | 60 | 55.6 | 27.0 | **2,059** | 8.2× |

The workload (4.4 GB corpus) fits entirely in the VM's 64 GiB RAM, and these rates exceed
**both** the P40's 250 MB/s media cap **and** the VM's own 1,000 MB/s *cached*-path ceiling
— so the reads are being served from memory in the virtualization stack, not from P40
media. (`FILE_FLAG_NO_BUFFERING` defeats the *guest* page cache but **not** the host cache;
see `VM_DISK_INVESTIGATION.md` §5.) Queries also run **sequentially** here — measured QPS
= 1/mean-latency exactly — so these are true single-query rates, not aggregate.

Consequences:

- The measured latencies are a **reasonable proxy for local NVMe or Premium SSD v2**
  (both ~2,000 MB/s), which is convenient.
- They are **5–8× optimistic for any real managed disk**, including the P40 attached here.
- Any conclusion about *disk* p99 / queueing behaviour from these runs is invalid — and
  per §4.4 the tail is what actually governs this workload.
- Re-baseline on Lsv4/Lasv3 (local NVMe) or PSSDv2, or on a data disk created with
  `caching = None`.

---

## 3. Operating points — the iso-recall trade-off

**This is the table that answers "what latency at which cluster size".**
Measured, lotte-all-forum, N = 2,819,103, d = 1536, `minmax8`, recall@50, `batch_size` 4096.

### Recall ≈ 93.5%

| `split_thr` | clusters `k` | `c̄` | `nlist` | recall | **IOs/q** | **MB/q** | centroid RAM | measured mean ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **50** | 115,251 | 24.5 | 360 | 93.63% | **359** | **15.8** | 689 MiB | 19.5 |
| **120** | 38,706 | 72.8 | 220 | 93.74% | **219** | **26.4** | 232 MiB | 21.1 |
| **790** | 4,779 | 589.9 | 60 | 93.00% | **60** | **55.6** | 29 MiB | 33.9 |

### Recall ≈ 88% / ≈ 96%

| Target | `split_thr` | `nlist` | recall | IOs/q | MB/q | measured mean ms |
|---|---:|---:|---:|---:|---:|---:|
| 88% | 50 | 120 | 88.28% | 119 | 5.5 | 11.0 |
| 88% | 120 | 60 | 86.54% | 59 | 7.4 | 10.3 |
| 88% | 790 | 25 | 87.24% | 25 | 23.3 | 25.0 |
| 96% | 50 | 900 | 96.28% | 899 | 38.6 | 40.1 |
| 96% | 120 | 450 | 96.12% | 449 | 53.4 | 38.6 |
| 96% | 790 | 130 | 96.20% | 130 | 120.1 | 73.6 |

### What the trade actually costs

| Recall target | IOs saved (T 50→790) | Bytes paid (T 50→790) |
|---|---:|---:|
| 88% | **4.8× fewer** | **4.3× more** |
| 93.5% | **6.0× fewer** | **3.5× more** |
| 96% | **6.9× fewer** | **3.1× more** |

Empirically at iso-recall: **`IOs ∝ T^-0.65`**, **`bytes ∝ T^+0.46`**, **`RAM ∝ T^-1.15`**.

> **Read this as**: coarsening is a *win on blob* (fewer billed operations, fewer round
> trips) and a *loss on disk* (more bytes, more latency), and it is the only way to get
> the centroid graph small enough for a low-RAM host. There is no setting that is good
> on all three axes — you must pick the medium first, then `T`.

### Projected index shape at a different scale (N = 10M, d = 768, `graph_degree` = 32)

Using the corrected `ρ(T)` and `γ(T)`:

| `split_thr` | `ρ` | `c̄` | clusters `k` | **centroid RAM** | req KiB (f16) | req KiB (minmax8) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.51 | 33 | 304,467 | **929 MiB** | 55.0 | 28.5 |
| 128 | 0.58 | 74 | 135,229 | **413 MiB** | 117.6 | 60.6 |
| 256 | 0.64 | 164 | 60,821 | **186 MiB** | 260.8 | 134.1 |
| 512 | 0.71 | 362 | 27,634 | **84 MiB** | 546.1 | 280.4 |
| **1024** | 0.75 | 768 | 13,021 | **40 MiB** | 1,158.5 | 594.5 |
| 2048 | 0.75 | 1,536 | 6,510 | **20 MiB** | 2,316.5 | 1,188.5 |
| 4096 | 0.75 | 3,072 | 3,255 | **10 MiB** | 4,632.5 | 2,376.5 |

`minmax8` halves every byte figure vs. `float16` and quarters vs. `float32` for ~0.5 recall
points (§10 of `INVESTIGATION_RESULTS.md`) — the single cheapest lever in the model.

---

## 4. Storage medium characteristics

### 4.1 Throughput ceilings

| Medium | Bandwidth ceiling | IOPS / req-rate ceiling | Billing axis |
|---|---|---|---|
| **Local NVMe** L8as_v3 | 2,000 MB/s | 400,000 IOPS | bundled in VM price |
| **Local NVMe** L32s_v4 | **12,000 MB/s** | 2,200,000 IOPS | bundled in VM price |
| **Premium SSD v2** | 2,000 MB/s /disk | 80,000 IOPS /disk | capacity **+ provisioned IOPS + provisioned MB/s** |
| **Premium SSD v1** P40 *(this VM)* | **250 MB/s** | 7,500 IOPS | capacity tier only |
| **Ultra Disk** | 4,000 MB/s /disk | 400,000 IOPS /disk | capacity + IOPS + MB/s |
| **Premium block blob** | account 200 Gbps egress | **3,000 req/s per blob** | **per read op** + capacity |
| **Standard hot blob** | account 40,000 req/s (E US) | **3,000 req/s per blob** | **per read op** + capacity |

### 4.2 Time-to-first-byte, minimum IO size, and latency profile

This is what actually governs graph-IVF query latency, because a query blocks on **all**
`R` reads at once (see §4.4).

| Medium | **TTFB (p50)** | **p99** | **p99.9** | Distribution shape | **Min / granular IO** | Concurrency model |
|---|---:|---:|---:|---|---|---|
| **Local NVMe** (Lsv4/Lasv3) | **~20–80 µs** | ~0.2 ms | ~0.8 ms | Tight, near-Gaussian. Tail from NAND GC only | 512 B logical (512e); 4 KiB on 4Kn. NAND page ~16 KiB internally | IOCP/io_uring, **QD 128** (§4.4); device scales far beyond |
| **PSSD v1, host-cached (hit)** | **0.21 ms** *(measured)* | 1.1 ms | **16 ms** | **Strongly bimodal** — RAM hits vs. media misses. Fat tail | 512e (512 B) | QD 128; cached path capped at VM cached limit |
| **PSSD v1, uncached** | ~0.6–1.5 ms | ~4 ms | ~20 ms | Network-attached; long tail, **multi-second under throttle** | 512e (512 B) | QD 128 |
| **Premium SSD v2** | **~0.4–0.8 ms** | ~2 ms | ~5 ms | "Sub-ms 99.9% of the time" *if* provisioning is not exceeded; cliff past it | **4096 B physical** (default), 512e optional | QD 128 |
| **Ultra Disk** | ~0.3–0.5 ms | ~1.2 ms | ~3 ms | Tightest of the network-attached tiers | 4096 B (512e optional) | QD 128 |
| **Premium block blob** | **~3–5 ms** (MS demo: 5.3 ms mean) | **6.9 ms** | ~20 ms | Tight for a network service — SSD-backed, low variance | No minimum (arbitrary Range GET); **billed per 4 MiB** | 1 HTTP GET per read; bounded by client connection pool |
| **Standard hot blob** | **~25–60 ms** (MS demo: 61.4 ms mean) | **287 ms** | ~800 ms | **Very fat tail** — HDD-backed, heavily multi-tenant | No minimum; **billed per 4 MiB** | 1 HTTP GET per read |

Sources: Microsoft's premium-block-blob performance demo (5.3 ms mean / 6.9 ms p99 small
random read premium, vs 61.4 ms mean / 287.3 ms p99 standard); `disks-types` Learn docs;
p50/p99/p99.9 for PSSDv1 measured locally in [`../VM_DISK_INVESTIGATION.md`](../VM_DISK_INVESTIGATION.md).
NVMe, PSSDv2 and Ultra rows are estimates — **unmeasured** (see §8 gaps).

**How this interacts with our IO sizes.** `storage.rs` uses `ALIGN = 512`, so every read
is 512-byte aligned and a multiple of 512 — a perfect fit for 512e media. On a **4Kn
Premium SSD v2** each read straddles at most one extra 4 KiB block at each end, adding
≤ 8 KiB to a 42 KiB–2.4 MiB request (**≤ 19% at the smallest `T`, < 1% at `T` ≥ 512**).
Not a concern, but it argues against pushing `T` below ~64 on 4Kn media.

**Blob's 4 MiB metering unit.** Blob transactions are billed per 4 MiB *or part thereof*,
so a read only costs more than one operation once it exceeds 4 MiB:

| `split_thr` | req KiB (minmax8, d=768) | 4 MiB units billed | req KiB (f16, d=768) | 4 MiB units billed |
|---:|---:|---:|---:|---:|
| 512 | 280 | 1 | 546 | 1 |
| 1024 | 594 | 1 | 1,158 | 1 |
| 2048 | 1,188 | 1 | 2,316 | 1 |
| 4096 | 2,376 | 1 | **4,632** | **2** |
| 8192 | **4,752** | **2** | 9,264 | 3 |

So the §7 blob costs hold unchanged for `T ≤ 2048` (minmax8) — **but the "coarsen to cut
blob cost" strategy stops paying off past `T ≈ 4096`**, where each read starts billing as
2+ operations and `$/query` flattens out. That places a hard ceiling on lever (3) in §6.

### 4.3 Host caching: Premium SSD v1 vs v2

| | **Premium SSD v1** | **Premium SSD v2** |
|---|---|---|
| Host caching | **ReadOnly / ReadWrite / None** | **Not supported at all** |
| Read latency, cache **hit** | **~0.1–0.3 ms** (host RAM) | n/a |
| Read latency, cache **miss** | ~0.6–1.5 ms (falls back to uncached) | n/a |
| Read latency, caching = **None** | ~0.6–1.5 ms p50, ~4 ms p99 | **~0.4–0.8 ms p50, ~2 ms p99** |
| IOPS accounting on a hit | against the VM's **cached** limit (D16as_v5: 75,000 IOPS / 1,000 MB/s) | against the VM's **uncached** limit only |
| Effect of a hit | Frees disk IOPS for misses | n/a |

**Consequences for this workload:**

1. **PSSDv2 uncached is *faster* than PSSDv1 uncached** (~0.5 ms vs ~1.0 ms p50) and has a
   far tighter tail (~5 ms vs ~20 ms p99.9). Its lack of caching is not a regression —
   v2's raw path beats v1's raw path.
2. **But PSSDv1 *with a cache hit* beats PSSDv2** (~0.2 ms vs ~0.5 ms). If an index fits in
   host RAM, cached v1 is the lowest-latency managed-disk option. That is exactly the
   regime this dev box is in — and exactly why §2's numbers are not disk numbers.
3. **The cache is useless at the scale this architecture targets.** The premise is *many*
   indexes with a small RAM footprint; the host cache is sized to the VM, so hit rate
   collapses as soon as the working set exceeds host RAM. **Assume caching = None and
   design to the uncached column.**
4. **Caching converts a bandwidth problem into a capacity problem.** On a cache hit you are
   no longer bound by the disk's 250 MB/s or 2,000 MB/s — you are bound by VM RAM. This is
   why the ListCache (`src/cache.rs`) exists, and it is the right lever *if* RAM is
   available. It is the wrong lever for the "many indexes, little RAM" target.
5. ⚠️ **ReadWrite caching is not durable** — writes are acknowledged before persistence.
   Fine for a read-only index; **not** safe for the online build's write path.

### 4.4 Tail amplification — why TTFB percentiles dominate

**What QD means here.** `QD` = *queue depth*: how many reads are in flight at once.
It is not a tuning knob — it is the hard constant
[`MAX_IO_CONCURRENCY = 128`](../diskann-disk/src/utils/aligned_file_reader/windows_aligned_file_reader.rs)
(same value in `linux_aligned_file_reader.rs`). `Searcher::search_profiled` hands **all**
`R` reads to `reader.read()` in one call, and the reader then splits them into
`⌈R/128⌉` **serial waves**: it submits up to 128 overlapped reads (IOCP) / SQEs (io_uring),
**blocks until every one of them completes**, and only then submits the next wave.

So a query is not one deep queue of `R` reads — it is a chain of `⌈R/128⌉` barriers,
and **each barrier waits for the slowest read in its wave**:

| `split_thr` | reads `R` | waves | wave sizes | quantile sampled per wave |
|---:|---:|---:|---|---|
| 50 | 359 | **3** | 128 + 128 + 103 | p99.22, p99.22, p99.03 |
| 120 | 219 | **2** | 128 + 91 | p99.22, p98.90 |
| 790 | 60 | **1** | 60 | p98.33 |

A wave of `w` reads samples the `(1 − 1/w)` quantile, and a full wave of 128 samples
**p99.22**. **The median TTFB is nearly irrelevant; p99–p99.9 is the design number.**

Read latency implied by the tail alone (Σ of wave maxima, before any bandwidth term):

| Medium | p50 | p99 | p99.9 | `T`=790 (R=60) | `T`=120 (R=219) | `T`=50 (R=359) |
|---|---:|---:|---:|---:|---:|---:|
| Local NVMe (Lsv4/Lasv3) | 0.08 ms | 0.2 ms | 0.8 ms | **0.2 ms** | **0.6 ms** | **1.0 ms** |
| Ultra Disk | 0.40 ms | 1.2 ms | 3.0 ms | 1.2 ms | 2.8 ms | 4.5 ms |
| Premium SSD v2 | 0.60 ms | 2.0 ms | 5.0 ms | 2.0 ms | 4.7 ms | 7.6 ms |
| PSSDv1 host-cached | 0.21 ms | 1.1 ms | 16.0 ms | 1.1 ms | 5.8 ms | 11.0 ms |
| PSSDv1 uncached | 1.00 ms | 4.0 ms | 20.0 ms | 4.0 ms | 11.9 ms | 20.3 ms |
| Premium block blob | 4.00 ms | 6.9 ms | 20.0 ms | 6.9 ms | 17.0 ms | 27.5 ms |
| **Standard hot blob** | 25.0 ms | 287 ms | 800 ms | **283 ms** | **698 ms** | **1,127 ms** |

Four conclusions:

- **Standard hot blob is disqualified.** Its 287 ms p99 makes any query issuing ≥25 reads
  a several-hundred-millisecond operation. The §5 figure of 25 ms was the *mean* and is
  badly misleading. Do not use standard tier on the query path at any `T`.
- **Fine partitioning is tail-sensitive; coarse partitioning is bandwidth-sensitive.**
  Small `T` (many small reads) is safe *only* on media with a tight tail — local NVMe, or
  PSSDv2/Ultra. On any fat-tailed medium, raise `T` to cut `R`.
- **This is an independent argument for the same `T` ≥ 512 recommendation**, arrived at
  from the latency distribution rather than from RAM. Note that `T` ≥ 512 also drops most
  queries to `R` ≤ 128, i.e. **a single wave** — which removes the barrier entirely.
- **The per-wave barrier is itself an optimization target.** Nothing requires waiting for
  all 128 completions before submitting more; a sliding window that keeps 128 reads in
  flight continuously would remove `⌈R/128⌉ − 1` serialized tail events. This matters
  only at `R` > 128, i.e. exactly the fine-`T` regime, and it compounds with the window
  coalescing already listed as deferred item **B** in `PERFORMANCE.md`.

### 4.5 Prices (East US, LRS, Azure Retail Prices API)

| Medium | Capacity | Provisioned performance | Per-operation |
|---|---|---|---|
| Local NVMe Lasv3 | **$0.237/GiB-mo** (implied, VM bundle) | included | **none** |
| Local NVMe Lsv3/Lsv4 | **$0.265/GiB-mo** (implied) | included | **none** |
| **Premium SSD v2** | **$0.0803/GiB-mo** | **$0.00511/IOPS-mo** (3,000 free) · **$0.04015/MBps-mo** (125 free) | **none** |
| Premium SSD v1 P40 (2 TiB) | $259.05/mo flat | included (7,500 IOPS) | **none** |
| Ultra Disk | $0.1197/GiB-mo | $0.0496/IOPS-mo · $0.3497/MBps-mo | **none** |
| **Premium block blob** | $0.150/GB-mo | — | **$0.14 / million reads** |
| **Standard hot blob** | $0.0208/GB-mo | — | **$0.40 / million reads** |
| Standard cool blob | $0.0152/GB-mo | — | **$1.00 / million reads** + $0.01/GB retrieval |

Notes:
- Managed disks (all tiers) have **no transaction charge** — IO is free once provisioned.
- Blob read operations are billed **per 4 MiB or part thereof** — see §4.2. Our reads stay
  within one unit for `T ≤ 2048` (minmax8), so the per-read prices above apply directly.
- Host caching: see §4.3. PSSDv2 does not support it; PSSDv1 does (this VM uses ReadWrite),
  which is why §2's numbers are memory-served.
- **PSSDv2's provisioned-MB/s charge is the real cost driver here**, not capacity:
  1,000 MB/s = **$35.13/mo**, 2,000 MB/s = **$75.28/mo**, versus **$0.59/mo** for the
  7.4 GiB of data. This workload buys bandwidth, not space.
- Same-region VM ↔ Blob traffic is **free**; cross-region is $0.02/GB.
- Managed-disk IOPS accounting: an IO ≤ 256 KiB = **1 IOPS**; larger IOs consume several.
  Our reads run 25 KiB – 2.4 MiB, so `T ≥ 512` (f16) starts consuming multiple IOPS/read.
- VM uncached caps usually bind before disk caps: `D16as_v5` = 25,600 IOPS / 384 MB/s;
  `E32bds_v5` = 174,200 / 4,800; `E112ibds_v5` = 400,000 / 10,000.

---

## 5. Latency

Model: `max( ⌈ios/QD⌉ · service_latency , bytes / bandwidth )`, QD = 128 (IOCP/io_uring),
QD = 256 assumed for blob HTTP.

### At the measured iso-recall 93.5% operating points (2.82M × 1536)

| `split_thr` | IOs/q | MB/q | L8as_v3 | **L32s_v4** | PSSDv2 | P40 | Prem blob | Std hot blob |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 359 | 15.8 | 7.9 ms | **1.3 ms** | 7.9 ms | 63.3 ms | 8.0 ms | 50.0 ms |
| 120 | 219 | 26.4 | 13.2 ms | **2.2 ms** | 13.2 ms | 105.6 ms | 4.0 ms | 25.0 ms |
| 790 | 60 | 55.6 | 27.8 ms | **4.6 ms** | 27.8 ms | 222.3 ms | 4.0 ms | 25.0 ms |

### Projected, N=10M × 768, `nlist` = 256, `minmax8`

| `split_thr` | MiB/q | L8as_v3 | **L32s_v4** | PSSDv2 | P40 | Prem blob | Std hot blob |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 7.1 | 3.7 | **0.6** | 3.7 | 29.9 | 4.0 | 25.0 |
| 128 | 15.1 | 7.9 | **1.3** | 7.9 | 63.5 | 4.0 | 25.0 |
| 256 | 33.5 | 17.6 | **2.9** | 17.6 | 140.6 | 4.0 | 25.0 |
| 512 | 70.1 | 36.8 | **6.1** | 36.8 | 294.0 | 4.0 | 25.0 |
| 1024 | 148.6 | 77.9 | **13.0** | 77.9 | 623.4 | 6.2 | 25.0 |
| 2048 | 297.1 | 155.8 | **26.0** | 155.8 | 1,246.2 | 12.5 | 25.0 |
| 4096 | 594.1 | 311.5 | **51.9** | 311.5 | 2,491.9 | 24.9 | 25.0 |

Add the **CPU floor** — centroid graph search, RAM-only, measured at `centroid_search_l`=1024:
**4.5 ms** at `k`=115K · **4.1 ms** at `k`=39K · **2.2–2.6 ms** at `k`=4.8K. This floor
*falls* as `T` rises (fewer centroids), partly offsetting the rising read cost.

### Reading the latency tables

- **Disk media are bandwidth-bound, not latency-bound.** Premium SSD v2 (0.7 ms/IO) and
  local NVMe (0.1 ms/IO) yield *identical* latency, because both cap at 2,000 MB/s and one
  query already submits 128 concurrent reads. This reproduces §9.2/§11: **the only way to
  cut latency is to read fewer bytes.** Per-IO service latency is nearly irrelevant.
- **Only raw MB/s separates the disk options.** L32s_v4 at 12 GB/s is **6× faster than
  everything else**, on bandwidth alone. It is the single highest-leverage hardware choice.
- **Blob is flat in `T`** (round-trip-bound, not bandwidth-bound) and therefore *wins* at
  large `T` — but its floor (4 ms premium / 25 ms standard) is worse than NVMe below
  `T ≈ 512`, and it assumes 256 concurrent GETs/query. At QD=64 the premium row is ~16 ms.
- ⚠️ **The blob columns above are means and are optimistic.** Per §4.4 a query samples the
  p99, not the p50: premium block blob is really **7–17 ms**, and standard hot blob is
  **287–700 ms**, not 25 ms. Treat the standard-blob column as disqualifying.
- **The P40 column is what a real (uncached) managed disk of that tier does** — 8–16×
  worse than production hardware. Do not benchmark here.

---

## 6. Throughput

Max sustainable **QPS per index**, device-limited.

### At the measured iso-recall 93.5% points

| `split_thr` | IOs/q | MB/q | L8as_v3 | L32s_v4 | PSSDv2 | P40 | **Blob (1 blob)** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 359 | 15.8 | 127 | **759** | 127 | 16 | **8** |
| 120 | 219 | 26.4 | 76 | **455** | 76 | 9 | **14** |
| 790 | 60 | 55.6 | 36 | **216** | 36 | 4 | **50** |

### The blob wall

The documented target request rate for a **single block blob is 3,000 req/s**, and one
index is one `.graphivf_lists` blob:

| `nlist` | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| **max QPS for one index blob** | 93.8 | 46.9 | 23.4 | **11.7** | 5.9 | 2.9 |

Mitigations in order of preference:
1. **Shard each index's list file across ≥8 blobs** (cluster id → blob). Cheapest fix; the
   account-level ceiling (40,000 req/s standard, 200 Gbps premium) is far higher.
2. **Coalesce adjacent probed windows** — deferred optimization **B** in `PERFORMANCE.md`.
   Worth far more on blob (fewer *billed* ops and round trips) than on disk.
3. **Raise `T`** to buy a lower `nlist` — but only ~`T^-0.65`, so this is the weakest lever.

---

## 7. Cost

### 7a. Cost decomposition per index (10M × 768, `minmax8` = **7.38 GiB**, `nlist`=256)

| Medium | Capacity $/mo | Provisioned perf $/mo | **Marginal IO $/M queries** |
|---|---:|---:|---:|
| Local NVMe (Lasv3 bundle) | $1.75 | included | **$0.00** |
| Local NVMe (Lsv3/Lsv4) | $1.95 | included | **$0.00** |
| Premium SSD v2 | $0.59 | **$122.00** @20K IOPS/1 GB/s · $468.75 @max | **$0.00** |
| Premium block blob | $1.19 | n/a | **$35.84** |
| Standard hot blob | $0.16 | n/a | **$102.40** |

### 7b. Marginal IO cost at the measured iso-recall 93.5% points

| `split_thr` | IOs/q | Premium block blob | Standard hot blob |
|---:|---:|---:|---:|
| 50 | 359 | **$50.26 / M queries** | $143.60 / M |
| 120 | 219 | **$30.66 / M queries** | $87.60 / M |
| 790 | 60 | **$8.40 / M queries** | $24.00 / M |

### 7c. Fully-loaded serving cost at device saturation ($/M queries, N=10M × 768, `nlist`=256)

Includes VM + storage + transactions; blob assumed 8-way sharded so the per-blob wall
is not binding.

| `split_thr` | MiB/q | L8as_v3 NVMe | **L32s_v4 NVMe** | D16as_v5 + PSSDv2 | D16as_v5 + prem blob |
|---:|---:|---:|---:|---:|---:|
| 256 | 33.5 | $3.05 | **$2.26** | $8.35 | $37.88 |
| 512 | 70.1 | $6.37 | **$4.74** | $17.48 | $37.88 |
| 1024 | 148.6 | $13.51 | **$10.04** | $37.05 | $37.88 |
| 2048 | 297.1 | $27.00 | **$20.08** | $74.08 | **$38.22** |
| 4096 | 594.1 | $53.99 | $40.15 | $148.12 | **$40.60** |

### The decisive results

```
$/query on blob  =  ios × price_per_read          ← falls as T^-0.65
$/query on disk  ∝  bytes/query / device_capacity ← rises as T^+0.46
```

1. **Blob is 12–50× more expensive than NVMe in the useful `T` range** (`T` ≤ 1024).
   At `nlist`=256 premium blob is **$35.84/M queries** — at 100 QPS sustained, **$93/day
   in transactions alone** for one index.
2. **But there is a crossover at `T ≈ 1024–2048.`** Above it, bandwidth on disk becomes so
   expensive that blob wins outright. If the architecture is genuinely RAM-starved (which
   forces large `T` anyway), **blob stops being obviously wrong** — this is the one regime
   where "many indexes in blob" is defensible.
3. **On blob the only cost lever is `nlist`.** Halving `nlist` halves cost; halving `T`
   does nothing to cost (and *raises* it via the recall→`nlist` coupling).
4. **On PSSDv2 the cost is provisioned MB/s, not capacity.** $0.59/mo of data behind
   $122/mo of bandwidth. Provision honestly against the MB/q column in §3.
5. Recall anchors confirm `nlist` responds only weakly to granularity: 85% recall@10 needed
   `nlist`=128 at `c̄`=531 but `nlist`=256 at both `c̄`=66 and `c̄`=27 — an **8× coarser**
   partition bought only **2× fewer probes** while costing **4× the bytes**.

---

## 8. Recommendations

| Decision | Recommendation | Why |
|---|---|---|
| **Storage medium** | **Local NVMe (L32s_v4 / Lasv3)** primary; **Premium SSD v2** where capacity must exceed local NVMe or survive VM loss | Bandwidth-bound workload. L32s_v4's 12 GB/s is 6× every alternative and is the single biggest latency lever available |
| **Blob** | **Premium block blob only** — as a cold tier behind `ListCache`, or the *only* option if `T` ≥ 2048 is forced by RAM budget. Shard ≥8 ways. **Never standard tier** | 3,000 req/s per blob → 12 QPS/index; $35.84/M queries. Standard's 287 ms p99 → ~700 ms/query (§4.4) |
| **`split_threshold`** | **512–2048** | Three independent arguments converge here: below 512 the f32 centroid graph (84–929 MiB/index) defeats the low-RAM premise **and** the read count `R` pushes you into the p99.7 tail (§4.4); above 2048 read latency exceeds 100 ms and blob's 4 MiB metering kicks in (§4.2) |
| **Host caching** | **Assume `None`; design to the uncached column** | Cache hit rate collapses once many indexes exceed host RAM — the exact premise of this architecture (§4.3) |
| **`data_type`** | **`minmax8`** | Halves bytes/query vs f16, quarters vs f32, for ~0.5 recall pts |
| **`nlist`** | as low as recall permits | The *only* cost axis on blob, and a linear latency axis everywhere |
| **VM** | `L32s_v4` / `L32as_v3`. **Not** `D16as_v5` + P40 | Current dev box caps at 250 MB/s and its results are memory-served, not disk-served (§2) |
| **Next code change** | **Window coalescing** (`PERFORMANCE.md` item **B**), then **continuous IO submission** (replace the per-wave barrier in the aligned readers with a sliding 128-deep window) | Both attack `R`: coalescing cuts the read count that prices blob and gates the 3,000 req/s wall; the sliding window removes `⌈R/128⌉ − 1` serialized p99.2 tail events (§4.4) |

### Gaps to close before trusting this model in production

1. **No measurements exist on Premium SSD v2 or local NVMe.** Every §4.2/§5/§6 number for
   those media is analytical, and their **TTFB percentiles are estimates**. Since §4.4
   shows the p99–p99.9 governs query latency, **measuring the read-latency distribution
   (not the mean) on the target medium is the single highest-value experiment.** §2 shows
   the current results are memory-served, so nothing here is validated against a device.
2. **`nlist` → recall at 10M × 768 is extrapolated** from 1M × 384 and 2.82M × 1536 corpora.
   `nlist` drives cost, latency, *and* tail exposure, so redo this at target scale.
3. **`ρ(T)` was fit on one corpus** with `reassign_neighbors`=32, `batch_size`=4096. It may
   shift with those knobs; re-fit if they change materially.
4. **Blob latency rows assume 256 concurrent GETs/query**, which real clients rarely sustain.
   Measure achievable concurrency, and measure blob TTFB percentiles from inside the region.
5. **Window coalescing is unquantified** — it attacks `R` directly, so it improves blob cost,
   the 3,000 req/s wall, *and* tail exposure simultaneously. Highest-value code change.
6. **The 4Kn read-amplification estimate for PSSDv2 is untested** — confirm whether the disk
   is provisioned 512e or 4096, since `ALIGN = 512` assumes 512e.
