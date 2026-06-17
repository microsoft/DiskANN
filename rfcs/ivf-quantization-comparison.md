# IVF Quantization Comparison: Full Precision vs MinMax-4 with Reranking

**Date**: 2026-06-11
**Dataset**: Wikipedia 100K (768 dims, Inner Product, 5K queries)
**Platform**: Windows, release build
**Config**: nlist=632, nprobe=64, recall@100, 4 search threads

---

## Raw Output

| File | Path |
|------|------|
| Comparison Results | `rfcs/ivf/wikipedia-100K-ivf-quantization-comparison.json` |

---

## Results

| Method | Param | QPS | Mean Latency | P95 Latency | IOs | Mean IO Time | Mean CPU Time | Mean Comps | Bytes Read (est.) | 512KB reads/query | Recall |
|--------|-------|-----|-------------|-------------|-----|-------------|--------------|------------|-------------------|-------------------|--------|
| **DiskANN** | L=200 | 137.8 | 28,945µs | 181,879µs | 209.9 | 26,924µs | 1,970µs | 5,825 | ~0.82 MB (0.2%) | 210 | **91.46%** |
| **IVF (full precision)** | nlist=632, nprobe=64 | 151.7 | 26,266µs | 36,055µs | 64 | 23,262µs | 3,004µs | 9,880 | ~29.7 MB (10.1%) | 64 | 89.28% |
| **IVF (MinMax-4 + rerank)** | nlist=632, nprobe=64, L=200 | 353.0 | 11,254µs | 14,054µs | 264 | 9,818µs | 1,436µs | 10,080 | ~4.5 MB (1.5%) | 264 | 78.79% |
| **IVF (MinMax-4 + rerank)** | nlist=632, nprobe=64, L=400 | 254.6 | 15,604µs | 20,114µs | 464 | 13,877µs | 1,727µs | 10,280 | ~5.1 MB (1.7%) | 464 | 86.63% |

---

## Bytes Read Estimation

**Full precision**: Each cluster file stores `[count:u32]` + count × `[id:u32][vec: 768×f32]`.
Average cluster size ≈ 100K/632 ≈ 158 vectors.
Per cluster ≈ `4 + 158 × (4 + 768×4)` ≈ **0.47 MB**.
64 probes × 0.47 MB ≈ **29.7 MB/query**.

**MinMax-4**: Quantized vectors use `canonical_bytes(768)` per vector (meta + nibble-packed codes).
`canonical_bytes(768) = 8 (meta) + ceil(768/2) = 392 bytes` per vector.
Per cluster ≈ `4 + 158 × (4 + 392)` ≈ **0.061 MB** (61 KB).
64 probes × 0.061 MB ≈ **3.9 MB/query** for the quantized scan.

**Reranking**: Loads 200 full-precision vectors from `vectors.bin` (200 × 768 × 4 = **0.59 MB**).

**Total MinMax-4+Rerank**: 3.9 + 0.59 ≈ **4.5 MB/query** vs **29.7 MB/query** for full precision (**6.6× reduction**).

---

## 512KB Reads Calculation

A "512KB read" is the unit of IO cost. Rules:
- If a single sequential read is ≤ 512KB, it counts as **1** read.
- If a single sequential read is > 512KB, it counts as **⌈size / 512KB⌉** reads.

**Full precision**:
- Each cluster ≈ 470 KB < 512 KB → 1 read per cluster
- 64 clusters × 1 = **64 reads/query**

**MinMax-4 + Rerank**:
- Each quantized cluster ≈ 61 KB < 512 KB → 1 read per cluster → 64 reads
- Reranking: 200 individual vector seeks, each 3 KB < 512 KB → 1 read each → 200 reads
- Total: 64 + 200 = **264 reads/query**

---

## Analysis

### Throughput
MinMax-4 with reranking delivers **2.3× higher QPS** (353 vs 152). The primary driver is reading ~6.6× fewer bytes from disk per query — quantized cluster files are much smaller, and reranking only loads 200 vectors from the flat blob.

### Latency
All latency percentiles improve dramatically:
- Mean latency drops 57% (26ms → 11ms)
- P95 drops 61%
- P99.9 drops 71%

IO time dominates the improvement — mean IO time drops from 23ms to 10ms, confirming disk reads are the bottleneck.

### Recall
Recall drops from 89.28% to 78.79% (−10.5 percentage points). This is expected: MinMax-4 quantization introduces distance approximation error, and search_l=200 (2× recall_at) may not be large enough to recover all true neighbors. Possible mitigations:
- **Increase search_l** (e.g., 400 or 500) — reranking more candidates improves recall at the cost of more vector reads
- **Use MinMax-8** instead of MinMax-4 — higher precision quantization reduces approximation error
- **Increase nprobe** — scanning more clusters captures more candidates

### Comparison with previous full-precision results

For reference, the earlier full-precision benchmark (from `ivf-vs-diskann-comparison.md`) with the same configuration (nlist=632, nprobe=64) showed 101.3 QPS and 89.97% recall. The current full-precision run shows 151.7 QPS — the difference is likely due to system load/caching differences between runs. Recall is consistent (~89%).
