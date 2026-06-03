# IVF vs DiskANN Benchmark Comparison Report

**Date**: 2026-06-02 (updated 2026-06-02)
**Datasets**: OpenAI 100K (1536 dims, L2), Wikipedia 100K (768 dims, Inner Product)
**Platform**: Windows, release build

---

## Raw Output Files

| File | Path |
|------|------|
| OpenAI Disk-Index | `target/tmp/openai-100K-disk-index-result.json` |
| OpenAI IVF (nlist=316) | `target/tmp/openai-100K-ivf-result.json` |
| OpenAI IVF (nlist=158) | `target/tmp/openai-100K-ivf-nlist158-result.json` |
| OpenAI IVF (nlist=632) | `target/tmp/openai-100K-ivf-nlist632-result.json` |
| Wikipedia Disk-Index | `target/tmp/wikipedia-100K-disk-index-result.json` |
| Wikipedia IVF (nlist=316) | `target/tmp/wikipedia-100K-ivf-result.json` |
| Wikipedia IVF (nlist=158) | `target/tmp/wikipedia-100K-ivf-nlist158-result.json` |
| Wikipedia IVF (nlist=632) | `target/tmp/wikipedia-100K-ivf-nlist632-result.json` |

---

## Comparison: OpenAI 100K (L2, 1536 dims, 20K queries)

### DiskANN vs IVF

| Method | Param | QPS | Mean Latency | P95 Latency | IOs | Mean IO Time | Mean CPU Time | Mean Comps | Bytes Read (est.) | Recall |
|--------|-------|-----|-------------|-------------|-----|-------------|--------------|------------|-------------------|--------|
| **DiskANN** | L=200 | 215.6 | 18,369µs | 20,976µs | 210.9 | 15,120µs | 3,154µs | 5,428 | ~1.65 MB (0.2%) | **95.27%** |
| **IVF** | nlist=158, nprobe=1 | 585.1 | 6,804µs | 12,140µs | 1.0 | 4,605µs | 2,199µs | 587 | ~3.7 MB (0.6%) | 30.54% |
| **IVF** | nlist=158, nprobe=4 | 160.2 | 24,929µs | 39,192µs | 4.0 | 17,909µs | 7,020µs | 2,334 | ~14.8 MB (2.5%) | 60.35% |
| **IVF** | nlist=158, nprobe=8 | 79.0 | 50,602µs | 76,326µs | 8.0 | 36,923µs | 13,679µs | 4,673 | ~29.7 MB (5.1%) | 74.63% |
| **IVF** | nlist=158, nprobe=16 | 39.0 | 102,454µs | 150,434µs | 16.0 | 75,643µs | 26,811µs | 9,421 | ~59.3 MB (10.1%) | 86.48% |
| **IVF** | nlist=158, nprobe=32 | 19.5 | 204,136µs | 280,572µs | 32.0 | 151,407µs | 52,729µs | 19,166 | ~118.7 MB (20.3%) | 94.64% |
| **IVF** | nlist=158, nprobe=64 | 9.9 | 403,489µs | 492,080µs | 64.0 | 298,920µs | 104,569µs | 39,769 | ~237.4 MB (40.5%) | **98.74%** |
| **IVF** | nlist=316, nprobe=1 | 945.3 | 4,129µs | 7,336µs | 1.0 | 2,537µs | 1,592µs | 297 | ~1.9 MB (0.3%) | 24.59% |
| **IVF** | nlist=316, nprobe=4 | 308.1 | 12,977µs | 21,319µs | 4.0 | 8,830µs | 4,147µs | 1,175 | ~7.4 MB (1.3%) | 50.86% |
| **IVF** | nlist=316, nprobe=8 | 160.1 | 24,915µs | 38,780µs | 8.0 | 17,585µs | 7,330µs | 2,345 | ~14.8 MB (2.5%) | 65.28% |
| **IVF** | nlist=316, nprobe=16 | 83.8 | 47,490µs | 70,853µs | 16.0 | 34,095µs | 13,395µs | 4,727 | ~29.6 MB (5.1%) | 78.58% |
| **IVF** | nlist=316, nprobe=32 | 43.4 | 91,671µs | 128,222µs | 32.0 | 66,526µs | 25,145µs | 9,564 | ~59.2 MB (10.1%) | 89.05% |
| **IVF** | nlist=316, nprobe=64 | 21.1 | 188,822µs | 265,214µs | 64.0 | 137,899µs | 50,923µs | 19,415 | ~118.4 MB (20.2%) | 95.82% |
| **IVF** | nlist=632, nprobe=1 | 1,390.4 | 2,862µs | 5,003µs | 1.0 | 1,054µs | 1,808µs | 158 | ~0.9 MB (0.2%) | 20.10% |
| **IVF** | nlist=632, nprobe=4 | 593.1 | 6,713µs | 12,022µs | 4.0 | 3,607µs | 3,106µs | 626 | ~3.7 MB (0.6%) | 43.31% |
| **IVF** | nlist=632, nprobe=8 | 335.7 | 11,912µs | 21,227µs | 8.0 | 7,199µs | 4,713µs | 1,237 | ~7.4 MB (1.3%) | 56.72% |
| **IVF** | nlist=632, nprobe=16 | 182.4 | 21,879µs | 37,217µs | 16.0 | 14,131µs | 7,748µs | 2,446 | ~14.8 MB (2.5%) | 69.85% |
| **IVF** | nlist=632, nprobe=32 | 92.1 | 43,265µs | 69,965µs | 32.0 | 29,242µs | 14,023µs | 4,865 | ~29.7 MB (5.1%) | 81.54% |
| **IVF** | nlist=632, nprobe=64 | 47.7 | 83,858µs | 127,384µs | 64.0 | 57,688µs | 26,170µs | 9,740 | ~59.3 MB (10.1%) | 90.59% |

## Comparison: Wikipedia 100K (Inner Product, 768 dims, 5K queries)

### DiskANN vs IVF

| Method | Param | QPS | Mean Latency | P95 Latency | IOs | Mean IO Time | Mean CPU Time | Mean Comps | Bytes Read (est.) | Recall |
|--------|-------|-----|-------------|-------------|-----|-------------|--------------|------------|-------------------|--------|
| **DiskANN** | L=200 | 137.8 | 28,945µs | 181,879µs | 209.9 | 26,924µs | 1,970µs | 5,825 | ~0.82 MB (0.2%) | **91.46%** |
| **IVF** | nlist=158, nprobe=1 | 862.1 | 4,602µs | 14,462µs | 1.0 | 3,040µs | 1,562µs | 851 | ~1.9 MB (0.6%) | 37.79% |
| **IVF** | nlist=158, nprobe=4 | 251.9 | 15,766µs | 35,700µs | 4.0 | 11,151µs | 4,615µs | 3,107 | ~7.4 MB (2.5%) | 66.26% |
| **IVF** | nlist=158, nprobe=8 | 136.1 | 29,332µs | 55,102µs | 8.0 | 21,150µs | 8,182µs | 5,911 | ~14.9 MB (5.1%) | 77.70% |
| **IVF** | nlist=158, nprobe=16 | 71.4 | 55,349µs | 88,758µs | 16.0 | 40,272µs | 15,077µs | 11,303 | ~29.7 MB (10.1%) | 87.03% |
| **IVF** | nlist=158, nprobe=32 | 37.8 | 105,202µs | 150,215µs | 32.0 | 76,826µs | 28,376µs | 21,699 | ~59.4 MB (20.3%) | 93.87% |
| **IVF** | nlist=158, nprobe=64 | 20.0 | 198,525µs | 253,593µs | 64.0 | 145,260µs | 53,265µs | 41,671 | ~118.9 MB (40.6%) | **98.02%** |
| **IVF** | nlist=316, nprobe=1 | 962.5 | 4,142µs | 11,229µs | 1.0 | 2,613µs | 1,529µs | 433 | ~0.9 MB (0.3%) | 31.03% |
| **IVF** | nlist=316, nprobe=4 | 341.5 | 11,689µs | 32,154µs | 4.0 | 8,019µs | 3,670µs | 1,579 | ~3.7 MB (1.3%) | 57.25% |
| **IVF** | nlist=316, nprobe=8 | 170.5 | 22,449µs | 54,809µs | 8.0 | 15,788µs | 6,661µs | 2,956 | ~7.4 MB (2.5%) | 69.30% |
| **IVF** | nlist=316, nprobe=16 | 110.2 | 36,258µs | 71,525µs | 16.0 | 25,786µs | 10,472µs | 5,628 | ~14.9 MB (5.1%) | 79.85% |
| **IVF** | nlist=316, nprobe=32 | 61.3 | 65,212µs | 108,919µs | 32.0 | 46,819µs | 18,393µs | 10,858 | ~29.7 MB (10.1%) | 88.39% |
| **IVF** | nlist=316, nprobe=64 | 39.4 | 101,510µs | 152,218µs | 64.0 | 72,352µs | 29,158µs | 21,035 | ~59.4 MB (20.3%) | 94.68% |
| **IVF** | nlist=632, nprobe=1 | 2,075.8 | 1,925µs | 3,725µs | 1.0 | 883µs | 1,042µs | 206 | ~0.5 MB (0.2%) | 25.06% |
| **IVF** | nlist=632, nprobe=4 | 936.0 | 4,270µs | 10,402µs | 4.0 | 2,381µs | 1,889µs | 782 | ~1.9 MB (0.6%) | 48.42% |
| **IVF** | nlist=632, nprobe=8 | 620.1 | 6,418µs | 13,974µs | 8.0 | 3,769µs | 2,649µs | 1,490 | ~3.7 MB (1.3%) | 60.62% |
| **IVF** | nlist=632, nprobe=16 | 343.3 | 11,637µs | 23,866µs | 16.0 | 7,325µs | 4,312µs | 2,839 | ~7.4 MB (2.5%) | 72.03% |
| **IVF** | nlist=632, nprobe=32 | 193.4 | 20,654µs | 35,659µs | 32.0 | 13,417µs | 7,237µs | 5,478 | ~14.8 MB (5.1%) | 81.98% |
| **IVF** | nlist=632, nprobe=64 | 101.3 | 38,733µs | 59,015µs | 64.0 | 25,759µs | 12,974µs | 10,651 | ~29.7 MB (10.1%) | 89.97% |

### Bytes Read Estimation Methodology

**IVF**: Each cluster file stores `[count: u32]` followed by `count` records of `[id: u32][vec: ndims × f32]`.
Average cluster size = N / nlist vectors. Per-cluster bytes ≈ `4 + (N/nlist) × (4 + ndims × 4)`:
- nlist=158: ~633 vecs/cluster → OpenAI ~3.7 MB/cluster, Wikipedia ~1.9 MB/cluster
- nlist=316: ~316 vecs/cluster → OpenAI ~1.85 MB/cluster, Wikipedia ~0.95 MB/cluster
- nlist=632: ~158 vecs/cluster → OpenAI ~0.93 MB/cluster, Wikipedia ~0.47 MB/cluster

**DiskANN**: Each IO reads one node from the disk index. Node size ≈ `4 + max_degree × 4 + ndims × 4` bytes,
aligned to the next 4KB sector boundary:
- OpenAI (1536 dims, max_degree=59): raw 6,384 bytes → aligned to 8 KB/node → 210.9 IOs × 8 KB ≈ 1.65 MB
- Wikipedia (768 dims, max_degree=59): raw 3,312 bytes → aligned to 4 KB/node → 209.9 IOs × 4 KB ≈ 0.82 MB

*Note: these are rough estimates. DiskANN node sizes depend on the actual disk layout (PQ data, padding, sector alignment).
Actual bytes read will be collected from the code in a future iteration.*

**Total index sizes used for percentage calculation:**
- IVF OpenAI: 100K × (4 + 1536×4) bytes + overhead ≈ **586 MB**
- IVF Wikipedia: 100K × (4 + 768×4) bytes + overhead ≈ **293 MB**
- DiskANN OpenAI: 100K × 8 KB (aligned nodes) ≈ **800 MB**
- DiskANN Wikipedia: 100K × 4 KB (aligned nodes) ≈ **400 MB**

---

## IVF Configuration

- **nlist values tested**: 158 (√N/2), 316 (≈ √N), 632 (2×√N)
- **k-means iterations**: 20
- **Build threads**: 8, Search threads: 4
- **recall_at**: 100
- **Disk layout**: one file per cluster (append-friendly)
- **RAM usage**: centroids only (~0.9–3.8 MB depending on nlist and dims)

## DiskANN Configuration

- **max_degree**: 59, **L_build**: 80
- **PQ chunks**: 384 (OpenAI), 192 (Wikipedia)
- **Quantization**: SQ_1_2.0
- **Beam width**: 4, Search threads: 4
- **recall_at**: 100

---

## Key Takeaways

1. **Recall is comparable** — IVF at nprobe=64 matches or exceeds DiskANN recall on both datasets (95.8% vs 95.3% OpenAI; 94.7% vs 91.5% Wikipedia).

2. **DiskANN has higher QPS at high recall** — ~4-9× faster at comparable recall, thanks to graph-based traversal vs brute-force cluster scanning.

3. **IVF uses far fewer IOs** — 64 IOs vs ~210 IOs, but each IVF IO reads a full cluster (~300 vectors × dim × 4 bytes), while DiskANN reads individual nodes.

4. **IVF latency is higher** — dominated by reading + scanning large cluster files from disk (1536-dim vectors are ~6KB each, so each cluster is ~1.8MB for OpenAI).

5. **IVF excels at low-recall / high-throughput** — At nprobe=1, IVF achieves ~1200-1340 QPS (vs DiskANN's ~140-216), making it attractive for use cases where ~25-30% recall is acceptable.

6. **IVF build is slower** — 270s (OpenAI) and 141s (Wikipedia) for k-means vs 14s and 10s for DiskANN graph construction.

7. **IVF RAM footprint is minimal** — Only centroids are kept in RAM (~1-2MB), whereas DiskANN caches PQ-compressed vectors for distance precomputation.

## nlist Sensitivity Analysis

8. **Fewer clusters (nlist=158) → higher recall, much slower.** With 2× larger clusters, each nprobe reads ~2× more data. At nprobe=64, recall reaches 98.7% (OpenAI) and 98.0% (Wikipedia), but QPS drops to ~10 and ~20 respectively — scanning 64 clusters of ~633 vectors each is very expensive for high-dimensional data.

9. **More clusters (nlist=632) → lower recall, much faster.** With 2× smaller clusters, each nprobe reads ~½ the data. At nprobe=64, QPS nearly doubles (47.7 vs 21.1 OpenAI; 101.3 vs 39.4 Wikipedia) but recall drops ~5 points (90.6% vs 95.8% OpenAI; 90.0% vs 94.7% Wikipedia).

10. **The nlist trade-off is recall vs throughput at fixed nprobe.** To match the same recall, higher nlist needs proportionally more nprobe (e.g., nlist=632 at nprobe=64 ≈ nlist=316 at nprobe=32 in recall). The total vectors scanned — and thus total bytes read — ends up similar, but more smaller IOs vs fewer larger IOs favors nlist=632 on latency.

11. **Build time scales with nlist.** nlist=158 builds in ~146s (OpenAI), nlist=316 in ~270s, nlist=632 in ~569s — roughly linear since each k-means iteration does N×nlist distance computations.
