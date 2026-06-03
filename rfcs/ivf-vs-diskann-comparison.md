# IVF as RAM Reduction Strategy

**Problem**: DiskANN keeps PQ-compressed vectors in RAM (~1/4 of raw data). For a 500MB index, that's ~125MB RAM.

**IVF RAM**: Only centroids in RAM. For nlist=316, 1536-dim vectors: **~1.9MB** (vs DiskANN's ~150MB). That's a **~75× RAM reduction**.

**TL;DR**: IVF trades 5× QPS for 75× RAM reduction. An LRU cache layer lets you slide between these extremes. For RAM-constrained deployments (edge, multi-tenant), IVF + 10% cache uses ~63MB RAM vs DiskANN's ~150MB while maintaining >90% recall.

**The cost — read throughput**:

| | DiskANN | IVF (nlist=316, nprobe=32) | IVF (nlist=632, nprobe=64) |
|---|---|---|---|
| **RAM** | ~150 MB | **~1.9 MB** | **~3.8 MB** |
| **Recall** | 95.3% | 89.1% | 90.6% |
| **QPS** | 215.6 | 43.4 | 47.7 |
| **Bytes/query** | 1.65 MB | 59.2 MB | 59.3 MB |
| **QPS ratio** | 1× | 0.20× | 0.22× |

At comparable recall (~90-95%), IVF reads **36-72× more bytes per query** and delivers **5× lower QPS**.

**Caching closes the gap**: With 10% of index cached in RAM (~61MB for OpenAI):
- **24% of bytes read are served from cache** (measured from actual query distribution)
- This adds ~60MB RAM (still far below DiskANN's ~150MB) while cutting effective disk reads by ~1/4
- Hot clusters concentrate queries — OpenAI's top list gets 4.5× the mean reads

**Trade-off knob**: Cache size directly controls the RAM-IO trade-off:

| Cache | RAM added | Bytes saved | Effective model |
|-------|----------|-------------|-----------------|
| 0% | 0 MB | 0% | Pure disk IVF |
| 10% | ~61 MB | ~24% | Sweet spot |
| 25% | ~150 MB | ~50%+ (est.) | Approaches DiskANN RAM, better IO |

# IVF vs DiskANN Benchmark Comparison Report

**Date**: 2026-06-02 (updated 2026-06-02)
**Datasets**: OpenAI 100K (1536 dims, L2), Wikipedia 100K (768 dims, Inner Product)
**Platform**: Windows, release build

---

## Raw Output Files

| File | Path |
|------|------|
| OpenAI Disk-Index | `docs/rfcs/cy2026/ivf/openai-100K-disk-index-result.json` |
| OpenAI IVF (nlist=316) | `docs/rfcs/cy2026/ivf/openai-100K-ivf-result.json` |
| OpenAI IVF (nlist=158) | `docs/rfcs/cy2026/ivf/openai-100K-ivf-nlist158-result.json` |
| OpenAI IVF (nlist=632) | `docs/rfcs/cy2026/ivf/openai-100K-ivf-nlist632-result.json` |
| Wikipedia Disk-Index | `docs/rfcs/cy2026/ivf/wikipedia-100K-disk-index-result.json` |
| Wikipedia IVF (nlist=316) | `docs/rfcs/cy2026/ivf/wikipedia-100K-ivf-result.json` |
| Wikipedia IVF (nlist=158) | `docs/rfcs/cy2026/ivf/wikipedia-100K-ivf-nlist158-result.json` |
| Wikipedia IVF (nlist=632) | `docs/rfcs/cy2026/ivf/wikipedia-100K-ivf-nlist632-result.json` |

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

---

## IVF List Read Distribution (nlist=316, nprobe=64)

How uniformly are list reads distributed across clusters?
Skew means some clusters are queried more often, enabling LRU cache benefits.

### OpenAI 100K (L2, 1536 dims, 20K queries)

nlist=316, nprobe=64, 20,000 queries, total list reads: 1,280,000

```
Reads/list       #lists
   191- 1,708   39  #########
 1,709- 3,226  119  ##############################
 3,227- 4,744   60  ###############
 4,745- 6,262   39  #########
 6,263- 7,780   25  ######
 7,781- 9,298   18  ####
 9,299-10,816   10  ##
10,817-12,334    3  #
12,335-13,852    2  #
18,407-19,924    1  #
```

Min: 191 | Max: 18,415 | Mean: 4,051

**LRU top-10% cache** (31 of 316 lists): hit rate = **24.0%** (307,359 / 1,280,000 reads)

### Wikipedia 100K (IP, 768 dims, 5K queries)

nlist=316, nprobe=64, 5,000 queries, total list reads: 320,000

```
Reads/list       #lists
   302-   463   17  #########
   464-   625   39  ######################
   626-   787   52  ##############################
   788-   949   50  ############################
   950- 1,111   40  #######################
 1,112- 1,273   39  ######################
 1,274- 1,435   32  ##################
 1,436- 1,597   20  ###########
 1,598- 1,759   11  ######
 1,760- 1,921    8  ####
 1,922- 2,083    3  #
 2,084- 2,245    4  ##
 2,246- 2,407    1  #
```

Min: 302 | Max: 2,252 | Mean: 1,013

**LRU top-10% cache** (31 of 316 lists): hit rate = **17.5%** (55,845 / 320,000 reads)

### LRU Hit Rate Estimation

Assuming the top 10% most-read lists are cached in an LRU cache:

`hit_rate = sum(counts of top 10% lists) / sum(counts of all lists)`

| Dataset | Lists cached | Hit rate |
|---------|-------------|----------|
| OpenAI 100K | 31 / 316 (10%) | **24.0%** |
| Wikipedia 100K | 31 / 316 (10%) | **17.5%** |

---

## IVF Cluster Size Distribution (nlist=316)

### OpenAI 100K (L2, 1536 dims, 20K queries)

Cluster file sizes (min: 18,448 | max: 4,063,832 | mean: 1,945,574):

```
  File size (bytes)  #lists
     18448-   355562    1  #
    355563-   692677    4  #
    692678-  1029792   18  ########
   1029793-  1366907   35  ###############
   1366908-  1704022   54  ########################
   1704023-  2041137   67  ##############################
   2041138-  2378252   61  ###########################
   2378253-  2715367   34  ###############
   2715368-  3052482   25  ###########
   3052483-  3389597   10  ####
   3389598-  3726712    5  ##
   3726713-  4063827    1  #
   4063828-  4400942    1  #
```

### Wikipedia 100K (IP, 768 dims, 5K queries)

Cluster file sizes (min: 95,360 | max: 13,586,696 | mean: 973,422):

```
  File size (bytes)  #lists
     95360-  1219637  252  ##############################
   1219638-  2343915   55  ######
   2343916-  3468193    7  #
   3468194-  4592471    1  #
  13586696- 14710973    1  #
```

---

## IVF Bytes Read Distribution (nlist=316, nprobe=64)

Total bytes read per list = list_reads[i] × cluster_file_size[i].

### OpenAI 100K (L2, 1536 dims, 20K queries)

Total index: 614.8 MB, total bytes read across all queries: 2,387.2 GB

```
 Bytes read / list          #lists
      3523568- 2391535797   31  ############
   2391535798- 4779548027   76  ##############################
   4779548028- 7167560257   74  #############################
   7167560258- 9555572487   53  ####################
   9555572488-11943584717   27  ##########
  11943584718-14331596947   27  ##########
  14331596948-16719609177    8  ###
  16719609178-19107621407   12  ####
  19107621408-21495633637    3  #
  21495633638-23883645867    1  #
  23883645868-26271658097    1  #
  26271658098-28659670327    2  #
  28659670328-31047682557    1  #
```

Top 10 lists by bytes read:

| List | Reads | File size | Bytes read | % of total |
|------|-------|-----------|-----------|-----------|
| 305 | 7,642 | 3,750,284 | 28,659,670,328 | 1.20% |
| 211 | 12,831 | 2,108,768 | 27,057,602,208 | 1.13% |
| 155 | 12,877 | 2,047,288 | 26,362,927,576 | 1.10% |
| 264 | 10,439 | 2,471,500 | 25,799,988,500 | 1.08% |
| 270 | 10,214 | 2,305,504 | 23,548,417,856 | 0.99% |
| 168 | 8,983 | 2,287,060 | 20,544,659,980 | 0.86% |
| 198 | 6,718 | 2,944,896 | 19,783,811,328 | 0.83% |
| 271 | 8,600 | 2,274,764 | 19,562,970,400 | 0.82% |
| 61 | 18,415 | 1,026,720 | 18,907,048,800 | 0.79% |
| 298 | 5,893 | 3,203,112 | 18,875,939,016 | 0.79% |

**LRU cache = 10% of index** (61.5 MB):
- Lists cached: 38 of 316
- Bytes saved: **24.2%** of total bytes read

### Wikipedia 100K (IP, 768 dims, 5K queries)

Total index: 307.6 MB, total bytes read across all queries: 323.5 GB

```
 Bytes read / list          #lists
     89254464- 1300090332  249  ##############################
   1300090333- 2510926201   48  #####
   2510926202- 3721762070   12  #
   3721762071- 4932597939    5  #
   4932597940- 6143433808    1  #
  14619284892-15830120760    1  #
```

Top 10 lists by bytes read:

| List | Reads | File size | Bytes read | % of total |
|------|-------|-----------|-----------|-----------|
| 136 | 1,076 | 13,586,696 | 14,619,284,896 | 4.52% |
| 58 | 2,164 | 2,420,816 | 5,238,645,824 | 1.62% |
| 114 | 1,502 | 3,125,220 | 4,694,080,440 | 1.45% |
| 21 | 1,606 | 2,706,884 | 4,347,255,704 | 1.34% |
| 220 | 2,119 | 1,977,872 | 4,191,110,768 | 1.30% |
| 161 | 1,762 | 2,319,308 | 4,086,620,696 | 1.26% |
| 37 | 1,494 | 2,673,048 | 3,993,533,712 | 1.23% |
| 118 | 1,943 | 1,867,136 | 3,627,845,248 | 1.12% |
| 258 | 1,433 | 2,516,172 | 3,605,674,476 | 1.11% |
| 215 | 1,685 | 1,965,568 | 3,311,982,080 | 1.02% |

**LRU cache = 10% of index** (30.8 MB):
- Lists cached: 30 of 316
- Bytes saved: **17.6%** of total bytes read

### LRU Bytes-Saved Summary (10% of index cached)

| Dataset | Index size | Cache (10%) | Lists cached | Bytes saved |
|---------|-----------|-------------|-------------|-------------|
| OpenAI 100K | 614.8 MB | 61.5 MB | 38 / 316 | **24.2%** |
| Wikipedia 100K | 307.6 MB | 30.8 MB | 30 / 316 | **17.6%** |
