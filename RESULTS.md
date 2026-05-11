# Dataset = Wiki Screenshots
## #Docs: 10K docs, #Queries: 1K, #Build-threads: 16,, #Query-threads: 16
Abbreviations:
- Full-MV1280 : Full precision (FP16) multi-vector with up-to 1280 vecs x 128-dim ==> 320KB/doc
- Full-MV32 : Full precision (FP16) multi-vector with 32 vecs x 128-dim. 32-vecs are found using HAC + Wards-criterion ==> 8KB/doc
- MP-MV1280 : Mean Pool of MV1280 ==> 256B/doc
- MP-MV32 : Mean Pool of MV32 ==> 256B/doc
- SphQ-MV32 : 1-bit Spherical quantization (SPQ) applied to Full-MV32 ==> 512B/doc
- SphQ-MV1280 : 1-bit Spherical quantization (SPQ) applied to Full-MV1280 ==> 20KB/doc

| Build | Inner-loop | Re-rank | Recall-10@100 (Ls=100) | Avg. Query latency (Ls=100, ms) | Inner time (ms) | Rerank time (ms) | Recall-10@100 (Ls=500) | 
|-------|------------|----------|----------------|----------------|-------------------------|----------------------|-----------------------|
| MP-MV1280 | MP-MV1280 | Full-MV1280 | 0.62 |  19.47 ms | 0.38 ms| 19.48 ms | 0.8533 |
| MP-MV1280 | SphQ-MV1280 | Full-MV1280 | 0.833 | 197.0ms | 181.6ms | 15.4ms | 0.971 |
| SphQ-MV32 | MP-MV1280 | Full-MV1280 | 0.62 |  20.4 ms | 0.41 ms| 19.99 ms | 0.8532 |
| SphQ-MV32 | SphQ-MV1280 | Full-MV1280 | 0.844 | 194.3ms | 178.7ms | 15.6ms | 0.978 |
| MP-MV32 | MP-MV32 | Full-MV32 | 0.64 | 0.6 ms | 0.23ms | 0.38ms | 0.846 |
| MP-MV32 | SphQ-MV32 | Full-MV32 | 0.704 | 6.7ms | 6.3ms | 0.43ms | 0.889 |
| MP-MV32 | SphQ-MV32 | SphQ-MV1280 | 0.71 | 19.6ms | 7.7ms | 11.9ms | 0.861 |
| SphQ-MV32 | MP-MV32 | Full-MV32 | 0.64 |  0.63 ms | 0.23 ms| 0.4 ms | 0.847 |
| SphQ-MV32 | SphQ-MV32 | SphQ-MV32 | 0.712 | 7.2 ms | 6.8ms | 0.37ms | 0.726 |
| SphQ-MV32 | SphQ-MV32 | Full-MV32 | 0.71 | 7.7ms | 7.2ms | 0.47ms | 0.893 |
| SphQ-MV32 | SphQ-MV32 | Full-MV1280 | 0.71 | 21.3ms | 6.8ms | 14.5ms | 0.921 |
| SphQ-MV32 | SphQ-MV32 | SphQ-MV1280 | 0.71 | 16.6ms | 7.0ms | 9.6ms | 0.862 |

Takeaways:
- Cost of a single distance comparison: MP-MVX = 200ns, SphQ-MV32: 2us, SphQ-MV1280: 80us, Full-MV32: 4us, Full-MV1280: 200us
- Index can be built with either mean-pool or spherical quantizer using MV32. Mean-pool using MV1280 does not seem to have any advantages over mean-pool with MV32.
- Search cannot be conducted using mean-pool vectors: Using SphQ-MV32 instead of MP-MV32 for inner-loop improves recall by ~6 points, but comes with a 10x increase in latency!! This is because each SphQ-MV32 distance comparison is currently 10x the cost of a single MP-MV32 distance comparison.
- Re-ranking with Full-MV1280 is effective, SphQ-MV1280 is not. Re-ranking with Full-MV32 is a good sweet spot: 40x faster than Full-MV1280 and higher recall than SphQ-MV1280.
- Current recommendation: Build index with MP-MV32/SphQ-MV32, running query inner-loop with SphQ-MV32 (oversample 50x), and re-rank with Full-MV32. This presents a good balance of recall and latency.

## #Docs: 100k, #Queries: 40k, #Build-threads: 16, #Query-threads: 16
| Build | Inner-loop | Re-rank | Recall-10@100 (Ls=100) | Avg. Query latency (Ls=100, ms) | Inner time (ms) | Rerank time (ms) | Recall-10@100 (Ls=500) | 
|-------|------------|----------|----------------|----------------|-------------------------|----------------------|-----------------------|
| MP-MV1280 | MP-MV1280 | Full-MV1280 | 0.437 | 17.9ms | 0.57ms | 17.4ms | 0.656 |
| MP-MV32 |  SphQ-MV32 | Full-MV32 | 0.543 | 10.8ms | 10.2ms | 0.58ms | 0.757 |


# Dataset = Vidore3
## Subset = 19K docs [Combined], Queries: 2.4K [Combined], #Build-threads: 16,, #Query-threads: 16
| Build | Inner-loop | Re-rank | Recall-10@100 (Ls=100) | Avg. Query latency (Ls=100, ms) | Inner time (ms) | Rerank time (ms) | Recall-10@100 (Ls=500) | 
|-------|------------|----------|----------------|----------------|-------------------------|----------------------|-----------------------|
| MP-MV1280 | MP-MV1280 | Full-MV1280 | 0.753 | 38.8ms | 0.3ms | 38.5ms | 0.94 |
| SphQ-MV32 | SphQ-MV32 | Full-MV32 | 0.905 | 11.6ms | 10.8ms | 0.75ms | 0.982 |
| SphQ-MV32 | SphQ-MV1280 | Full-MV1280 | 0.964 | 442ms | 0.6ms  | 36ms | 0.995 |
| SphQ-MV32 [SYM] | SphQ-MV32 [SYM] | Full-MV32 | 0.833 | 21.3ms | 20.5ms | 0.8ms | 0.97 |

## Subset = ComputerScience [Easy], #Build-threads: 16,, #Query-threads: 16
| Build | Inner-loop | Re-rank | Recall-10@100 (Ls=100) | Avg. Query latency (Ls=100, ms) | Inner time (ms) | Rerank time (ms) | Recall-10@100 (Ls=500) | 
|-------|------------|----------|----------------|----------------|-------------------------|----------------------|-----------------------|
| MP-MV1280 | MP-MV1280 | Full-MV1280 | 0.892 | 29.3ms | 0.15ms | 29.2ms | 0.994 |
| SphQ-MV32 | SphQ-MV32 | Full-MV32 | 0.967 | 4.97ms | 4.45ms | 0.52ms | 0.993 |
| SphQ-MV32 | SphQ-MV1280 | Full-MV1280 | 0.985 | 210ms | 181ms  | 28ms | 0.999 |
| SphQ-MV32 [SYM] | SphQ-MV32 [SYM] | Full-MV32 | 0.936 | 10.2ms | 9.6ms | 0.6ms | 0.993 |

## Subset = Industrial [Hard], #Build-threads: 16,, #Query-threads: 16
| Build | Inner-loop | Re-rank | Recall-10@100 (Ls=100) | Avg. Query latency (Ls=100, ms) | Inner time (ms) | Rerank time (ms) | Recall-10@100 (Ls=500) | 
|-------|------------|----------|----------------|----------------|-------------------------|----------------------|-----------------------|
| MP-MV1280 | MP-MV1280 | Full-MV1280 | 0.635 | 42.9ms | 0.27ms | 42.6ms | 0.903 |
| SphQ-MV32 | SphQ-MV32 | Full-MV32 | 0.868 | 8.8ms | 8.2ms  | 0.6ms | 0.965 |
| SphQ-MV32 | SphQ-MV1280 | Full-MV1280 | 0.963 | 344ms | 312ms  | 31.9ms | 0.997 |
| SphQ-MV32 [SYM] | SphQ-MV32 [SYM] | Full-MV32 | 0.811 | 17.6ms | 16.8ms  | 0.8ms | 0.953 |

# Dataset = MSMARCO
## Subset = 100k docs, 3.8k queries
- MVX = Multi-vector with 128-dim vectors, variable number of vectors per doc: avg. 79 vecs/doc, max. 760. 

| Build | Inner-loop | Re-rank | Recall-10@100 (Ls=100) | Avg. Query latency (Ls=100, ms) | Inner time (ms) | Rerank time (ms) | Recall-10@100 (Ls=500) | 
|-------|------------|----------|----------------|----------------|-------------------------|----------------------|-----------------------|
| MP-MVX | MP-MVX | Full-MVX | 0.618 | 1.3ms | 0.43ms | 0.92ms | 0.798 |
| SphQ-MV32 | SphQ-MV32 | Full-MV32 | 0.828 | 8.3ms | 7.78ms | 0.5ms | 0.935 |
| SphQ-MVX | SphQ-MVX | Full-MVX | 0.818 | 17.8ms | 16.3ms | 1.5ms | 0.921 |
| SphQ-MV32 | Full-MV32 | Full-MVX | 0.832 | 9.32ms | 8.1ms | 1.3ms | 0.935 |