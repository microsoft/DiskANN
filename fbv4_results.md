## Recall@10 on enron-email-1M-fbv4 (1,087,932 × 384, fp16, IP, 1000 queries)

### 4-bit

| Pipeline | g=1.00 | g=0.90 | g=0.60 |
|---|---:|---:|---:|
| f32 → 4 (direct) | 0.9001 | **0.9084** | 0.8093 |
| 8 → 4 (recompress) | 0.9001 | 0.9061 | 0.8088 |

### 2-bit

| Pipeline | g=1.00 | g=0.90 | g=0.60 |
|---|---:|---:|---:|
| f32 → 2 (direct) | 0.5970 | 0.6370 | **0.6952** |
| 8 → 2 (recompress) | 0.5970 | 0.6375 | 0.6950 |

Baseline: f32 → 8-bit direct = **0.9934**.

Build time (whole 1.09M-vector corpus): direct compression ~290 ms; recompression from existing 8-bit ~220–280 ms.

## Recompress kernel latency (per-vector, dim=384, ~5.4M ops per measurement)

| Conversion | Legacy | New (g=1.0) | Δ vs legacy | New (g=0.6) | Δ vs legacy |
|---|---:|---:|---:|---:|---:|
| 8 → 4 | 1717 ns | 1840 ns | +124 ns (+7.2%) | 1955 ns | +238 ns (+13.9%) |
| 8 → 2 | 1757 ns | 1911 ns | +154 ns (+8.8%) | 1937 ns | +180 ns (+10.2%) |

## Takeaways

- **4-bit prefers `g = 0.9`** (best: 0.9084 vs 0.9001 at g=1.0, vs 0.8093 at g=0.6).
- **2-bit prefers `g = 0.6`** (best: 0.6952 vs 0.5970 at g=1.0).
- **Recompress ≡ direct** to within 0.0023 across all 12 cases — Math A holds at scale.
- **Latency**: new kernel adds ~7–14% per vector vs legacy; sub-2 μs per 384-dim vector → full 1.09M corpus recompresses in ~220–280 ms. Overhead is consistent across `g`.
