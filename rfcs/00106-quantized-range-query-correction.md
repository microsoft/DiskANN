# RFC: Range Query Correction for Quantized Vector Indices

## Summary

This RFC proposes a statistical correction mechanism for range queries on quantized vector indices. We introduce a two-phase approach: phase one, during index build, computing empirical distance ratio statistics (mean μ and standard deviation σ) from a representative sample to understand quantization bias when we generate quantization schema, phase two, during query time, using these statistics to conservatively expand query ranges in quantized space followed by full-precision reranking to ensure accuracy. The key formulas convert user-specified full-precision ranges to quantized thresholds: 
- For upper bound queries ($d \leq T$): $T_{\text{quantized}} = \frac{T_{\text{full}}}{\mu - k\sigma}$
- For lower bound queries ($d \geq T$): $T_{\text{quantized}} = \frac{T_{\text{full}}}{\mu + k\sigma}$

where k is a confidence multiplier (typically 2 for 95% confidence). The solution bridges the gap between user-specified full-precision ranges and quantized distance computations by using empirical statistics to expand search ranges and rerank results with full-precision distances.

## Motivation

### Problem Statement

When using quantized vector indices (e.g., Product Quantization), users specify range queries in **full-precision distance space**, but the index operates in **quantized distance space** where distances have systematic bias and variance from their true values.

**Example Problem:**
```
User Query: "Find all points within distance 10.0 of my query vector"
Reality: Quantized distances ≠ Full-precision distances
Result: Missing valid results or returning invalid ones
```

### Current Limitations

1. **Semantic Gap**: Users think in full-precision terms, indices operate in quantized space
2. **Accuracy Loss**: Direct quantized range queries miss valid results or include invalid ones
3. **Unpredictable Behavior**: No systematic way to account for quantization bias
4. **Manual Tuning**: Users must manually adjust ranges without principled guidance

## Detailed Design

### Mathematical Foundation

#### Distance Ratio Analysis

For any query-data pair, we define the scaling ratio:
$$\text{ratio} = \frac{d_{\text{full-precision}}(q, x)}{d_{\text{quantized}}(q, x)}$$

From a representative sample, we extract:
- **Mean ratio** (`μ`): Average full/quantized scaling factor
- **Standard deviation** (`σ`): Variability of the ratio

**Interpretation**: If `μ > 1`, quantized distances are smaller; if `μ < 1`, quantized distances are larger.

#### Distance Ratio Statistics on openai-v3 1m
##### Configuration Details
- **Data Type**: float32
- **Distance**: squared_l2


**Product Quantization (PQ)**

  | PQ Chunks | PQ Centers | Samples | Mean Ratio | Standard Deviation | Min Ratio | Max Ratio |
 |-----------|------------|---------|------------|-----------|-----------|-----------|
 | 256       | 256        | 100,000 | 1.295259   | 0.032240  | 0.992527  | 1.437354  |
 | 384       | 256        | 100,000 | 1.184973   | 0.023721  | 0.989759  | 1.297321  |
 | 384       | 256        | 1,000   | 1.187889   | 0.025986  | 1.089831  | 1.264932  |

**Spherical Quantization**

| Num Bits | Quantized Bytes | Transform | Samples | Mean Ratio | Standard Deviation | Min Ratio | Max Ratio |
|----------|-----------------|-----------|---------|------------|-----------|-----------|-----------|
| 1 | 390 | padding_hadamard(same_dim) | 100,000 | 1.000528 | 0.024086 | 0.896261 | 1.126034 |
| 1 | 390 | padding_hadamard(same_dim) | 1,000 | 1.000603 | 0.024424 | 0.924180 | 1.084980 |
| 1 | 262 | padding_hadamard(2048) | 1,000 | 1.001691 | 0.032577 | 0.906938 | 1.131068 |
| 1 | 262 | padding_hadamard(2048) | 10,000 | 1.000916 | 0.030967 | 0.886590 | 1.140134 |
| 1 | 262 | padding_hadamard(2048) | 100,000 | 1.000896 | 0.031404 | 0.874722 | 1.210204 |
| 2 | 774 | padding_hadamard(same_dim) | 100,000 | 1.000169 | 0.013314 | 0.943462 | 1.072325 |

##### Key Observations

###### Product Quantization
**Deviation Improvement (256 → 384 PQ Chunks)**
- **Mean Ratio**: change by ~8.5% (1.295 → 1.185)
- **Standard Deviation**: Reduced by ~26.4% (0.032 → 0.024)
- **Max Ratio**: change by ~9.7% (1.437 → 1.297)

**Sample Size Impact (384 PQ Chunks)**
- **100K vs 1K samples**: Similar mean ratios (1.185 vs 1.188)
- **Min Ratio**: Higher floor with smaller sample (0.990 vs 1.090)
- **Standard Deviation**: Slightly higher with smaller sample (0.024 vs 0.026), but 1k sample seems good enough

###### Spherical Quantization
- **Mean Ratio**: at 1.000169-1.000528 (~0.017-0.053% bias vs PQ's 18.5-29.5% bias)
- **Standard Deviation**: at 0.013314-0.024086 (~44-80% lower than best PQ configuration)
- **Range Stability**: Narrower ratio ranges vs PQ's wider variations
- **Quantization Overhead**: Minimal distortion with only 1-2 bit encoding

**Bit-Level Analysis (1-bit vs 2-bit)**
- **1-bit encoding (100K samples)**: μ=1.000528, σ=0.024086, range=[0.896-1.126]
- **1-bit encoding (1K samples)**: μ=1.000603, σ=0.024424, range=[0.924-1.085]
- **2-bit encoding (100K samples)**: μ=1.000169, σ=0.013314, range=[0.943-1.072]
- **Precision trade-off**: 2-bit achieves ~45% lower standard deviation and tighter range vs 1-bit
- **Storage efficiency**: 1-bit uses 50% less space (390 vs 774 bytes) with acceptable accuracy degradation

**Sample Size Impact (1-bit Spherical)**
- **100K vs 1K samples**: Similar mean ratios (1.0005 vs 1.0006) and standard deviations (~0.024)
- **Min/Max Range**: 1K sample shows narrower range (0.924-1.085 vs 0.896-1.126)
- **Stability**: Both sample sizes demonstrate consistent statistical characteristics
- **Practical conclusion**: 1K samples appear sufficient for 1-bit Spherical quantization statistics

**Comparative Analysis**
- **Bias**: Spherical is nearly unbiased (μ ≈ 1.0) while PQ shows systematic overestimation
- **Variance**: Spherical achieves 44-80% lower standard deviation than PQ
- **Robustness**: Spherical's narrow ratio range requires less safety margins

#### Range Conversion Formula

Given user-specified range bounds in full-precision space, we convert to quantized thresholds:

**For Lower Bound Queries** (distance ≥ threshold):
$$R_{\text{lower-quantized}} = \frac{R_{\text{lower-full}}}{\mu + k\sigma}$$

**For Upper Bound Queries** (distance ≤ threshold):
$$R_{\text{upper-quantized}} = \frac{R_{\text{upper-full}}}{\mu - k\sigma}$$

**For Range Queries** (lower ≤ distance ≤ upper):
Apply both transformations:
$$R_{\text{lower-quantized}} = \frac{R_{\text{lower-full}}}{\mu + k\sigma}$$
$$R_{\text{upper-quantized}} = \frac{R_{\text{upper-full}}}{\mu - k\sigma}$$

Where `k` is the confidence multiplier:
- `k = 1`: 68% confidence (~1-sigma)
- `k = 2`: 95% confidence (~2-sigma) 
- `k = 3`: 99.7% confidence (~3-sigma)

**Rationale**: 

The goal is to **over-capture in quantized space** to avoid false negatives, then filter with full-precision reranking:

- For upper bounds (≤ T): Divide by (μ - kσ) → **larger** quantized threshold → accept more candidates (including some beyond T)
- For lower bounds (≥ T): Divide by (μ + kσ) → **smaller** quantized threshold → accept more candidates (including some below T)
- In both cases: Full-precision reranking removes false positives
- This works regardless of whether μ > 1 or μ < 1 because the division operation automatically handles bias direction

**Strategy**:
1. Use conservative ratio bounds (μ ± kσ) to expand search in quantized space
2. Over-capture candidates is safer than under-capturing
3. Verify candidates with exact full-precision distances
4. Return only points meeting the original user-specified threshold

 
#### Intuitive Explanation: Safest Approach

The fundamental insight is:

> **Safest Approach** = Search up to `user_threshold / (minimum_possible_ratio)`

This ensures we capture **all valid points**, even those with unfavorable distortion ratios.

#### Why This Works

Given:

$$d_{\text{full}} = \text{ratio} \cdot d_{\text{quantized}}$$

Since ratios vary with standard deviation `σ`:
- Minimum possible ratio = `μ - kσ`
- Maximum possible ratio = `μ + kσ`

*Upper Bound Case*: User wants all points where

 $$d_{\text{full}} \leq T_{\text{full}}$$

**Worst case scenario**: A point has the minimum ratio (`μ - kσ`)
$$T_{\text{full}} = (\mu - k\sigma) \cdot d_{\text{quantized}}$$

Solving for the quantized threshold:
$$d_{\text{quantized}} = \frac{T_{\text{full}}}{\mu - k\sigma}$$

Since `(μ - kσ) < μ`, dividing by it gives a **larger** threshold → captures more candidates.

**Example**: If `T_full = 10.0`, `μ = 1.05`, `σ = 0.08`, `k = 2`:
- Minimum ratio = `1.05 - 2(0.08) = 0.89`
- Search threshold = `10.0 / 0.89 = 11.24` (vs naive `10.0/1.05 = 9.52`)
- We search farther to ensure we don't miss points with unfavorable ratios

*Lower Bound Case*: User wants all points where 

$$d_{\text{full}} \geq T_{\text{full}}$$

**Worst case scenario**: A point has the maximum ratio (`μ + kσ`)
$$T_{\text{full}} = (\mu + k\sigma) \cdot d_{\text{quantized}}$$

Solving for the quantized threshold:
$$d_{\text{quantized}} = \frac{T_{\text{full}}}{\mu + k\sigma}$$

Since `(μ + kσ) > μ`, dividing by it gives a **smaller** threshold → captures more candidates.

**Example**: If `T_full = 20.0`, `μ = 1.05`, `σ = 0.08`, `k = 2`:
- Maximum ratio = `1.05 + 2(0.08) = 1.21`
- Search threshold = `20.0 / 1.21 = 16.53` (vs naive `20.0/1.05 = 19.05`)
- We search closer to catch candidates with unfavorable ratios

#### Why Both Expand Search Space

| Query Type | Divisor | Effect | Intuition |
|-----------|---------|--------|-----------|
| Upper (≤ T) | μ - kσ (small) | Larger threshold | Search farther to find all ≤ T |
| Lower (≥ T) | μ + kσ (large) | Smaller threshold | Search closer to find all ≥ T |

In both cases, the **search space expands** to capture all valid candidates, and full-precision reranking filters false positives.

#### Robustness: Works for μ > 1 or μ < 1

This approach is **bias-agnostic**:
- If μ > 1 (quantized smaller): Division automatically expands appropriately
- If μ < 1 (quantized larger): Division still expands appropriately  
- No conditional logic needed — the math handles both cases automatically


**Guarantee**:
Using $k=2$ with $N \geq 1000$ samples means:
- ~95% confidence that the true ratio falls within $(\mu-2\sigma, \mu+2\sigma)$
- ~5% chance of missing points with extreme ratios (acceptable tradeoff)
- Full-precision reranking provides final correctness guarantee

#### Visual Illustration

```
Distance Relationship Example (μ = 1.05, σ = 0.08):
(Full-precision distances are ~5% larger than quantized)

Relationship: full_distance ≈ μ × quantized_distance
Inversion:    quantized_distance ≈ full_distance / μ

Full-Precision Distance Space:
|-------|-------|-------|-------|-------|-------|-------|
0       5      10      15      20      25      30      35
                    ↑
              User Threshold
           (full: "≤ 10.0")

Quantized Distance Space (quantized values are smaller):
|-------|-------|-------|-------|-------|-------|-------|
0       5      10      15      20      25      30      35
              ↑                       ↑
      Naive Conversion        Safe Expanded Threshold
      (10.0/1.05=9.5)          (10.0/0.89=11.2)

Range Conversion Process:
1. User specifies: distance ≤ 10.0 (full-precision)
2. Compute safe quantized threshold: 10.0 / (1.05 - 2×0.08) = 10.0 / 0.89 = 11.2
   (Divide by smaller value → get LARGER threshold → move RIGHT)
3. Search quantized index: distance ≤ 11.2
4. Rerank with full-precision: keep only distance ≤ 10.0
5. Return sorted results

```

**Key Insight**: By dividing full-precision thresholds by the appropriate ratio bound, we automatically get the correct movement direction in quantized space.

#### Confidence Interpretation

**2-sigma confidence (k=2)** means:
- 95% of quantized distances fall within the predicted range
- Only 5% are outliers that might be missed
- Optimal balance between recall and computational cost

### Implementation Architecture

```rust
struct QuantizationStats {
    mean_ratio: f64,       // μ: full/quantized scaling factor
    std_ratio: f64,        // σ: variability measure  
    confidence_level: f64, // k: typically 2.0
    num_samples: usize,    // validation sample size
}

impl QuantizationStats {
    fn convert_upper_bound(&self, user_upper: f32) -> f32 {
        let divisor = (self.mean_ratio - self.confidence_level * self.std_ratio)
            .max(0.5); // Safety clamp
        let converted = user_upper / divisor as f32;
        converted.min(user_upper * 1.5) // Cap at 1.5× inflation
    }
    
    fn convert_lower_bound(&self, user_lower: f32) -> f32 {
        let divisor = self.mean_ratio + self.confidence_level * self.std_ratio;
        user_lower / divisor as f32
    }
}
```

#### 2. Corrected Range Search

sample code 
```rust
pub fn corrected_range_search(
    &self,
    query: &[f32],
    user_range: f32,
    stats: &QuantizationStats,
) -> Vec<(NodeId, f32)> {
    
    // Phase 1: Convert upper bound to quantized space
    let quantized_range = stats.convert_upper_bound(user_range);
    
    // Phase 2: Search quantized index with converted range
    let candidates = self.quantized_range_search(query, quantized_range);
    
    // Phase 3: Rerank with full precision
    let mut results = Vec::new();
    for candidate in candidates {
        let fp_distance = self.compute_full_precision_distance(query, candidate.id);
        if fp_distance <= user_range {
            results.push((candidate.id, fp_distance));
        }
    }
    
    // Phase 4: Sort by actual distance
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    results
}
```


#### 3. Schema-Time Statistics Collection

For **production implementations**, we propose collecting quantization statistics during the PQ schema building phase, rather than at query time. This approach provides several advantages:

- **Pre-computed Statistics**: Ratio statistics (μ, σ) are sampled and stored when building the PQ schema, eliminating runtime computation overhead
- **Representative Sampling**: During schema creation, sample 1,000-10,000 data points to compute mean ratio (μ) and standard deviation (σ) 
- **Persistent Storage**: Statistics are serialized and stored alongside the PQ schema metadata for immediate availability
- **Offline Defaults**: As a bootstrapping mechanism, we can provide conservative default values based on common PQ configurations:
  - **PQ** (32+ chunks, 256+ centers): μ_default = 1.02, σ_default = 0.05 --> need to test with wiki etc dataset

**Implementation Strategy:**
```rust
struct PQSchemaMetadata {
    // Existing PQ parameters
    num_chunks: usize,
    num_centers: usize,
    
    // New: Quantization statistics
    quantization_stats: Option<QuantizationStats>,
    
    // Fallback defaults when statistics unavailable
    default_mean_ratio: f64,
    default_std_ratio: f64,
}

impl PQSchemaMetadata {
    fn get_stats(&self) -> QuantizationStats {
        self.quantization_stats.unwrap_or_else(|| {
            QuantizationStats {
                mean_ratio: self.default_mean_ratio,
                std_ratio: self.default_std_ratio,
                confidence_level: 2.0,  // More conservative when using defaults
                num_samples: 0,  // Indicates default values
            }
        })
    }
}
```

This approach enables immediate deployment of range query correction while allowing for more precise statistics collection as the system matures.
