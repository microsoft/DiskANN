# Stage 2 — Groundtruth, and auditing it

Recall is measured against an exact top-K list. If that list is wrong, every recall number
in the study is wrong in a way that looks like an algorithmic limitation. **Audit before
sweeping.**

## Format

| Offset | Type | Meaning |
|---|---|---|
| 0 | `u32` | `nq` |
| 4 | `u32` | `K` |
| 8 | `u32 × nq × K` | neighbour ids |
| … | `f32 × nq × K` | distances |

The runner's truthset loader accepts ids-only files or ids followed by distances. Keep the
distances: recall uses them to include exact ties at the cutoff, and the audit needs them.

## Computing it

Exact brute force over the **f32/fp16 source corpus** (not the quantized one):

```text
python diskann-graphivf/scripts/dataprep/vecprep.py groundtruth \
    --corpus corpus_fp16.bin --queries queries_sample1000_fp16.bin \
    --out groundtruth_recall_1000_query_1000.bin -k 1000
```

It chunks the corpus (`--chunk`, default 32768 rows) and merges a running top-K via
`argpartition`, so an 8 GB corpus never lands in memory whole. Select the same metric used
by the benchmark: default `squared_l2`, or `--metric mips` for maximum inner product.

Ids are global row indices into the corpus file. If the corpus was filtered, they index the
**filtered** corpus — regenerate the groundtruth after any filtering, never remap.

Write both the full-query-set groundtruth and the sampled-query one, using the same evenly
spaced indices the query subset used.

For a pre-created streaming runbook, query row `i` must continue to correspond to truthset
row `i` in **every** `step<stage>.gt<K>` file. If only the first `N` queries are required,
write a new query matrix with header `(N, dim)` and a new truthset per search stage with
header `(N, K)`. Truthset ids and distances are separate full row-major blocks, so byte-
truncating the original file is wrong: copy the first `N*K` ids, seek to the original
distance block, then copy the first `N*K` distances. See [stage 7](./07-online-runbooks.md).

A note on reproducibility: recomputing an existing groundtruth will not always reproduce it
id-for-id. Chunk size changes BLAS summation order, which perturbs distances at the ~1e-6
level and reorders **exact ties**. Validating the tool against a stored NQ groundtruth gave
994/1000 identical id sets, with all six differences being pairs at bit-identical distances.
Compare with `np.allclose` on distances, not equality on ids.

## The audit — it runs automatically

`groundtruth` audits what it computed and **exits non-zero without writing** if the labels
look untrustworthy. `audit` re-runs the same checks on files that already exist:

```text
python vecprep.py audit --groundtruth groundtruth_recall_1000.bin --corpus corpus_fp16.bin
```

The checks, in the order they run:

1. **Degenerate corpus rows appearing inside the groundtruth** — count, plus
   mean/median/min/max per query, plus the implied recall ceiling.
2. **Distance monotonicity** along each row.
3. **Id range** — `max(id) < npts`.
4. **Distance ranges** for top-1 and top-K.

Expected: zero degenerate rows, monotone distances, ids in range, and a K-th distance
comfortably above the top-1 distance. `--allow-degenerate` overrides the block; there is
almost never a good reason to use it.

## Why this matters — the MTEB-NQ case

NQ shipped 507 all-zero documents (empty upstream text). Under squared L2 a zero vector
sits at distance $\|q\|^2$ from every query; for a unit-norm corpus that is exactly **1.0**,
which falls *inside* the top-1000 band. The audit found:

```
zero rows: 507
zero docs in GT top-1000: mean 506.6, median 507, min 194, max 507
zero docs in GT top-50:   mean 16.95, max 50
distance range, top-1000: 0.303 .. 1.461
```

More than half of every query's groundtruth was padding, so recall@1000 flatlined near 55%
no matter how much of the index was scanned. The symptom read as "the index cannot reach
high recall"; extrapolating the sweep suggested `nlist` would have to approach the full
cluster count. All of that was an artifact.

**Symptoms that should trigger this audit:**

- recall@1000 plateaus far below recall@50 and refuses to climb with `nlist`
- recall saturates well under 100% even when scanning a large fraction of the corpus
- the recall ceiling is suspiciously close to `1 - (degenerate_count / K)`

## Fixing it

Remove the rows from the corpus; do not try to special-case them.

1. Confirm no *judged* document is affected (the manifest usually says).
2. `vecprep.py filter --drop zero` on the already-written `fp16` **and** `minmax8` files.
   It slices rows **bit-identically** in chunks — re-deriving from source would perturb
   quantization, whereas slicing guarantees surviving vectors are unchanged — asserts the
   dropped rows really are all-zero, and writes a `*_ids.json` mapping back to original ids.
3. **Recompute the groundtruth** against the filtered corpus.
4. Rename everything downstream with a marker (`_nozero_`, `_nz_`) so the superseded runs
   cannot be confused with the corrected ones. The analysis selectors key on these paths.

Validation after the NQ fix:

```
K-th (1000th) distance range: 1.020 .. 1.507     # all above the 1.0 the zero rows occupied
monotone distances: True
```

That the whole band moved above 1.0 confirmed the diagnosis exactly.
