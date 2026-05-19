# diskann-mv-rerank-benchmark

End-to-end benchmark that

1. builds (or loads) a full-precision DiskANN graph on **single-vector** embeddings,
2. runs top-N single-vector graph-walk search per query,
3. reranks every candidate the graph walk produced with **multi-vector**
   Chamfer (via [`QueryComputer`] from `diskann-quantization::multi_vector`),
4. reports BEIR-style recall@N — `hits / min(N, |relevant|)` — over a
   variable-k groundtruth file.

[`QueryComputer`]: ../diskann-quantization/src/multi_vector/distance/query_computer/mod.rs

The pipeline is fail-loud by design: missing sidecars, unknown JSON fields,
malformed qrels, mismatched record counts, and mismatched MV dimensions all
produce named errors at validation time rather than silently corrupting recall.

---

## Bring your own dataset

You need **all** of the following — none is optional, the fallback heuristics
that used to make some of these optional silently produced wrong recall and
have been removed.

| File | Content |
|---|---|
| `docs.npy` | Single-vector doc embeddings, shape `(N_docs, D_sv)`, dtype `float16`/`float32`/`float64`. |
| `queries.npy` | Single-vector query embeddings, shape `(N_q, D_sv)`, same dtypes. |
| `embeddings_part*.pt` (docs) | `torch.load`-able to `dict[int corpus_id, ndarray of shape [K, D_mv]]`. |
| `embeddings_part*.pt` (queries) | Same dict shape. |
| `qrels.tsv` | `qid<TAB>doc_id<TAB>score` per row. Every row must have a numeric score column. Rows with `score > 0` are kept as positive judgments. |
| `docids.npy` | Int64 1-D array of length `N_docs`. Element `i` is the corpus id of doc row `i` in `docs.npy`. |
| `query_ids.txt` | One integer per line, count `N_q`. Line `i` is the qid of query row `i` in `queries.npy`. |

The dataset directory layout this README assumes (the user-local
`multivector_data/<name>/` is just a convention — point the script's flags
wherever your files actually live):

```
multivector_data/arguana/
├── singlevectors/documents/docs_chunk00000.npy
├── singlevectors/queries/query_vech.npy
├── multivectors/documents/embeddings_part*.pt
├── multivectors/queries/embeddings_part*.pt
├── docids/docids_chunk00000.npy
├── queryids/query_ids.txt
└── qrels/dev.tsv
```

### What if I don't have docids / query_ids?

You need them. The trivial 0..N forms work *only* if you're absolutely certain
the corpus id space is dense and the `.npy` rows are already in id order — in
which case it's a 3-line script:

```python
np.save("docids.npy", np.arange(docs.shape[0], dtype=np.int64))
with open("query_ids.txt", "w") as f: f.write("\n".join(str(i) for i in range(queries.shape[0])))
```

For BEIR-style data with gappy ids (ArguANA, etc.) you must supply real
sidecars — the converter will fail with `qrels doc-id N not present in
--docids sidecar` rather than silently scoring wrong docs.

---

## Step 1 — analyze (no writes)

```sh
python scripts/convert_dataset.py analyze \
    --single-vec-docs    path/to/docs.npy \
    --single-vec-queries path/to/queries.npy \
    --multi-vec-docs     "path/to/docs/*.pt" \
    --multi-vec-queries  "path/to/queries/*.pt" \
    --qrels              path/to/qrels.tsv
```

Prints `.npy` dtypes/shapes, `.pt` top-level types (must be `dict` for `convert`),
and qrels stats (positive count, relevant-per-query min/avg/max). Run this once
to confirm the inputs match the expected shapes before committing to a `convert`.

## Step 2 — convert

```sh
python scripts/convert_dataset.py convert \
    --single-vec-docs    multivector_data/arguana/singlevectors/documents/docs_chunk00000.npy \
    --single-vec-queries multivector_data/arguana/singlevectors/queries/query_vech.npy \
    --multi-vec-docs     "multivector_data/arguana/multivectors/documents/embeddings_part*.pt" \
    --multi-vec-queries  "multivector_data/arguana/multivectors/queries/embeddings_part*.pt" \
    --qrels              multivector_data/arguana/qrels/dev.tsv \
    --docids             multivector_data/arguana/docids/docids_chunk00000.npy \
    --query-ids          multivector_data/arguana/queryids/query_ids.txt \
    --out-dir            multivector_data/arguana/diskann \
    --name               arguana
```

Outputs five files in `--out-dir`:

| File | Binary layout |
|---|---|
| `arguana_docs.fbin` | `u32 LE npoints` + `u32 LE ndims` + `npoints*ndims` f32 row-major |
| `arguana_queries_dev.fbin` | same layout, filtered to dev qids (in qrels-file order) |
| `arguana_gt.bin` | variable-k: `u32 LE nqueries` + `u32 LE total_results` + `nqueries` u32 sizes + `total_results` u32 flat ids (in query order); no distance slab |
| `arguana_docs.mvbin` | concatenated records, each `u32 LE K` + `u32 LE D` + `K*D` f16 row-major; ordered to match `arguana_docs.fbin` via `--docids` |
| `arguana_queries_dev.mvbin` | same record layout, one record per dev qid in qrels-file order |

Notes:
- Single-vector files are upcast to f32 on disk; the Rust benchmark consumes f32.
- Multi-vector files stay f16 on disk; the Rust loader upcasts on read.
- The doc fbin / mvbin row orderings are guaranteed identical (both keyed by `--docids`).
- The docs `.mvbin` writer materializes all `.pt` records into a single dict before
  emitting them in `--docids` order. Comfortable up to a few million docs; on
  MS-MARCO-scale corpora (tens of millions of multi-vector docs) this will OOM —
  process the corpus in id-disjoint chunks and concatenate the resulting `.mvbin`s.

## Step 3 — run the benchmark

```sh
cargo run --release -p diskann-mv-rerank-benchmark -- run \
    --input-file diskann-mv-rerank-benchmark/example/multi-vector-rerank.json \
    --output-file out.json
```

The runner reads the example JSON (see "Input JSON schema" below), executes
each job sequentially, and writes a JSON array of per-job results.

### First-run bootstrap

`example/multi-vector-rerank.json` has two jobs: a **Build** that saves the graph
to `arguana_graph`, and a **Load** that reads it back. The runner pre-flight-checks
every input file before running any job, so the Load job's `load_path` must exist
when the run starts.

On the very first run (or after deleting the saved graph), comment out or remove
the Load job from the example, run the Build job, then restore the Load job for
subsequent runs.

---

## Input JSON schema

A Build job (paths are resolved against `search_directories`):

```json
{
  "type": "multi-vector-rerank-build",
  "content": {
    "source": {
      "index-source": "Build",
      "data_type": "float32",  "distance": "inner_product",
      "data": "arguana_docs.fbin",
      "max_degree": 32, "l_build": 64, "alpha": 1.2, "backedge_ratio": 1.0,
      "num_threads": 30, "start_point_strategy": "medoid",
      "save_path": "arguana_graph"
    },
    "search": {
      "queries": "arguana_queries_dev.fbin",
      "groundtruth": "arguana_gt.bin",
      "reps": 1, "num_threads": [30],
      "runs": [{ "search_n": 10, "search_l": [10, 50, 100, 200, 500] }]
    },
    "doc_mv":   "arguana_docs.mvbin",
    "query_mv": "arguana_queries_dev.mvbin"
  }
}
```

For a Load job swap `source` to `{"index-source": "Load", "data_type": ..., "distance": ..., "load_path": "arguana_graph"}`. Everything else is the same.

Key fields:

| Field | Meaning |
|---|---|
| `search_n` | top-N returned per query; also the K in BEIR `recall@N` and the cap on the denominator (`min(N, \|relevant\|)`) |
| `search_l` | per-run sweep of graph-walk list sizes. Every one of the L candidates is scored by the multi-vector Chamfer reranker; the top `search_n` (by rerank score) are returned. |
| `reps` | repetitions per `(num_threads, search_l)` config — latency percentiles are aggregated across reps |
| `start_point_strategy` | `"medoid"` \| `"first_vector"` \| `{"latin_hyper_cube": [n, seed]}` \| `{"random_samples": {...}}` \| `{"random_vectors": {...}}` |
| `distance` | `"inner_product"` \| `"squared_l2"` \| `"cosine"` \| `"cosine_normalized"` |

Unknown JSON fields are **rejected** at deserialization (typos surface as
`unknown field 'recall_k', expected 'search_n' or 'search_l'` rather than
being silently ignored).

## Adapting to a new BEIR dataset

1. Place data anywhere; the script's `--*` flags take absolute or relative paths.
2. Run `analyze` first; the `.pt` files must come out as `dict`. If they're a flat
   tensor or list, preprocess them into the dict form before continuing.
3. Confirm `max(qid)` in qrels is covered by `query_ids.txt`, and `max(doc_id)` by
   `docids.npy`. The converter errors loudly if any qrels id isn't in the sidecar.
4. Run `convert`. Then point the example JSON's `search_directories` and per-job
   `data` / `queries` / `groundtruth` / `doc_mv` / `query_mv` paths at the outputs.

## Output

Two outputs per run:

- **stdout** — a per-job table:
  ```
     L,    N,   Thr,   Cmps,   Hops,      QPS,   Mean(us),   p99(us),   Rerank(us),   Recall@N
  ===========================================================================================
    10,   10,    30,    407,     15,   9635.1,    2621.4,     4584us,      1145.0,     0.6467
    50,   10,    30,   1086,     54,   2815.9,    9047.5,    14975us,      4805.2,     0.6400
   ...
  ```
- **`--output-file <path>`** — a JSON array, one entry per job, with `build`
  stats (Build jobs only) and a `search` array containing recall fields
  (`recall_avg`/`recall_min`/`recall_max`/`recall_at`), per-rep latencies,
  rerank-time slice, and `mean_cmps`/`mean_hops`.
