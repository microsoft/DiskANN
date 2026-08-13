# Stage 1 — Dataset preparation

Corpora live outside the repository, one directory per dataset under a root of your
choosing — `<data-root>` throughout these notes:

```
enron-email-1M-fbv4   enron-email-1M-fbv8   MERB-Corpus-12K   msmarco-fbv8rc1
caselaw-1M            glove-100-angular     glove-200-angular
lotte-all-forum       mteb-nq-full
```

Each holds the source download, the converted binaries, the groundtruth, and a `_tmp/`
subdirectory of saved indexes. The tooling that produced them is tracked in the repo (see
below) — only data lives here.

## Binary formats

Both are little-endian with an 8-byte header.

**f32/fp16** — full-precision inputs used for exact groundtruth computation.

| Offset | Type | Meaning |
|---|---|---|
| 0 | `u32` | `npts` |
| 4 | `u32` | `dim` |
| 8 | `f32` or `f16 × npts × dim` | row-major vectors |

**minmax8** — the quantized form the index is built and searched on.

| Offset | Type | Meaning |
|---|---|---|
| 0 | `u32` | `npts` |
| 4 | `u32` | `row_bytes` |
| 8 | `u8 × npts × row_bytes` | row-major quantized records |

`row_bytes` is the record width, **not** the dimension: it is `dim + 20` for every corpus
checked (1536→1556, 384→404, 256→276, 200→220). Read it from the header rather than
deriving it, and note that some older workbook entries label the record width as "dim".

Corpus byte size for the scan-percentage axis is `npts * row_bytes`, i.e. the file size
minus the 8-byte header.

## The prep tool

All preparation goes through one tracked, dataset-agnostic CLI:
[`diskann-graphivf/scripts/dataprep/vecprep.py`](../../../../diskann-graphivf/scripts/dataprep/vecprep.py).
It needs only `numpy`. Nothing in it knows the name of any dataset — **do not fork it per
dataset**; if a new corpus needs behaviour it lacks, add a flag.

| Subcommand | Purpose |
|---|---|
| `info` | Print header, element type, and file-size arithmetic for any `.bin` |
| `convert` | `.npy` / `.bin` → `fp16` or `f32`, streamed in chunks |
| `normalize` | L2-normalize rows in place-safe fashion, reporting any zero rows |
| `subset` | Sample rows (`--mode even\|head\|random`) into a smaller file |
| `filter` | Drop rows (`--drop zero`) and emit the surviving-index map |
| `groundtruth` | Exact top-K, **audited before it will write** |
| `audit` | Re-run the groundtruth checks on files that already exist |

The element type is *not* stored in the header; every subcommand recovers it from the
file-size arithmetic, which is why none of them take a `--dtype` flag. The four payload
widths in play (u8=1, fp16=2, f32=4, groundtruth=8) are distinct, so the inference is
unambiguous. `info` shows what it inferred — run it first when a file is unfamiliar.

Quantization to `minmax8` is the one step that stays in Rust, because the index must be
built on bit-identical records:

```text
cargo run --release --example compress_minmax -- corpus_fp16.bin  corpus_minmax8.bin  fp16
cargo run --release --example compress_minmax -- queries_fp16.bin queries_minmax8.bin fp16
```

Run it on **both** corpus and queries, and always subset/filter *before* quantizing —
`vecprep` slices quantized rows verbatim, but re-quantizing a subset would recompute
per-row scales and silently change the records.

Conventional names: `corpus_fp16.bin`, `corpus_minmax8.bin`, `queries_full_minmax8.bin`,
`queries_sample<N>_minmax8.bin`, `groundtruth_recall_1000_query_<N>.bin`.

## Procedure

1. **Acquire** the embeddings and any manifest. Read the manifest: it usually states the
   encoder, dimensionality, dtype, whether vectors are unit-norm, and — critically — any
   count of degenerate or empty documents.
2. **Convert** with `vecprep.py convert`, which streams in chunks via `np.memmap`. A
   2.7M × 1536 fp16 corpus is ~8 GB; do not load it whole.
3. **Filter** degenerate rows *now*, with `vecprep.py filter --drop zero`, not after the
   groundtruth is computed — see [stage 2](./02-groundtruth.md). This is the single most
   expensive mistake in the pipeline to discover late.
4. **Subsample queries** to ~1000–2000 with `vecprep.py subset --mode even`. The sweeps are
   single-threaded and a full query set makes each sweep point cost minutes for no extra
   signal. Even spacing avoids bias from any ordering in the source; `--mode head` would
   inherit it.
5. **Verify sizes** with `vecprep.py info`: `8 + npts * row_bytes` must equal the file size
   exactly. A mismatch means the header and body disagree and everything downstream is
   garbage.

## Normalization

For online graph-IVF, `normalize: true` normalizes **warmup and split centroids only**. It
does not rewrite corpus or query rows; online rows are stored verbatim. If cosine semantics
are needed, normalize corpus and queries during preparation, then use
`distance: cosine_normalized`. Use `normalize: false` for ordinary squared-L2 datasets such
as MSTuring.

Remove zero rows before normalization. For a genuinely unit-norm corpus, squared L2 and
inner product produce the same ordering, but record the metric actually used rather than
assuming every embedding dataset is normalized.
