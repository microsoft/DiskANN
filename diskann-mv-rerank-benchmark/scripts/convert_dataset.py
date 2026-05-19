"""
Convert single-vector / multi-vector embedding datasets into the binary formats
consumed by the DiskANN benchmark crates.

Two modes:

  analyze  Read inputs (if provided) and print structure/stats only. No writes.
  convert  Read inputs and write .fbin / .bin (groundtruth) / .mvbin outputs.

The script is dataset-agnostic — point it at any (.npy / .pt / qrels.tsv) combination.
For ArguANA-style datasets the standard invocation is::

    python convert_dataset.py analyze \\
        --single-vec-docs    multivector_data/arguana/singlevectors/documents/docs_chunk00000.npy \\
        --single-vec-queries multivector_data/arguana/singlevectors/queries/query_vech.npy \\
        --multi-vec-docs     "multivector_data/arguana/multivectors/documents/embeddings_part*.pt" \\
        --multi-vec-queries  "multivector_data/arguana/multivectors/queries/embeddings_part*.pt" \\
        --qrels              multivector_data/arguana/qrels/dev.tsv

    python convert_dataset.py convert \\
        ...same flags as above... \\
        --out-dir multivector_data/arguana/diskann \\
        --name    arguana

See the README.md next to this script for the binary layouts produced.
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
import sys
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Binary writers
# ---------------------------------------------------------------------------

def write_fbin_f32(out_path: str, mat: np.ndarray) -> None:
    """Write a 2D float32 matrix as a .fbin: u32 npoints | u32 ndims | row-major f32."""
    if mat.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {mat.shape}")
    arr = np.ascontiguousarray(mat, dtype=np.float32)
    npoints, ndims = arr.shape
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", npoints, ndims))
        f.write(arr.tobytes())


def write_groundtruth_range_bin(out_path: str, per_query_ids: List[List[int]]) -> None:
    """Write a variable-k groundtruth file.

    Layout (matches diskann's ``load_range_groundtruth``):
        u32 LE num_queries
        u32 LE total_results       (= sum of per-query sizes)
        num_queries * u32 LE sizes
        total_results * u32 LE ids (flat, in query order)

    There is no distances slab — the Rust loader doesn't read it for recall.
    Each query's relevant doc-id list may be empty (size 0).
    """
    num_queries = len(per_query_ids)
    sizes = np.array([len(row) for row in per_query_ids], dtype=np.uint32)
    total = int(sizes.sum())
    flat = np.empty(total, dtype=np.uint32)
    cursor = 0
    for row in per_query_ids:
        n = len(row)
        if n:
            flat[cursor : cursor + n] = np.asarray(row, dtype=np.uint32)
            cursor += n
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", num_queries, total))
        f.write(sizes.tobytes())
        f.write(flat.tobytes())


def write_mvbin_records(out_path: str, records: Iterable[np.ndarray]) -> Tuple[int, int]:
    """Write a sequence of per-doc multi-vector records as a .mvbin.

    Each record on disk: u32 K | u32 D | K*D f16 row-major. No top-level header.
    The dimension D must be constant across records.

    Returns (n_records_written, dim) for sanity reporting.
    """
    written = 0
    dim: Optional[int] = None
    with open(out_path, "wb") as f:
        for rec in records:
            if rec.ndim != 2:
                raise ValueError(f"record {written}: expected 2D (K, D), got shape {rec.shape}")
            k, d = rec.shape
            if k == 0:
                raise ValueError(f"record {written}: K=0 is not allowed by .mvbin format")
            if dim is None:
                dim = d
            elif d != dim:
                raise ValueError(f"record {written}: D={d} differs from earlier records D={dim}")
            rec16 = np.ascontiguousarray(rec, dtype=np.float16)
            f.write(struct.pack("<II", k, d))
            f.write(rec16.tobytes())
            written += 1
    if dim is None:
        dim = 0
    return written, dim


# ---------------------------------------------------------------------------
# .pt loading + per-doc record iteration
# ---------------------------------------------------------------------------

def _torch_load(path: str):
    """Import torch lazily so analyze-only flows without torch installed still work for npy/tsv."""
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "torch is required to read .pt files. Install with `pip install torch`."
        ) from e
    return torch.load(path, map_location="cpu", weights_only=False)


def _describe_pt(obj, depth: int = 0) -> List[str]:
    """Produce a few lines describing the structure of a loaded .pt object."""
    prefix = "  " * depth
    lines: List[str] = []
    try:
        import torch  # type: ignore
        is_tensor = isinstance(obj, torch.Tensor)
    except Exception:
        is_tensor = False
    if is_tensor:
        lines.append(f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        lines.append(f"{prefix}dict len={len(keys)} sample_keys={keys[:5]}")
        if keys:
            first = obj[keys[0]]
            lines.append(f"{prefix}  first value:")
            lines.extend(_describe_pt(first, depth + 2))
    elif isinstance(obj, (list, tuple)):
        lines.append(f"{prefix}{type(obj).__name__} len={len(obj)}")
        if obj:
            lines.append(f"{prefix}  [0]:")
            lines.extend(_describe_pt(obj[0], depth + 2))
    else:
        lines.append(f"{prefix}{type(obj).__name__}")
    return lines


def _iter_pt_records(obj, source: str) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield `(corpus_id, np.ndarray[K, D])` records from a loaded .pt object.

    Required shape: ``dict[int corpus_id, np.ndarray-or-Tensor of shape [K, D]]``.
    Anything else raises ``ValueError`` with the offending type / shape — there is
    no auto-detection. If your data is in a different layout (list-of-tensors,
    Tensor[N,K,D], etc.) preprocess it into the dict form before running the
    converter.
    """
    if not isinstance(obj, dict):
        raise ValueError(
            f"{source}: expected top-level dict[int, ndarray], got "
            f"{type(obj).__name__}. Preprocess your .pt into that form."
        )
    items = []
    for k, v in obj.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            raise ValueError(f"{source}: non-integer key {k!r} in .pt dict")
        items.append((ki, v))
    items.sort(key=lambda x: x[0])
    for ki, v in items:
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"{source}: key {ki} record has shape {arr.shape}, expected 2D")
        yield ki, arr


def _collect_pt_records(paths: List[str]) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (corpus_id, [K, D]) records across all .pt files, sorted within each file."""
    for p in paths:
        obj = _torch_load(p)
        yield from _iter_pt_records(obj, source=p)


# ---------------------------------------------------------------------------
# Qrels parsing
# ---------------------------------------------------------------------------

def parse_qrels(
    path: str, qid_col: int, doc_col: int, score_col: Optional[int] = None
) -> List[Tuple[int, int, float]]:
    """Parse a TSV qrels file. Returns a list of (qid, doc_id, score) tuples in file order.

    Every row must have ``qid_col``, ``doc_col``, and ``score_col``. Missing or
    non-numeric values trigger ``ValueError`` with ``path:lineno`` — no silent defaults.

    ``score_col`` defaults to the column right after the larger of qid_col/doc_col.
    """
    if score_col is None:
        score_col = max(qid_col, doc_col) + 1
    required_cols = max(qid_col, doc_col, score_col) + 1
    triples: List[Tuple[int, int, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < required_cols:
                raise ValueError(
                    f"{path}:{lineno}: row has {len(cols)} columns, need at least "
                    f"{required_cols} (qid_col={qid_col}, doc_col={doc_col}, score_col={score_col})"
                )
            try:
                qid = int(cols[qid_col])
                doc_id = int(cols[doc_col])
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: non-integer id in column: {e}")
            try:
                score = float(cols[score_col])
            except ValueError as e:
                raise ValueError(
                    f"{path}:{lineno}: non-numeric score in column {score_col} "
                    f"({cols[score_col]!r}): {e}"
                )
            triples.append((qid, doc_id, score))
    return triples


def dev_query_order(triples: List[Tuple[int, int, float]]) -> List[int]:
    """First-occurrence-ordered list of unique qids that have at least one positive judgment."""
    seen = set()
    order: List[int] = []
    for qid, _doc_id, score in triples:
        if score <= 0:
            continue
        if qid not in seen:
            seen.add(qid)
            order.append(qid)
    return order


def relevant_docs_by_qid(triples: List[Tuple[int, int, float]]) -> dict:
    """Map qid → ordered list of relevant doc-ids (score > 0), deduplicated."""
    out: dict = {}
    for qid, doc_id, score in triples:
        if score <= 0:
            continue
        bucket = out.setdefault(qid, [])
        if doc_id not in bucket:
            bucket.append(doc_id)
    return out


# ---------------------------------------------------------------------------
# Analyze mode
# ---------------------------------------------------------------------------

def analyze(args: argparse.Namespace) -> int:
    print("[analyze]")
    if args.single_vec_docs:
        for path in _expand_paths(args.single_vec_docs):
            arr = np.load(path, mmap_mode="r")
            print(
                f"  single-vec-docs  {path}: shape={arr.shape} dtype={arr.dtype} "
                f"size={os.path.getsize(path)}"
            )
            if arr.ndim == 2 and arr.shape[0] > 0:
                first = np.asarray(arr[0])
                print(
                    f"                   row0 min={float(first.min()):.4f} "
                    f"max={float(first.max()):.4f} mean={float(first.mean()):.4f}"
                )
    if args.single_vec_queries:
        for path in _expand_paths(args.single_vec_queries):
            arr = np.load(path, mmap_mode="r")
            print(
                f"  single-vec-queries  {path}: shape={arr.shape} dtype={arr.dtype} "
                f"size={os.path.getsize(path)}"
            )
    if args.multi_vec_docs:
        for path in _expand_paths(args.multi_vec_docs):
            print(f"  multi-vec-docs  {path}: size={os.path.getsize(path)}")
            obj = _torch_load(path)
            for line in _describe_pt(obj, depth=2):
                print(line)
    if args.multi_vec_queries:
        for path in _expand_paths(args.multi_vec_queries):
            print(f"  multi-vec-queries  {path}: size={os.path.getsize(path)}")
            obj = _torch_load(path)
            for line in _describe_pt(obj, depth=2):
                print(line)
    if args.qrels:
        triples = parse_qrels(args.qrels, args.query_id_col, args.doc_id_col)
        n = len(triples)
        qids = [t[0] for t in triples]
        docs = [t[1] for t in triples]
        positives = [t for t in triples if t[2] > 0]
        unique_qids = len(set(qids))
        eq = sum(1 for q, d, _ in triples if q == d)
        rel_by_qid = relevant_docs_by_qid(triples)
        per_q_sizes = [len(v) for v in rel_by_qid.values()]
        max_rel = max(per_q_sizes) if per_q_sizes else 0
        avg_rel = sum(per_q_sizes) / len(per_q_sizes) if per_q_sizes else 0.0
        print(
            f"  qrels  {args.qrels}: rows={n} positive_rows={len(positives)} "
            f"unique_qids={unique_qids} qids_with_relevant={len(rel_by_qid)} "
            f"qid_range=[{min(qids)}, {max(qids)}] doc_range=[{min(docs)}, {max(docs)}] "
            f"qid==doc_count={eq} max_relevant_per_query={max_rel} "
            f"avg_relevant_per_query={avg_rel:.2f}"
        )
    return 0


# ---------------------------------------------------------------------------
# Convert mode
# ---------------------------------------------------------------------------

_ALLOWED_NPY_DTYPES = {np.dtype("float16"), np.dtype("float32"), np.dtype("float64")}


def _check_npy_dtype(path: str, dtype: np.dtype) -> None:
    if dtype not in _ALLOWED_NPY_DTYPES:
        raise SystemExit(
            f"{path}: dtype {dtype} is not supported. Only float16/float32/float64 "
            "are allowed (integer dtypes would silently produce garbage after f32 cast)."
        )


def convert(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    name = args.name
    os.makedirs(out_dir, exist_ok=True)

    if args.qrels and (args.docids is None or args.query_ids is None):
        raise SystemExit(
            "--qrels requires both --docids and --query-ids. The previous fallback "
            "(row i = qrels-id i + base) silently produced wrong recall on datasets "
            "whose corpus id space has gaps; it has been removed."
        )

    docid_to_row: Optional[dict] = None
    qid_to_row: Optional[dict] = None
    if args.docids:
        docids_arr = np.load(args.docids)
        if docids_arr.ndim != 1:
            raise SystemExit(f"--docids must be a 1D array; got shape {docids_arr.shape}")
        docid_to_row = {int(d): i for i, d in enumerate(docids_arr.tolist())}
        if len(docid_to_row) != docids_arr.shape[0]:
            raise SystemExit(
                f"--docids contains {docids_arr.shape[0] - len(docid_to_row)} duplicate entries; "
                "each corpus id must appear at most once"
            )
        print(
            f"[convert] loaded docids sidecar {args.docids}: "
            f"{len(docid_to_row)} entries, range [{int(docids_arr.min())}, {int(docids_arr.max())}]"
        )
    if args.query_ids:
        with open(args.query_ids, "r", encoding="utf-8") as f:
            qid_lines = [line.strip() for line in f if line.strip()]
        try:
            qids_list = [int(s) for s in qid_lines]
        except ValueError as e:
            raise SystemExit(f"--query-ids must contain integers (one per line): {e}")
        qid_to_row = {q: i for i, q in enumerate(qids_list)}
        if len(qid_to_row) != len(qids_list):
            raise SystemExit(
                f"--query-ids contains {len(qids_list) - len(qid_to_row)} duplicate entries; "
                "each qid must appear at most once"
            )
        print(
            f"[convert] loaded query-ids sidecar {args.query_ids}: "
            f"{len(qid_to_row)} entries, range [{min(qids_list)}, {max(qids_list)}]"
        )

    dev_qids: Optional[List[int]] = None
    dev_relevant: Optional[dict] = None
    if args.qrels:
        triples = parse_qrels(args.qrels, args.query_id_col, args.doc_id_col)
        dev_qids = dev_query_order(triples)
        dev_relevant = relevant_docs_by_qid(triples)
        rel_counts = [len(dev_relevant[q]) for q in dev_qids]
        total_rel = sum(rel_counts)
        print(
            f"[convert] {len(dev_qids)} unique dev qids from {args.qrels}; "
            f"total positive judgments = {total_rel}; "
            f"relevant-per-query min/avg/max = "
            f"{min(rel_counts) if rel_counts else 0}/"
            f"{(total_rel / len(rel_counts) if rel_counts else 0.0):.2f}/"
            f"{max(rel_counts) if rel_counts else 0}"
        )
        zero_pos = [q for q, _doc, score in triples if score <= 0 and q not in dev_relevant]
        if zero_pos:
            print(
                f"[convert] WARNING: {len(zero_pos)} qids in {args.qrels} have only "
                f"non-positive judgments and will be skipped (first few: {zero_pos[:5]})"
            )

    def qid_row(q: int) -> int:
        assert qid_to_row is not None  # guaranteed by the --qrels arg-check above
        row = qid_to_row.get(q)
        if row is None:
            raise SystemExit(f"qrels qid {q} not present in --query-ids sidecar")
        return row

    def docid_row(d: int) -> int:
        assert docid_to_row is not None  # same
        row = docid_to_row.get(d)
        if row is None:
            raise SystemExit(f"qrels doc-id {d} not present in --docids sidecar")
        return row

    # 1. Single-vector docs → .fbin (f32).
    if args.single_vec_docs:
        srcs = _expand_paths(args.single_vec_docs)
        if len(srcs) != 1:
            raise SystemExit(
                f"--single-vec-docs expects exactly one .npy; got {len(srcs)} matches"
            )
        arr = np.load(srcs[0])
        _check_npy_dtype(srcs[0], arr.dtype)
        if arr.ndim != 2:
            raise SystemExit(f"--single-vec-docs: expected 2D array, got shape {arr.shape}")
        if docid_to_row is not None and arr.shape[0] != len(docid_to_row):
            raise SystemExit(
                f"--single-vec-docs row count {arr.shape[0]} does not match --docids length "
                f"{len(docid_to_row)}; the two must be aligned"
            )
        arr = arr.astype(np.float32, copy=False)
        out = os.path.join(out_dir, f"{name}_docs.fbin")
        write_fbin_f32(out, arr)
        print(f"[convert] wrote {out}  shape={arr.shape}  size={os.path.getsize(out)}")

    # 2. Single-vector queries → filtered .fbin (f32).
    if args.single_vec_queries:
        srcs = _expand_paths(args.single_vec_queries)
        if len(srcs) != 1:
            raise SystemExit(
                f"--single-vec-queries expects exactly one .npy; got {len(srcs)} matches"
            )
        all_q = np.load(srcs[0])
        _check_npy_dtype(srcs[0], all_q.dtype)
        if all_q.ndim != 2:
            raise SystemExit(f"--single-vec-queries: expected 2D array, got shape {all_q.shape}")
        if qid_to_row is not None and all_q.shape[0] != len(qid_to_row):
            raise SystemExit(
                f"--single-vec-queries row count {all_q.shape[0]} does not match --query-ids "
                f"length {len(qid_to_row)}; the two must be aligned"
            )
        if dev_qids is None:
            raise SystemExit("--single-vec-queries requires --qrels for dev-set filtering")
        rows = np.asarray([qid_row(q) for q in dev_qids], dtype=np.int64)
        if rows.max() >= all_q.shape[0]:
            raise SystemExit(
                f"qid {dev_qids[int(rows.argmax())]} maps to row {int(rows.max())} but only "
                f"{all_q.shape[0]} query rows exist; --query-ids / queries.npy are inconsistent"
            )
        sub = np.ascontiguousarray(all_q[rows], dtype=np.float32)
        out = os.path.join(out_dir, f"{name}_queries_dev.fbin")
        write_fbin_f32(out, sub)
        print(f"[convert] wrote {out}  shape={sub.shape}  size={os.path.getsize(out)}")

    # 3. Variable-k groundtruth from qrels → .bin (range-GT format).
    # DiskANN ids are 0-indexed = row index in the .fbin we write. If --docids was
    # supplied, each doc-id is translated via the sidecar (handles gaps in the underlying
    # corpus id space). Otherwise we fall back to the naive `doc_id - base` mapping.
    if args.qrels:
        assert dev_qids is not None and dev_relevant is not None
        per_query_ids: List[List[int]] = []
        for q in dev_qids:
            relevant = dev_relevant.get(q, [])
            per_query_ids.append([docid_row(d) for d in relevant])
        out = os.path.join(out_dir, f"{name}_gt.bin")
        write_groundtruth_range_bin(out, per_query_ids)
        sizes = [len(r) for r in per_query_ids]
        print(
            f"[convert] wrote {out}  queries={len(per_query_ids)} "
            f"total_relevant={sum(sizes)} "
            f"sizes_min/avg/max={min(sizes) if sizes else 0}/"
            f"{(sum(sizes)/len(sizes) if sizes else 0.0):.2f}/"
            f"{max(sizes) if sizes else 0}  size_bytes={os.path.getsize(out)}"
        )

    # 4. Multi-vector docs → .mvbin (f16). Emit records in the SAME row order as the
    # single-vector .fbin file: the order dictated by --docids.
    doc_mv_dim: Optional[int] = None
    if args.multi_vec_docs:
        if args.docids is None:
            raise SystemExit("--multi-vec-docs requires --docids for row-aligned emission")
        srcs = sorted(_expand_paths(args.multi_vec_docs))
        out = os.path.join(out_dir, f"{name}_docs.mvbin")
        docids_arr = np.load(args.docids)
        pt_records: dict = {q_id: rec for q_id, rec in _collect_pt_records(srcs)}
        missing = [int(d) for d in docids_arr.tolist() if int(d) not in pt_records]
        if missing:
            raise SystemExit(
                f"{len(missing)} doc ids in --docids are absent from --multi-vec-docs "
                f"(first few: {missing[:5]})"
            )
        records = (pt_records[int(d)] for d in docids_arr.tolist())
        n_written, doc_mv_dim = write_mvbin_records(out, records)
        print(
            f"[convert] wrote {out}  records={n_written} dim={doc_mv_dim} "
            f"size={os.path.getsize(out)} (ordered via --docids)"
        )

    # 5. Multi-vector queries (filtered to dev set) → .mvbin (f16).
    if args.multi_vec_queries:
        if dev_qids is None:
            raise SystemExit("--multi-vec-queries requires --qrels for dev-set filtering")
        srcs = sorted(_expand_paths(args.multi_vec_queries))
        dev_set = set(dev_qids)
        bucket: dict = {}
        for q_id, rec in _collect_pt_records(srcs):
            if q_id in dev_set and q_id not in bucket:
                bucket[q_id] = rec
        missing = [q for q in dev_qids if q not in bucket]
        if missing:
            raise SystemExit(
                f"missing {len(missing)} dev qids in multi-vec queries (first few: {missing[:5]})"
            )
        out = os.path.join(out_dir, f"{name}_queries_dev.mvbin")
        n_written, query_mv_dim = write_mvbin_records(out, (bucket[q] for q in dev_qids))
        if doc_mv_dim is not None and query_mv_dim != doc_mv_dim:
            raise SystemExit(
                f"multi-vec dim mismatch: docs.mvbin dim={doc_mv_dim} vs "
                f"queries.mvbin dim={query_mv_dim}. The reranker scores docs against "
                f"queries; the per-vector dimensions must match."
            )
        print(
            f"[convert] wrote {out}  records={n_written} dim={query_mv_dim} "
            f"size={os.path.getsize(out)}"
        )

    return 0


# ---------------------------------------------------------------------------
# Path helpers + arg parsing
# ---------------------------------------------------------------------------

def _expand_paths(pattern: str) -> List[str]:
    """Expand a glob and return sorted matches; if no glob meta-chars, just return [pattern]."""
    if any(c in pattern for c in "*?[]"):
        return sorted(glob.glob(pattern))
    return [pattern]


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--single-vec-docs", help="path to .npy with shape (N_docs, D_sv); dtype f16/f32/f64")
    p.add_argument("--single-vec-queries", help="path to .npy with shape (N_q, D_sv); dtype f16/f32/f64")
    p.add_argument(
        "--multi-vec-docs",
        help=(
            "glob for .pt files containing multi-vector docs. Each .pt must be "
            "torch.load-able to a dict[int corpus_id -> np.ndarray of shape [K, D_mv]]."
        ),
    )
    p.add_argument(
        "--multi-vec-queries",
        help="glob for .pt files containing multi-vector queries (same dict shape as --multi-vec-docs)",
    )
    p.add_argument(
        "--qrels",
        help=(
            "path to qrels TSV (cols: qid, doc_id, score). When supplied, --docids and "
            "--query-ids are also required (no fallback heuristic)."
        ),
    )
    p.add_argument("--query-id-col", type=int, default=0)
    p.add_argument("--doc-id-col", type=int, default=1)
    p.add_argument(
        "--docids",
        help=(
            "Path to a docids_chunk*.npy sidecar: int64 array of length N_docs. Element i is "
            "the corpus id of doc row i. Required whenever --qrels is supplied."
        ),
    )
    p.add_argument(
        "--query-ids",
        help=(
            "Path to a query_ids.txt sidecar: one integer per line, count equal to N_q. "
            "Line i is the qid of query row i. Required whenever --qrels is supplied."
        ),
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset converter for DiskANN MV benchmarks.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_analyze = sub.add_parser("analyze", help="print structure/stats only; no writes")
    _add_common_args(p_analyze)

    p_convert = sub.add_parser("convert", help="produce DiskANN binary files")
    _add_common_args(p_convert)
    p_convert.add_argument("--out-dir", required=True, help="output directory")
    p_convert.add_argument("--name", required=True, help="filename prefix (e.g. 'arguana')")

    args = parser.parse_args(argv)
    if args.mode == "analyze":
        return analyze(args)
    if args.mode == "convert":
        return convert(args)
    parser.error(f"unknown mode {args.mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
