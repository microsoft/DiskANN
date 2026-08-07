# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dataset-agnostic preparation for DiskANN vector benchmarks.

Every stage between "a pile of embeddings" and "a corpus, a query set and a
groundtruth the benchmark runner can consume" lives here, as subcommands of one
tool. Nothing in this file knows the name of any dataset.

    python vecprep.py <command> --help

    info         report the layout of a .bin file
    convert      .npy embeddings -> DiskANN .bin
    normalize    L2-normalize rows onto the unit sphere
    subset       carve an aligned subset out of several parallel files
    filter       drop degenerate rows from several parallel files
    groundtruth  exact top-K by brute force, with an automatic audit
    audit        check an existing corpus/groundtruth pair for degeneracy

`compress_minmax` (the Rust example beside this file) does the 8-bit quantization
step; it is not reimplemented here.

FILE FORMATS
------------
All vector files share an 8-byte header of two little-endian u32s, followed by a
row-major payload:

    [npts u32][width u32][payload]

`width` is the row width in *elements* for float files and in *bytes* for
quantized files. The element type is not recorded, so it is recovered from the
file size -- see `sniff()`. Groundtruth files reuse the same header with
`width = K`, and carry two payloads: u32 ids then f32 distances.
"""

import argparse
import json
import sys

import numpy as np

# Rows per streaming block. Large enough to amortize I/O, small enough that a
# block of f32-widened fp16 vectors stays comfortably in cache-friendly memory.
CHUNK = 32768


# --------------------------------------------------------------------------
# Format handling
# --------------------------------------------------------------------------

# Payload bytes per element, keyed by the name used on the command line.
DTYPES = {
    "u8": np.uint8,
    "fp16": np.float16,
    "f32": np.float32,
}


class Spec:
    """The resolved layout of a `.bin` file: header plus recovered element type."""

    def __init__(self, path, npts, width, dtype, is_groundtruth=False):
        self.path = path
        self.npts = npts
        self.width = width
        self.dtype = dtype
        self.is_groundtruth = is_groundtruth

    @property
    def itemsize(self):
        return np.dtype(self.dtype).itemsize

    def __str__(self):
        if self.is_groundtruth:
            return f"{self.npts} x {self.width} groundtruth (u32 ids + f32 dists)"
        return f"{self.npts} x {self.width} {np.dtype(self.dtype).name}"


def read_header(path):
    with open(path, "rb") as fh:
        head = fh.read(8)
    if len(head) < 8:
        raise ValueError(f"{path}: shorter than an 8-byte header")
    npts, width = np.frombuffer(head, dtype="<u4")
    return int(npts), int(width)


def sniff(path):
    """Recover a file's layout from its header and its size on disk.

    The element type is not stored in the file, but the payload size is
    `npts * width * itemsize`, and the four cases we care about have distinct
    item sizes (u8=1, fp16=2, f32=4, groundtruth=8 for the id+distance pair).
    Given `npts` and `width` from the header, the file size therefore identifies
    the layout unambiguously -- which is what lets every subcommand below accept
    any file without a `--dtype` flag.
    """
    import os

    npts, width = read_header(path)
    payload = os.path.getsize(path) - 8
    if npts == 0 or width == 0:
        raise ValueError(f"{path}: degenerate header {npts} x {width}")
    if payload % (npts * width) != 0:
        raise ValueError(
            f"{path}: payload {payload} is not a multiple of "
            f"npts*width = {npts * width}; header and body disagree"
        )
    per_element = payload // (npts * width)
    if per_element == 8:
        return Spec(path, npts, width, np.uint32, is_groundtruth=True)
    for dtype in (np.uint8, np.float16, np.float32):
        if np.dtype(dtype).itemsize == per_element:
            return Spec(path, npts, width, dtype)
    raise ValueError(f"{path}: {per_element} bytes/element matches no known layout")


def open_matrix(spec):
    """Memory-map the payload of a non-groundtruth file."""
    if spec.is_groundtruth:
        raise ValueError(f"{spec.path}: expected a vector file, found groundtruth")
    return np.memmap(spec.path, dtype=spec.dtype, mode="r", offset=8,
                     shape=(spec.npts, spec.width))


def read_groundtruth(path):
    spec = sniff(path)
    if not spec.is_groundtruth:
        raise ValueError(f"{path}: not a groundtruth file ({spec})")
    n, k = spec.npts, spec.width
    with open(path, "rb") as fh:
        fh.seek(8)
        ids = np.fromfile(fh, dtype="<u4", count=n * k).reshape(n, k)
        dists = np.fromfile(fh, dtype="<f4", count=n * k).reshape(n, k)
    return ids, dists


def write_groundtruth(path, ids, dists):
    with open(path, "wb") as fh:
        fh.write(np.asarray(ids.shape, dtype="<u4").tobytes())
        fh.write(np.ascontiguousarray(ids, dtype="<u4").tobytes())
        fh.write(np.ascontiguousarray(dists, dtype="<f4").tobytes())
    print(f"  wrote {path}  {ids.shape[0]} x {ids.shape[1]}", flush=True)


def parse_pairs(args):
    """Parse `src=dst` arguments into a list of tuples."""
    pairs = []
    for arg in args:
        if "=" not in arg:
            raise SystemExit(f"expected src=dst, got {arg!r}")
        src, dst = arg.split("=", 1)
        pairs.append((src, dst))
    return pairs


def take_rows(spec, index, dst):
    """Write `spec`'s rows at `index` to `dst`, preserving the payload verbatim.

    Slicing the already-written file rather than re-deriving from source keeps
    surviving rows bit-identical, which matters for quantized files: re-running
    the quantizer over a subset would pick different scales.
    """
    if spec.is_groundtruth:
        ids, dists = read_groundtruth(spec.path)
        write_groundtruth(dst, ids[index], dists[index])
        return

    src = open_matrix(spec)
    with open(dst, "wb") as fh:
        fh.write(np.asarray([len(index), spec.width], dtype="<u4").tobytes())
        for start in range(0, len(index), CHUNK):
            block = index[start:start + CHUNK]
            fh.write(np.ascontiguousarray(src[block]).tobytes())
    print(f"  wrote {dst}  {len(index)} x {spec.width} "
          f"{np.dtype(spec.dtype).name}", flush=True)


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------

def cmd_info(args):
    for path in args.files:
        spec = sniff(path)
        print(f"{path}\n  {spec}")
        if spec.is_groundtruth:
            _, dists = read_groundtruth(path)
            print(f"  distance range: {dists.min():.4f} .. {dists.max():.4f}")
            print(f"  monotone per row: {bool((np.diff(dists, axis=1) >= -1e-6).all())}")
        elif spec.dtype != np.uint8:
            mm = open_matrix(spec)
            sample = np.asarray(mm[::max(1, spec.npts // 10000)], dtype=np.float32)
            norms = np.linalg.norm(sample, axis=1)
            print(f"  sampled norms: min={norms.min():.6f} max={norms.max():.6f}")
            print(f"  sampled zero rows: {int((norms == 0).sum())}")
    return 0


# --------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------

def cmd_convert(args):
    """`.npy` -> `.bin`, streamed so a corpus larger than RAM converts cleanly."""
    arr = np.load(args.src, mmap_mode="r")
    if arr.ndim != 2:
        raise SystemExit(f"{args.src}: expected 2-D, got shape {arr.shape}")

    out_dtype = DTYPES[args.dtype] if args.dtype else arr.dtype
    if out_dtype not in (np.float16, np.float32):
        raise SystemExit(f"convert writes fp16 or f32, not {args.dtype}")

    npts, dim = arr.shape
    with open(args.dst, "wb") as fh:
        fh.write(np.asarray([npts, dim], dtype="<u4").tobytes())
        for start in range(0, npts, CHUNK):
            block = np.ascontiguousarray(arr[start:start + CHUNK], dtype=out_dtype)
            fh.write(block.tobytes())
    print(f"{args.src} -> {args.dst}: {npts} x {dim} "
          f"{np.dtype(out_dtype).name}", flush=True)
    return 0


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

def cmd_normalize(args):
    """L2-normalize rows so squared L2 and cosine rank identically downstream.

    Angular datasets (GloVe and friends) ship unnormalized, and the online build
    stores rows verbatim -- it has no opportunity to normalize the corpus itself.
    """
    spec = sniff(args.src)
    if spec.dtype == np.uint8:
        raise SystemExit(f"{args.src}: cannot normalize a quantized file")
    src = open_matrix(spec)

    zeros = 0
    with open(args.dst, "wb") as fh:
        fh.write(np.asarray([spec.npts, spec.width], dtype="<u4").tobytes())
        for start in range(0, spec.npts, CHUNK):
            block = np.asarray(src[start:start + CHUNK], dtype=np.float32)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            # A zero row has no direction to preserve. Leaving it at the origin
            # is the only defensible choice, and it must not become NaN --
            # but it is also a bug upstream, so it is counted and reported.
            zeros += int((norms == 0).sum())
            norms[norms == 0] = 1.0
            out = (block / norms).astype(spec.dtype)
            fh.write(np.ascontiguousarray(out).tobytes())

    print(f"{args.src} -> {args.dst}: {spec.npts} x {spec.width}")
    if zeros:
        print(f"  WARNING: {zeros} zero rows passed through unnormalized; "
              f"run `vecprep filter --drop zero` before building")
    return 0


# --------------------------------------------------------------------------
# subset
# --------------------------------------------------------------------------

def subset_index(total, count, mode, seed):
    if count > total:
        raise SystemExit(f"cannot take {count} rows from {total}")
    if mode == "even":
        # Evenly spaced rather than the first N, so the sample is not biased by
        # any ordering in the source.
        idx = np.linspace(0, total - 1, count).round().astype(np.int64)
        if len(np.unique(idx)) != count:
            raise SystemExit(f"even sampling of {count} from {total} collided")
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False))


def cmd_subset(args):
    """Slice the same rows out of several parallel files.

    Query vectors, their quantized twin and their groundtruth must stay aligned,
    so they are sliced together with one index rather than by three scripts that
    each recompute it.
    """
    pairs = parse_pairs(args.pairs)
    specs = [sniff(src) for src, _ in pairs]
    totals = {s.npts for s in specs}
    if len(totals) != 1:
        raise SystemExit(f"inputs disagree on row count: "
                         + ", ".join(f"{s.path}={s.npts}" for s in specs))

    total = totals.pop()
    idx = subset_index(total, args.count, args.mode, args.seed)
    print(f"subset {args.count} of {total} rows ({args.mode})", flush=True)
    for spec, (_, dst) in zip(specs, pairs):
        take_rows(spec, idx, dst)

    if args.index_out:
        json.dump(idx.tolist(), open(args.index_out, "w"))
        print(f"  wrote {args.index_out}", flush=True)
    return 0


# --------------------------------------------------------------------------
# filter
# --------------------------------------------------------------------------

def zero_row_mask(spec):
    """Boolean mask of rows that are entirely zero."""
    src = open_matrix(spec)
    keep = np.ones(spec.npts, dtype=bool)
    for start in range(0, spec.npts, CHUNK):
        stop = min(start + CHUNK, spec.npts)
        block = np.asarray(src[start:stop])
        keep[start:stop] = block.any(axis=1)
    return keep


def cmd_filter(args):
    """Drop degenerate rows from a corpus and everything derived from it.

    Zero rows are not a cosmetic problem. Under squared L2 a zero vector sits at
    distance ||q||^2 from *every* query, which for a unit-norm corpus is exactly
    1.0 -- inside the top-K band. They therefore occupy real groundtruth slots
    and cap achievable recall at a value no amount of search effort can beat.
    See `audit` for the diagnostic.
    """
    pairs = parse_pairs(args.pairs)
    ref = sniff(args.mask_from)
    if ref.dtype == np.uint8:
        raise SystemExit(f"{args.mask_from}: derive the mask from the float file, "
                         f"not the quantized one")

    keep = zero_row_mask(ref)
    dropped = int((~keep).sum())
    print(f"{args.mask_from}: {ref.npts} rows, dropping {dropped}, "
          f"keeping {int(keep.sum())}", flush=True)
    if dropped == 0:
        print("  nothing to do")
        return 0

    idx = np.flatnonzero(keep)
    for src, dst in pairs:
        spec = sniff(src)
        if spec.npts != ref.npts:
            raise SystemExit(f"{src}: {spec.npts} rows, expected {ref.npts}")
        take_rows(spec, idx, dst)

    if args.ids:
        ids = json.load(open(args.ids))
        if len(ids) != ref.npts:
            raise SystemExit(f"{args.ids}: {len(ids)} ids, expected {ref.npts}")
        kept = [ids[i] for i in idx]
        json.dump(kept, open(args.ids_out, "w"))
        print(f"  wrote {args.ids_out}  {len(kept)} surviving ids", flush=True)

    print("\nGroundtruth must now be recomputed: ids refer to row positions, "
          "which have shifted.")
    return 0


# --------------------------------------------------------------------------
# groundtruth
# --------------------------------------------------------------------------

def exact_topk(corpus_spec, queries, k, metric, chunk):
    """Brute-force top-K over a streamed corpus.

    A running top-K is merged against each block via `argpartition`, so peak
    memory is one block plus the K-wide frontier, independent of corpus size.
    """
    nq = queries.shape[0]
    rows = np.arange(nq)[:, None]
    qsq = (queries * queries).sum(axis=1)[:, None]
    src = open_matrix(corpus_spec)

    best_dist = np.full((nq, k), np.inf, dtype=np.float32)
    best_ids = np.zeros((nq, k), dtype=np.int64)

    for start in range(0, corpus_spec.npts, chunk):
        stop = min(start + chunk, corpus_spec.npts)
        block = np.asarray(src[start:stop], dtype=np.float32)
        inner = queries @ block.T
        if metric == "squared_l2":
            bsq = (block * block).sum(axis=1)
            dist = qsq + bsq[None, :] - 2.0 * inner
        else:
            # Maximum inner product: negate so "smaller is better" holds and the
            # same partition/sort path serves both metrics.
            dist = -inner
        ids = np.arange(start, stop, dtype=np.int64)

        merged_dist = np.concatenate([best_dist, dist], axis=1)
        merged_ids = np.concatenate(
            [best_ids, np.broadcast_to(ids, (nq, stop - start))], axis=1)
        part = np.argpartition(merged_dist, k - 1, axis=1)[:, :k]
        best_dist = merged_dist[rows, part]
        best_ids = merged_ids[rows, part]

        if (start // chunk) % 10 == 0:
            pct = 100.0 * stop / corpus_spec.npts
            print(f"  [{start}:{stop}]  {pct:5.1f}%", flush=True)

    order = np.argsort(best_dist, axis=1)
    return best_ids[rows, order].astype(np.uint32), best_dist[rows, order]


def cmd_groundtruth(args):
    corpus = sniff(args.corpus)
    qspec = sniff(args.queries)
    if corpus.width != qspec.width:
        raise SystemExit(f"dim mismatch: corpus {corpus.width}, "
                         f"queries {qspec.width}")
    if corpus.dtype == np.uint8 or qspec.dtype == np.uint8:
        raise SystemExit("compute groundtruth from the float files; quantized "
                         "input would fold quantization error into the labels")
    if args.k > corpus.npts:
        raise SystemExit(f"k={args.k} exceeds corpus size {corpus.npts}")

    print(f"corpus {corpus}\nqueries {qspec}\nmetric {args.metric}, k={args.k}",
          flush=True)
    queries = np.asarray(open_matrix(qspec), dtype=np.float32)
    ids, dists = exact_topk(corpus, queries, args.k, args.metric, args.chunk)
    write_groundtruth(args.out, ids, dists)

    print()
    ok = report_audit(corpus, ids, dists, args.metric)
    if not ok and not args.allow_degenerate:
        print("\nFAILED: groundtruth contains degenerate rows. Fix the corpus "
              "(`vecprep filter --drop zero`) and recompute, or pass "
              "--allow-degenerate to keep it anyway.", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

def report_audit(corpus_spec, ids, dists, metric):
    """Report groundtruth health. Returns False if the labels are untrustworthy.

    Checks, in order of how expensive the mistake is:
      1. degenerate corpus rows appearing inside the groundtruth
      2. distances sorted within each row
      3. ids in range
    """
    print("=== groundtruth audit ===")
    ok = True

    keep = zero_row_mask(corpus_spec)
    zero_ids = np.flatnonzero(~keep)
    print(f"degenerate (all-zero) corpus rows: {zero_ids.size}")

    if zero_ids.size:
        hits = np.isin(ids, zero_ids.astype(ids.dtype)).sum(axis=1)
        k = ids.shape[1]
        print(f"  per query, degenerate rows inside top-{k}: "
              f"mean {hits.mean():.1f}, median {int(np.median(hits))}, "
              f"min {hits.min()}, max {hits.max()}")
        if hits.max() > 0:
            ok = False
            ceiling = 100.0 * (1.0 - hits.mean() / k)
            print(f"  -> recall@{k} is capped near {ceiling:.1f}% regardless of "
                  f"search effort")
            if metric == "squared_l2":
                print("  -> a zero row sits at ||q||^2 from every query, which "
                      "for a unit-norm corpus is exactly 1.0")

    monotone = bool((np.diff(dists, axis=1) >= -1e-6).all())
    print(f"distances sorted within each row: {monotone}")
    if not monotone:
        ok = False

    in_range = int(ids.max()) < corpus_spec.npts
    print(f"ids within corpus [0, {corpus_spec.npts}): {in_range}")
    if not in_range:
        ok = False

    print(f"top-1 distance range:  {dists[:, 0].min():.4f} .. {dists[:, 0].max():.4f}")
    print(f"top-{ids.shape[1]} distance range: "
          f"{dists[:, -1].min():.4f} .. {dists[:, -1].max():.4f}")
    print(f"verdict: {'OK' if ok else 'DEGENERATE'}")
    return ok


def cmd_audit(args):
    corpus = sniff(args.corpus)
    ids, dists = read_groundtruth(args.groundtruth)
    ok = report_audit(corpus, ids, dists, args.metric)
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vecprep", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="report the layout of .bin files")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("convert", help=".npy embeddings -> DiskANN .bin")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--dtype", choices=("fp16", "f32"),
                   help="output element type (default: keep the source's)")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("normalize", help="L2-normalize rows onto the unit sphere")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("subset", help="slice aligned rows out of parallel files")
    p.add_argument("pairs", nargs="+", metavar="src=dst")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--mode", choices=("even", "random"), default="even")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--index-out", help="write the chosen row indices as JSON")
    p.set_defaults(func=cmd_subset)

    p = sub.add_parser("filter", help="drop degenerate rows from parallel files")
    p.add_argument("pairs", nargs="+", metavar="src=dst")
    p.add_argument("--mask-from", required=True,
                   help="float file the keep-mask is derived from")
    p.add_argument("--drop", choices=("zero",), default="zero")
    p.add_argument("--ids", help="JSON list of external ids, one per input row")
    p.add_argument("--ids-out", help="where to write the surviving ids")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("groundtruth", help="exact top-K by brute force")
    p.add_argument("--corpus", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("-k", type=int, default=1000)
    p.add_argument("--metric", choices=("squared_l2", "mips"), default="squared_l2")
    p.add_argument("--chunk", type=int, default=CHUNK)
    p.add_argument("--allow-degenerate", action="store_true",
                   help="write the file even if the audit fails")
    p.set_defaults(func=cmd_groundtruth)

    p = sub.add_parser("audit", help="check a corpus/groundtruth pair")
    p.add_argument("--corpus", required=True)
    p.add_argument("--groundtruth", required=True)
    p.add_argument("--metric", choices=("squared_l2", "mips"), default="squared_l2")
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    if args.command == "filter" and args.ids and not args.ids_out:
        parser.error("--ids requires --ids-out")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
