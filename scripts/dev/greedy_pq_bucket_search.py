#!/usr/bin/env python3
"""
Greedy search over per-bucket PQ byte allocations to maximize recall of a quantized dataset.

Problem Setting
---------------
We have a 3072-dimensional float32 base dataset (binary format: uint32 npts, uint32 dim, then npts*dim floats)
plus a queries file of the same float32 binary format, and a raw ground-truth (GT) truthset file produced by
`compute_groundtruth` over the original uncompressed base vectors.

We partition the 3072 dimensions into 8 contiguous buckets of 384 dims each.
Within each bucket we allocate a number of PQ chunks; each chunk contributes 1 byte (because 256 centers) to the
compressed representation. Total bytes per vector = sum(chunks_per_bucket).

Using variable-chunk PQ (generate_pq_variable_chunks) we construct chunk offsets spanning all 3072 dims by further
subdividing each 384-d bucket according to its assigned number of chunks (allowing non-uniform chunk sizes when
384 is not divisible by the chunk count — remainders distributed +1 to first R chunks).

Search Strategy
---------------
Start from an initial uniform allocation (default: 8 chunks per bucket => 64 bytes / vector). At each iteration,
consider (up to) 8 candidate configurations formed by adding a fixed increment (default: +8 chunks) to exactly one
bucket (i.e. local move). Train & compress each candidate, compute quantized ground truth and recall@K vs raw GT,
select the candidate with highest recall improvement. Repeat until no candidate improves recall or a stopping
condition (max iterations / per-bucket cap / total byte cap) is met.

External Tools (must already be built and available):
  generate_pq_variable_chunks
  compute_groundtruth
  calculate_recall

These are invoked via subprocess. The script does not recompile anything.

Output
------
Prints a table of iterations with allocation vector and recall.
Writes (optional) JSON log of the search trajectory.
Retains artifacts (pq pivots/compressed, quantized GT) for the best configuration (and optionally all candidates).

Example Usage
-------------
python greedy_pq_bucket_search.py \
  --base_file /mnt/ravi/openai_rnd100k_data.bin \
  --query_file /mnt/ravi/openai_query.fbin \
  --raw_gt_file /mnt/ravi/openai_rnd100k_data.bin_gt100 \
  --work_dir /mnt/ravi/pq_search_runs/run1 \
  --tools_dir /home/rakri/DiskANN/build/apps/utils \
  --k 100 --sampling_rate 0.1 --initial_chunks 8 --increment 8 --max_per_bucket 64

Notes
-----
1. This can be computationally expensive; consider lowering sampling_rate during prototyping.
2. Ensure the build was compiled with SAVE_INFLATED_PQ if you require the inflated file _pq_compressed.bin_inflated.bin.
3. You can adjust --inflate_suffix if your build uses a different naming scheme.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# DIM and bucket layout are now inferred at runtime from the base/query files; these
# previous constants are kept only as defaults/documentation and are NOT used after
# dynamic inference in main().
DEFAULT_NUM_BUCKETS = 8
DEFAULT_DIM_PLACEHOLDER = -1  # Will be replaced after reading headers
RECALL_REGEX = re.compile(r"Avg\. recall@\d+ is ([0-9]*\.?[0-9]+)")

@dataclasses.dataclass
class Config:
    base_file: Path
    query_file: Path
    raw_gt_file: Path
    work_dir: Path
    tools_dir: Path
    # Derived at runtime
    dim: int = DEFAULT_DIM_PLACEHOLDER
    num_buckets: int = DEFAULT_NUM_BUCKETS
    bucket_sizes: List[int] = dataclasses.field(default_factory=list)  # length = num_buckets (sum == dim)
    k: int = 100
    sampling_rate: float = 0.1
    initial_chunks: int = 8
    increment: int = 8
    max_iters: int = 20
    max_per_bucket: int = 128
    max_total_bytes: int | None = None
    keep_all: bool = False
    sleep_wait: float = 0.5
    timeout_train: int = 0  # 0 => no extra timeout beyond process
    timeout_gt: int = 0
    timeout_recall: int = 0
    pq_prefix_base: str = "pq_cfg"
    inflate_suffix: str = "_pq_compressed.bin_inflated.bin"
    log_json: Path | None = None
    verbose: bool = True

@dataclasses.dataclass
class IterRecord:
    iteration: int
    allocation: List[int]
    recall: float
    bytes_per_vec: int
    prefix: str
    improved: bool
    tag: str
    artifacts: List[Path] = dataclasses.field(default_factory=list)

class CommandError(RuntimeError):
    pass

def run_cmd(cmd: List[str], timeout: int = 0, verbose: bool = True) -> str:
    if verbose:
        print("[CMD]", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout or None, check=False)
    except subprocess.TimeoutExpired:
        raise CommandError(f"Timeout running command: {' '.join(cmd)}")
    if verbose:
        print(proc.stdout)
    if proc.returncode != 0:
        raise CommandError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nOutput:\n{proc.stdout}")
    return proc.stdout

def build_chunk_offsets(cfg: Config, allocation: List[int]) -> List[int]:
    """Generate cumulative chunk offsets across the full dimensionality.

    For each bucket i with dimensional span bucket_sizes[i], subdivide that span into
    allocation[i] contiguous chunks. If the bucket dimension is not divisible by the chunk
    count, remainder dims are distributed one each to the first R chunks (greedy balancing).

    Returns: list[int] of monotonically increasing offsets of length (total_chunks + 1),
    where offsets[0] == 0 and offsets[-1] == cfg.dim.
    """
    if not cfg.bucket_sizes:
        raise ValueError("Config bucket_sizes not initialized.")
    if len(allocation) != len(cfg.bucket_sizes):
        raise ValueError("Allocation length does not match number of buckets")
    offsets: List[int] = [0]
    cur = 0
    for bucket_dim, c in zip(cfg.bucket_sizes, allocation):
        if c <= 0:
            raise ValueError("Chunk count must be positive per bucket")
        base = bucket_dim // c
        rem = bucket_dim % c
        for i in range(c):
            size = base + (1 if i < rem else 0)
            cur += size
            offsets.append(cur)
    assert cur == cfg.dim, f"Final offset {cur} != inferred dim {cfg.dim}"
    return offsets

def write_offsets_file(offsets: List[int], path: Path):
    with open(path, 'w') as f:
        f.write(" ".join(str(x) for x in offsets))
        f.write("\n")


def train_and_quantize(cfg: Config, allocation: List[int], tag: str) -> Tuple[Path, Path, Path, Path]:
    """Run PQ training + compression for a given allocation and return
    (prefix_path, compressed_file, inflated_file, offsets_file)."""
    offsets = build_chunk_offsets(cfg, allocation)
    offsets_file = cfg.work_dir / f"offsets_{tag}.txt"
    write_offsets_file(offsets, offsets_file)
    prefix_path = cfg.work_dir / f"{cfg.pq_prefix_base}_{tag}"
    gen_tool = cfg.tools_dir / "generate_pq_variable_chunks"
    cmd = [str(gen_tool), "float", str(cfg.base_file), str(prefix_path), str(offsets_file), str(cfg.sampling_rate)]
    run_cmd(cmd, timeout=cfg.timeout_train, verbose=cfg.verbose)
    compressed = Path(str(prefix_path) + "_pq_compressed.bin")
    inflated = Path(str(compressed) + "_inflated.bin")  # naming from pq.cpp
    return prefix_path, compressed, inflated, offsets_file

def compute_quantized_gt(cfg: Config, inflated_file: Path, tag: str) -> Path:
    gt_out = cfg.work_dir / f"quantized_gt_{tag}.bin"
    gt_tool = cfg.tools_dir / "compute_groundtruth"
    cmd = [str(gt_tool), "--data_type", "float", "--dist_fn", "l2", "--base_file", str(inflated_file),
           "--query_file", str(cfg.query_file), "--gt_file", str(gt_out), "--K", str(cfg.k)]
    run_cmd(cmd, timeout=cfg.timeout_gt, verbose=cfg.verbose)
    return gt_out

def compute_recall(cfg: Config, quantized_gt: Path) -> float:
    recall_tool = cfg.tools_dir / "calculate_recall"
    cmd = [str(recall_tool), str(cfg.raw_gt_file), str(quantized_gt), str(cfg.k)]
    out = run_cmd(cmd, timeout=cfg.timeout_recall, verbose=cfg.verbose)
    m = RECALL_REGEX.search(out)
    if not m:
        raise RuntimeError("Failed to parse recall from output:\n" + out)
    return float(m.group(1))

def evaluate_allocation(cfg: Config, allocation: List[int], tag: str) -> Tuple[float, IterRecord]:
    prefix_path, compressed, inflated, offsets_file = train_and_quantize(cfg, allocation, tag)
    if not inflated.exists():
        # If build not compiled with SAVE_INFLATED_PQ we cannot proceed.
        raise FileNotFoundError(f"Inflated file not found: {inflated}. Rebuild with SAVE_INFLATED_PQ defined.")
    quant_gt = compute_quantized_gt(cfg, inflated, tag)
    recall = compute_recall(cfg, quant_gt)
    if cfg.verbose:
        # Structured single-line log for filtering later
        print(
            "PQ_GREEDY_EVAL",
            f"tag={tag}",
            f"alloc={allocation}",
            f"bytes={sum(allocation)}",
            f"recall={recall:.6f}",
            flush=True,
        )
    artifacts = [compressed, inflated, quant_gt, offsets_file, prefix_path.with_name(prefix_path.name + "_pq_pivots.bin")]
    return recall, IterRecord(
        iteration=-1,
        allocation=allocation.copy(),
        recall=recall,
        bytes_per_vec=sum(allocation),
        prefix=str(prefix_path),
        improved=False,
        tag=tag,
        artifacts=artifacts,
    )

def greedy_search(cfg: Config) -> Dict[str, Any]:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    history: List[IterRecord] = []
    current_alloc = [cfg.initial_chunks for _ in range(cfg.num_buckets)]
    best_recall, rec = evaluate_allocation(cfg, current_alloc, tag="init")
    rec.iteration = 0
    rec.improved = True
    history.append(rec)
    if cfg.verbose:
        print(f"Initial allocation {current_alloc} => recall={best_recall:.6f}")
    # Track artifacts from the previous iteration (candidates) for cleanup.
    prev_iteration_records: List[IterRecord] = [rec]
    for it in range(1, cfg.max_iters + 1):
        candidates: List[Tuple[float, List[int], IterRecord]] = []
        improved = False
        for b in range(cfg.num_buckets):
            if current_alloc[b] + cfg.increment > cfg.max_per_bucket:
                print(f"Iter {it}: Bucket {b} exceeded max_per_bucket")
            if cfg.max_total_bytes is not None and sum(current_alloc) + cfg.increment > cfg.max_total_bytes:
                print(f"Iter {it}: Total allocation exceeded max_total_bytes")
                continue
            cand_alloc = current_alloc.copy()
            cand_alloc[b] += cfg.increment
            tag = f"it{it}_b{b}_plus{cfg.increment}"
            try:
                recall, rec_cand = evaluate_allocation(cfg, cand_alloc, tag=tag)
            except Exception as e:
                print(f"[WARN] Skipping candidate bucket {b} due to error: {e}")
                continue
            rec_cand.iteration = it
            candidates.append((recall, cand_alloc, rec_cand))
        if not candidates:
            if cfg.verbose:
                print("No feasible candidates; stopping.")
            break
        # Choose best recall (tie-breaker: fewer bytes, then lexicographically)
        candidates.sort(key=lambda x: (-x[0], sum(x[1]), x[1]))
        best_cand_recall, best_cand_alloc, best_cand_rec = candidates[0]
        if best_cand_recall > best_recall + 1e-9:  # improvement threshold
            best_recall = best_cand_recall
            current_alloc = best_cand_alloc
            best_cand_rec.improved = True
            history.append(best_cand_rec)
            improved = True
            if cfg.verbose:
                print(f"Iter {it}: improved recall -> {best_recall:.6f} with alloc {current_alloc}")
        else:
            # Record best (non-improving) candidate for audit, then stop.
            best_cand_rec.improved = False
            history.append(best_cand_rec)
            if cfg.verbose:
                print(f"Iter {it}: no improvement (best candidate recall={best_cand_recall:.6f}); stopping.")
            # Final cleanup of previous iteration candidates if needed
            if not cfg.keep_all:
                _cleanup_iteration(prev_iteration_records, preserve_allocation=current_alloc, verbose=cfg.verbose)
            break
        # After completing this iteration and deciding to continue, cleanup artifacts from previous iteration
        if not cfg.keep_all:
            _cleanup_iteration(prev_iteration_records, preserve_allocation=current_alloc, verbose=cfg.verbose)
        # Prepare for next iteration: current iteration's candidate records become previous
        prev_iteration_records = [cand[2] for cand in candidates]
    result = {
        "final_allocation": current_alloc,
        "final_recall": best_recall,
        "history": [dataclasses.asdict(r) for r in history],
    }
    if cfg.log_json:
        with open(cfg.log_json, 'w') as f:
            json.dump(result, f, indent=2)
    return result

def parse_args(argv: List[str]) -> Config:
    p = argparse.ArgumentParser(description="Greedy PQ bucket byte allocation search")
    p.add_argument('--base_file', required=True)
    p.add_argument('--query_file', required=True)
    p.add_argument('--raw_gt_file', required=True, help='Ground-truth on original base vectors')
    p.add_argument('--work_dir', required=True)
    p.add_argument('--tools_dir', required=True, help='Directory containing generate_pq_variable_chunks, compute_groundtruth, calculate_recall')
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--num_buckets', type=int, default=DEFAULT_NUM_BUCKETS, help='Number of contiguous buckets to partition the vector dim into')
    p.add_argument('--sampling_rate', type=float, default=0.1)
    p.add_argument('--initial_chunks', type=int, default=8)
    p.add_argument('--increment', type=int, default=8)
    p.add_argument('--max_iters', type=int, default=20)
    p.add_argument('--max_per_bucket', type=int, default=128)
    p.add_argument('--max_total_bytes', type=int, default=None)
    p.add_argument('--keep_all', action='store_true')
    p.add_argument('--pq_prefix_base', default='pq_cfg')
    p.add_argument('--inflate_suffix', default='_pq_compressed.bin_inflated.bin')
    p.add_argument('--log_json')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)
    return Config(
        base_file=Path(args.base_file),
        query_file=Path(args.query_file),
        raw_gt_file=Path(args.raw_gt_file),
        work_dir=Path(args.work_dir),
        tools_dir=Path(args.tools_dir),
        num_buckets=args.num_buckets,
        k=args.k,
        sampling_rate=args.sampling_rate,
        initial_chunks=args.initial_chunks,
        increment=args.increment,
        max_iters=args.max_iters,
        max_per_bucket=args.max_per_bucket,
        max_total_bytes=args.max_total_bytes,
        keep_all=args.keep_all,
        pq_prefix_base=args.pq_prefix_base,
        inflate_suffix=args.inflate_suffix,
        log_json=Path(args.log_json) if args.log_json else None,
        verbose=not args.quiet,
    )

def _read_fbin_header(path: Path) -> Tuple[int, int]:
    """Read DiskANN float32 binary header: returns (npts, dim)."""
    with open(path, 'rb') as f:
        import struct
        hdr = f.read(8)
        if len(hdr) != 8:
            raise ValueError(f"File too small for header: {path}")
        npts, dim = struct.unpack('<II', hdr)
        if npts == 0 or dim == 0:
            raise ValueError(f"Invalid header values in {path}: npts={npts} dim={dim}")
        return npts, dim

def _infer_dimension_and_buckets(cfg: Config):
    _, dim_base = _read_fbin_header(cfg.base_file)
    _, dim_query = _read_fbin_header(cfg.query_file)
    if dim_base != dim_query:
        raise ValueError(f"Base/query dim mismatch: {dim_base} vs {dim_query}")
    cfg.dim = dim_base
    if cfg.num_buckets <= 0 or cfg.num_buckets > cfg.dim:
        raise ValueError(f"num_buckets must be in (0, dim]; got {cfg.num_buckets} for dim={cfg.dim}")
    base_bucket = cfg.dim // cfg.num_buckets
    rem = cfg.dim % cfg.num_buckets
    cfg.bucket_sizes = [base_bucket + (1 if i < rem else 0) for i in range(cfg.num_buckets)]
    if cfg.verbose:
        print(f"Inferred dim={cfg.dim}; bucket_sizes={cfg.bucket_sizes}")

def _cleanup_iteration(records: List[IterRecord], preserve_allocation: List[int], verbose: bool):
    """Delete artifact files for a completed iteration except those matching preserve_allocation.

    We match on allocation list equality. Only executed when keep_all is False to save disk space.
    """
    preserved = preserve_allocation
    for rec in records:
        if rec.allocation == preserved:
            continue
        for path in rec.artifacts:
            try:
                if path and path.exists():
                    path.unlink()
            except Exception:
                pass
        # Also remove any file patterns based on prefix if directory is flat
        try:
            prefix = Path(rec.prefix)
            base = prefix.name
            parent = prefix.parent
            if parent.exists():
                for f in parent.glob(base + "*"):
                    if f.is_file():
                        try:
                            f.unlink()
                        except Exception:
                            pass
        except Exception:
            pass
    if verbose:
        print("[CLEANUP] Removed artifacts for completed iteration (except current best).")

def main(argv: List[str]) -> int:
    cfg = parse_args(argv)
    for path_attr in ('base_file','query_file','raw_gt_file','tools_dir'):
        p = getattr(cfg, path_attr)
        if not p.exists():
            print(f"ERROR: {path_attr} not found: {p}")
            return 1
    try:
        _infer_dimension_and_buckets(cfg)
    except Exception as e:
        print(f"ERROR: failed to infer dimension/buckets: {e}")
        return 1
    try:
        result = greedy_search(cfg)
    except CommandError as e:
        print(f"Execution failed: {e}")
        return 2
    print("\n=== Greedy Search Result ===")
    print("Final allocation:", result['final_allocation'])
    print(f"Final recall@{cfg.k}: {result['final_recall']:.6f}")
    if cfg.log_json:
        print("Log written to:", cfg.log_json)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
