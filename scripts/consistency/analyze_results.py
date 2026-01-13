#!/usr/bin/env python3

import argparse
import math
import struct
from pathlib import Path
from typing import List, Tuple


def read_bin_u32_matrix(path: Path) -> Tuple[int, int, List[int]]:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise EOFError(f"{path}: missing header")
        npts, ndims = struct.unpack("<II", header)
        payload = f.read()
    expected = 4 * npts * ndims
    if len(payload) != expected:
        raise EOFError(f"{path}: expected {expected} bytes, got {len(payload)}")
    data = list(struct.unpack(f"<{npts*ndims}I", payload))
    return npts, ndims, data


def read_truthset_ids(path: Path) -> Tuple[int, int, List[int]]:
    """Reads DiskANN compute_groundtruth output.

    The truthset file format is:
      u32 npts, u32 K,
      u32 ids[npts*K],
      f32 dists[npts*K]

    Some builds may omit the dists section; this reader handles both.
    """
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise EOFError(f"{path}: missing header")
        npts, k = struct.unpack("<II", header)

        ids_bytes = f.read(4 * npts * k)
        if len(ids_bytes) != 4 * npts * k:
            raise EOFError(f"{path}: missing ids matrix")
        ids = list(struct.unpack(f"<{npts*k}I", ids_bytes))
        return npts, k, ids


def read_bin_f32_matrix(path: Path) -> Tuple[int, int, List[float]]:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise EOFError(f"{path}: missing header")
        npts, ndims = struct.unpack("<II", header)
        payload = f.read()
    expected = 4 * npts * ndims
    if len(payload) != expected:
        raise EOFError(f"{path}: expected {expected} bytes, got {len(payload)}")
    data = list(struct.unpack(f"<{npts*ndims}f", payload))
    return npts, ndims, data


def recall_at_k(gt_ids: List[int], pred_ids: List[int], nq: int, k: int) -> float:
    hit = 0
    for qi in range(nq):
        gt = set(gt_ids[qi * k : (qi + 1) * k])
        pred = pred_ids[qi * k : (qi + 1) * k]
        hit += sum(1 for x in pred if x in gt)
    return hit / float(nq * k)


def top1_match_rate(a_ids: List[int], b_ids: List[int], nq: int, k: int) -> float:
    same = 0
    for qi in range(nq):
        if a_ids[qi * k] == b_ids[qi * k]:
            same += 1
    return same / float(nq)


def distance_error_common_ids(
    a_ids: List[int], a_dists: List[float], b_ids: List[int], b_dists: List[float], nq: int, k: int
) -> Tuple[float, float, float]:
    # Compare distance values only for IDs that appear in both top-k lists per query.
    # Returns (mean_abs, p99_abs, max_abs) over all common-id pairs.
    abs_errors: List[float] = []

    for qi in range(nq):
        a_map = {}
        base = qi * k
        for j in range(k):
            a_map[a_ids[base + j]] = a_dists[base + j]

        for j in range(k):
            bid = b_ids[base + j]
            if bid in a_map:
                abs_errors.append(abs(float(b_dists[base + j]) - float(a_map[bid])))

    if not abs_errors:
        return math.nan, math.nan, math.nan

    abs_errors.sort()
    mean_abs = sum(abs_errors) / float(len(abs_errors))
    p99_abs = abs_errors[int(0.99 * (len(abs_errors) - 1))]
    max_abs = abs_errors[-1]
    return mean_abs, p99_abs, max_abs


def _paths_for_prefix(prefix: Path, L: int) -> Tuple[Path, Path]:
    idx = Path(str(prefix) + f"_{L}_idx_uint32.bin")
    dist = Path(str(prefix) + f"_{L}_dists_float.bin")
    return idx, dist


def compare_block(name: str, gt_path: Path, float_prefix: Path, bf16_prefix: Path, L: int, k: int) -> None:
    gt_nq, gt_k, gt_ids = read_truthset_ids(gt_path)
    if gt_k < k:
        raise ValueError(f"GT K={gt_k} < requested K={k}")

    f_idx_path, f_dist_path = _paths_for_prefix(float_prefix, L)
    b_idx_path, b_dist_path = _paths_for_prefix(bf16_prefix, L)

    fq, fk, f_ids = read_bin_u32_matrix(f_idx_path)
    _, _, f_dists = read_bin_f32_matrix(f_dist_path)

    bq, bk, b_ids = read_bin_u32_matrix(b_idx_path)
    _, _, b_dists = read_bin_f32_matrix(b_dist_path)

    if fq != gt_nq or bq != gt_nq:
        raise ValueError(f"Query count mismatch: gt={gt_nq}, float={fq}, bf16={bq}")
    if fk != k or bk != k:
        raise ValueError(f"Result K mismatch: expected {k}, float={fk}, bf16={bk}")

    r_float = recall_at_k(gt_ids[: gt_nq * k], f_ids, gt_nq, k)
    r_bf16 = recall_at_k(gt_ids[: gt_nq * k], b_ids, gt_nq, k)
    top1 = top1_match_rate(f_ids, b_ids, gt_nq, k)
    mean_abs, p99_abs, max_abs = distance_error_common_ids(f_ids, f_dists, b_ids, b_dists, gt_nq, k)

    print(f"== {name} ==")
    print(f"L={L} K={k}")
    print(f"Recall@{k}: float={r_float*100:.2f}%  bf16={r_bf16*100:.2f}%  (delta={(r_bf16-r_float)*100:.2f}%)")
    print(f"Top1 ID match rate (bf16 vs float): {top1*100:.2f}%")
    print(
        "Distance abs error on common IDs: "
        f"mean={mean_abs:.6g}, p99={p99_abs:.6g}, max={max_abs:.6g}"
    )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare float vs bf16 search outputs (memory + disk)")
    ap.add_argument("--gt", required=True, help="Ground-truth .bin (uint32 ids), computed from float data")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)

    ap.add_argument("--mem_float", required=True, help="Memory float result prefix (no _L suffix)")
    ap.add_argument("--mem_bf16", required=True, help="Memory bf16 result prefix (no _L suffix)")

    ap.add_argument("--disk_float_full", required=True)
    ap.add_argument("--disk_bf16_full", required=True)

    ap.add_argument("--disk_float_pq", required=True)
    ap.add_argument("--disk_bf16_pq", required=True)

    args = ap.parse_args()

    gt_path = Path(args.gt)

    compare_block("Memory", gt_path, Path(args.mem_float), Path(args.mem_bf16), args.L, args.K)
    compare_block("Disk (full-precision)", gt_path, Path(args.disk_float_full), Path(args.disk_bf16_full), args.L, args.K)
    compare_block("Disk (PQ + reorder)", gt_path, Path(args.disk_float_pq), Path(args.disk_bf16_pq), args.L, args.K)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
