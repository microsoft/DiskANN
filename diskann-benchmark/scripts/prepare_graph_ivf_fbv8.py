# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Promote local FBV8 Mimir FP16 vectors to FP32 for the static IVF benchmark.

Requires NumPy. Preserves every base row and its original order. Queries may be
selected deterministically for a bounded quality check. Compute fresh squared-L2
groundtruth on the promoted corpus; the supplied truth file is not reused.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote(source, destination, query_count=None):
    with source.open("rb") as stream:
        rows, dimensions = struct.unpack("<II", stream.read(8))
    if not rows or dimensions != 896:
        raise ValueError(f"Expected nonempty 896D data: {source}")
    if source.stat().st_size != 8 + 2 * rows * dimensions:
        raise ValueError(f"Expected FP16 coordinates after the header: {source}")
    count = rows if query_count is None else query_count
    if not 1 <= count <= rows:
        raise ValueError(f"Row count must be between 1 and {rows}")
    ids = np.linspace(0, rows - 1, count, dtype=np.int64)
    original = np.memmap(
        source, mode="r", offset=8, dtype="<f2", shape=(rows, dimensions)
    )
    with destination.open("xb") as stream:
        stream.write(struct.pack("<II", count, dimensions))
        for start in range(0, count, 4096):
            block = original[ids[start : start + 4096]].astype("<f4")
            if not np.isfinite(block).all():
                raise ValueError(f"Nonfinite input at output row {start}: {source}")
            block.tofile(stream)
    result = np.memmap(
        destination, mode="r", offset=8, dtype="<f4", shape=(count, dimensions)
    )
    for start in range(0, count, 4096):
        expected = original[ids[start : start + 4096]].astype("<f4")
        if not np.array_equal(
            result[start : start + 4096].view("<u4"), expected.view("<u4")
        ):
            raise ValueError(f"Read-back mismatch at row {start}: {destination}")
    return {
        "source_name": source.name,
        "source_sha256": sha256(source),
        "output_name": destination.name,
        "output_sha256": sha256(destination),
        "source_rows": rows,
        "output_rows": count,
        "dimensions": dimensions,
        "query_row_ids": ids.tolist() if query_count is not None else None,
        "readback_bitwise_equal_to_fp16_promotion": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = [args.output / name for name in [
        "fbv8-base-f32.bin", "fbv8-queries-f32.bin", "manifest.json"
    ]]
    if any(path.exists() for path in targets):
        raise FileExistsError("Outputs already exist; choose a fresh directory")
    manifest = {
        "transformation": "Exact FP16-to-FP32 promotion; no normalization; all base rows in original order",
        "base": promote(args.source / "fbv_8_normalized_dataset_958152.bin", targets[0]),
        "queries": promote(args.source / "fbv_8_normalized_query_6380.bin", targets[1], args.query_count),
    }
    targets[2].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
