# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Promote Enron 384D FP16 files to FP32, preserving rows and normalization.

Requires NumPy. Source/generated data are not distributed with the repository.
Compute groundtruth separately on the exact promoted FP32 rows.
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


def convert(source, destination):
    with source.open("rb") as stream:
        rows, dimensions = struct.unpack("<II", stream.read(8))
    if not rows or dimensions != 384:
        raise ValueError(f"Expected nonempty 384D data: {source}")
    if source.stat().st_size != 8 + rows * dimensions * 2:
        raise ValueError(f"Expected FP16 data after the 8-byte header: {source}")
    original = np.memmap(
        source, mode="r", offset=8, dtype="<f2", shape=(rows, dimensions)
    )
    with destination.open("xb") as stream:
        stream.write(struct.pack("<II", rows, dimensions))
        for start in range(0, rows, 4096):
            block = original[start : start + 4096].astype("<f4")
            if not np.isfinite(block).all():
                raise ValueError(f"Nonfinite input at or after row {start}: {source}")
            block.tofile(stream)
    result = np.memmap(
        destination, mode="r", offset=8, dtype="<f4", shape=(rows, dimensions)
    )
    for start in range(0, rows, 4096):
        expected = original[start : start + 4096].astype("<f4")
        if not np.array_equal(
            result[start : start + 4096].view("<u4"), expected.view("<u4")
        ):
            raise ValueError(f"Read-back mismatch at row {start}: {destination}")
    return {
        "source_name": source.name,
        "source_sha256": sha256(source),
        "output_name": destination.name,
        "output_sha256": sha256(destination),
        "rows": rows,
        "dimensions": dimensions,
        "source_type": "fp16",
        "output_type": "fp32",
        "readback_bitwise_equal_to_fp16_promotion": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("target/enron-comparison/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output / name
        for name in ["enron-base-f32.bin", "enron-queries-f32.bin", "manifest.json"]
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Output files already exist; choose a fresh output directory")
    manifest = {
        "base": convert(args.base, outputs[0]),
        "queries": convert(args.queries, outputs[1]),
        "transformation": "Exact FP16-to-FP32 promotion; original row order; no renormalization",
    }
    outputs[2].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
