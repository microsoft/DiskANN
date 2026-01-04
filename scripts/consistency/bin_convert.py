#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path


def _read_u32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise EOFError("Unexpected EOF while reading u32")
    return struct.unpack("<I", b)[0]


def _float_to_bf16_bits_rne(f32: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", f32))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF


def float_bin_to_bf16_bin(in_path: Path, out_path: Path) -> None:
    with in_path.open("rb") as r:
        npts = _read_u32(r)
        ndims = _read_u32(r)
        total = npts * ndims

        payload = r.read(4 * total)
        if len(payload) != 4 * total:
            raise EOFError(
                f"Unexpected EOF: expected {4*total} bytes of float32 payload, got {len(payload)}"
            )

    floats = struct.unpack(f"<{total}f", payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as w:
        w.write(struct.pack("<II", npts, ndims))
        # Stream out as u16 bf16 payload.
        buf = bytearray(2 * total)
        off = 0
        for x in floats:
            struct.pack_into("<H", buf, off, _float_to_bf16_bits_rne(x))
            off += 2
        w.write(buf)


def bf16_bin_to_float_bin(in_path: Path, out_path: Path) -> None:
    with in_path.open("rb") as r:
        npts = _read_u32(r)
        ndims = _read_u32(r)
        total = npts * ndims

        payload = r.read(2 * total)
        if len(payload) != 2 * total:
            raise EOFError(
                f"Unexpected EOF: expected {2*total} bytes of bf16 payload, got {len(payload)}"
            )

    bf16_vals = struct.unpack(f"<{total}H", payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as w:
        w.write(struct.pack("<II", npts, ndims))
        buf = bytearray(4 * total)
        off = 0
        for v in bf16_vals:
            bits = (v & 0xFFFF) << 16
            struct.pack_into("<I", buf, off, bits)
            off += 4
        w.write(buf)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert DiskANN .bin between float32 and bf16")
    ap.add_argument("--mode", choices=["float_to_bf16", "bf16_to_float"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if args.mode == "float_to_bf16":
        float_bin_to_bf16_bin(in_path, out_path)
    else:
        bf16_bin_to_float_bin(in_path, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
