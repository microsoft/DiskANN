#!/usr/bin/env python3
"""
Slice (truncate) the dimensionality of a float32 binary matrix file with DiskANN-style header.

Input binary format (little-endian):
  uint32 N  -> number of rows
  uint32 D  -> original dimensionality
  N * D float32 values (row-major)

Output binary format:
  uint32 N  -> same number of rows
  uint32 D' -> requested truncated dimensionality (d_out)
  N * D' float32 values (row-major), consisting of the first D' components of each input row

This tool performs streaming / chunked processing so it can handle large matrices without
loading the entire file into RAM. For each chunk of rows, we read the full float32 block,
reshape, slice the first d_out columns, then write them out.

Example:
  python slice_f32_dims.py input.fbin output.fbin --dims-out 384

Validation:
  * Ensures 0 < D' <= D
  * (Optional) Verifies input file size matches header expectations

Notes:
  * The operation is a simple truncation; no re-ordering or projection.
  * If you need arbitrary dimension indices, extend the tool to accept an index list.
"""
from __future__ import annotations
import argparse
import os
import struct
from typing import BinaryIO
import numpy as np

HEADER_STRUCT = struct.Struct('<II')  # uint32 N, uint32 D

class FormatError(Exception):
    """Raised for structural format issues in the binary file."""
    pass

def _read_header(f: BinaryIO):
    hdr = f.read(HEADER_STRUCT.size)
    if len(hdr) != HEADER_STRUCT.size:
        raise FormatError("File too small to contain header (expected 8 bytes).")
    n_rows, dim = HEADER_STRUCT.unpack(hdr)
    if n_rows == 0 or dim == 0:
        raise FormatError(f"Invalid header values: rows={n_rows}, dim={dim}")
    return n_rows, dim

def slice_f32_dimensions(
    input_path: str,
    output_path: str,
    dims_out: int,
    chunk_rows: int = 65536,
    verify_size: bool = True,
    verbose: bool = True,
):
    """Truncate a float32 matrix file to its first `dims_out` columns.

    Parameters
    ----------
    input_path : str
        Path to source .bin file (float32 with 8-byte header).
    output_path : str
        Path to destination .bin file (will be overwritten).
    dims_out : int
        Number of leading dimensions to keep (D'). Must satisfy 0 < D' <= D.
    chunk_rows : int, optional
        Number of rows to process per chunk.
    verify_size : bool, optional
        If True, validates input size against header.
    verbose : bool, optional
        If True, prints progress and summary.
    """
    if input_path == output_path:
        raise ValueError("Input and output paths must differ; refusing in-place overwrite.")

    file_size = os.path.getsize(input_path)
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        n_rows, dim = _read_header(fin)
        if verbose:
            print(f"Header: rows={n_rows}, dim={dim}")
        if dims_out <= 0 or dims_out > dim:
            raise ValueError(f"dims_out must be in (0, {dim}] — got {dims_out}")

        expected_data_bytes = n_rows * dim * 4  # float32
        expected_total = expected_data_bytes + HEADER_STRUCT.size
        if verify_size and file_size != expected_total:
            raise FormatError(
                f"Input size mismatch: file has {file_size} bytes, expected {expected_total}")

        # Write new header with truncated dimension.
        fout.write(HEADER_STRUCT.pack(n_rows, dims_out))

        elems_per_row = dim
        row_bytes = elems_per_row * 4
        slice_cols = dims_out
        rows_done = 0

        # Streaming loop
        while rows_done < n_rows:
            rows_left = n_rows - rows_done
            this_chunk_rows = min(chunk_rows, rows_left)
            chunk_bytes = this_chunk_rows * row_bytes
            buf = fin.read(chunk_bytes)
            if len(buf) != chunk_bytes:
                raise FormatError(
                    f"Unexpected EOF: needed {chunk_bytes} bytes, got {len(buf)} at row {rows_done}.")
            # Interpret chunk as float32 flat array
            arr = np.frombuffer(buf, dtype=np.float32, count=this_chunk_rows * elems_per_row)
            # Reshape to (rows, dim) then slice first dims_out columns
            arr = arr.reshape(this_chunk_rows, elems_per_row)[:, :slice_cols]
            fout.write(arr.tobytes(order='C'))

            rows_done += this_chunk_rows
            if verbose and (rows_done % (chunk_rows * 10) == 0 or rows_done == n_rows):
                print(f"Processed {rows_done}/{n_rows} rows ({rows_done / n_rows:.1%}).")

        if verbose:
            out_size = os.path.getsize(output_path)
            expected_out = HEADER_STRUCT.size + n_rows * dims_out * 4
            print(f"Done. Output size = {out_size} bytes (expected {expected_out}).")
            if out_size != expected_out:
                raise FormatError(
                    f"Output size mismatch: got {out_size}, expected {expected_out}.")

def memory_map_slice_f32_dimensions(
    input_path: str,
    output_path: str,
    dims_out: int,
    verbose: bool = True,
):
    """Alternative implementation using a memory map of the full matrix.

    This requires enough addressable virtual memory for the entire input. It may
    be faster for SSD-backed files when the OS can efficiently page in needed regions.
    """
    file_size = os.path.getsize(input_path)
    with open(input_path, 'rb') as f:
        n_rows, dim = _read_header(f)
    if dims_out <= 0 or dims_out > dim:
        raise ValueError(f"dims_out must be in (0, {dim}] — got {dims_out}")

    expected_data_bytes = n_rows * dim * 4
    expected_total = expected_data_bytes + HEADER_STRUCT.size
    if file_size != expected_total:
        raise FormatError(
            f"Input size mismatch: file has {file_size} bytes, expected {expected_total}")
    if verbose:
        print(f"Memmap header: rows={n_rows}, dim={dim}")

    data_offset = HEADER_STRUCT.size
    mm = np.memmap(input_path, dtype=np.float32, mode='r', offset=data_offset, shape=(n_rows * dim,))
    arr = np.asarray(mm).reshape(n_rows, dim)[:, :dims_out]

    with open(output_path, 'wb') as fout:
        fout.write(HEADER_STRUCT.pack(n_rows, dims_out))
        fout.write(arr.astype(np.float32, copy=False).tobytes(order='C'))

    if verbose:
        print("Memmap slicing complete.")


def main():
    parser = argparse.ArgumentParser(description="Truncate float32 matrix binary to first D' dimensions.")
    parser.add_argument('input', help='Input .bin path (float32)')
    parser.add_argument('output', help='Output .bin path (float32 truncated)')
    parser.add_argument('--dims-out', type=int, required=True, help="Number of leading dimensions to retain (D')")
    parser.add_argument('--chunk-rows', type=int, default=65536, help='Rows per streaming chunk')
    parser.add_argument('--no-verify', action='store_true', help='Skip input size verification (streaming mode)')
    parser.add_argument('--memmap', action='store_true', help='Use memory-mapped whole-file method')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    args = parser.parse_args()

    if args.memmap:
        memory_map_slice_f32_dimensions(
            args.input,
            args.output,
            args.dims_out,
            verbose=not args.quiet,
        )
    else:
        slice_f32_dimensions(
            args.input,
            args.output,
            args.dims_out,
            chunk_rows=args.chunk_rows,
            verify_size=not args.no_verify,
            verbose=not args.quiet,
        )

if __name__ == '__main__':
    main()
