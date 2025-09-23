import os
import struct
import numpy as np
from typing import BinaryIO

"""
Binary format description (little-endian assumed):
- uint32: number of rows (N)
- uint32: dimension (D)
- N * D float16 values in row-major order (little-endian IEEE 754 half precision)

Goal: Convert to a new file with:
- uint32: number of rows (N)
- uint32: dimension (D)
- N * D float32 values (row-major) (4 bytes each)

The converter supports large files via chunked streaming to avoid loading the whole dataset in RAM.
"""

HEADER_STRUCT = struct.Struct('<II')  # little-endian unsigned int, unsigned int

class FormatError(Exception):
    pass

def _read_header(f: BinaryIO):
    header_bytes = f.read(HEADER_STRUCT.size)
    if len(header_bytes) != HEADER_STRUCT.size:
        raise FormatError("File too small to contain header (needs 8 bytes).")
    n_rows, dim = HEADER_STRUCT.unpack(header_bytes)
    if n_rows == 0 or dim == 0:
        raise FormatError(f"Invalid header values: rows={n_rows}, dim={dim}.")
    return n_rows, dim

def convert_fp16_to_fp32(
    input_path: str,
    output_path: str,
    chunk_rows: int = 65536,
    verify_size: bool = True,
    verbose: bool = True,
):
    """
    Convert a float16 binary matrix file (with 8-byte header) to float32 format.

    Parameters
    ----------
    input_path : str
        Path to source .bin file.
    output_path : str
        Path to destination .bin file (will be overwritten).
    chunk_rows : int, optional
        Number of rows to process per chunk (adjust for memory). Each chunk uses roughly
        chunk_rows * dim * (2 + 4) bytes transiently.
    verify_size : bool, optional
        If True, ensure input file size matches expected size from header.
    verbose : bool, optional
        If True, prints progress information.
    """
    if input_path == output_path:
        raise ValueError("Input and output paths must differ; refusing to overwrite in-place.")

    file_size = os.path.getsize(input_path)
    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        n_rows, dim = _read_header(fin)
        if verbose:
            print(f"Header: rows={n_rows}, dim={dim}")

        expected_data_bytes = n_rows * dim * 2  # float16 => 2 bytes
        expected_total = expected_data_bytes + HEADER_STRUCT.size
        if verify_size and file_size != expected_total:
            raise FormatError(
                f"Input size mismatch: file has {file_size} bytes, expected {expected_total}"
            )

        # Write new header (same values)
        fout.write(HEADER_STRUCT.pack(n_rows, dim))

        # Process in chunks
        rows_done = 0
        elems_per_row = dim
        f16_row_bytes = elems_per_row * 2

        while rows_done < n_rows:
            rows_left = n_rows - rows_done
            this_chunk_rows = min(chunk_rows, rows_left)
            chunk_bytes = this_chunk_rows * f16_row_bytes
            buf = fin.read(chunk_bytes)
            if len(buf) != chunk_bytes:
                raise FormatError(
                    f"Unexpected EOF: needed {chunk_bytes} bytes, got {len(buf)} at row {rows_done}."
                )
            # Interpret as float16
            arr_f16 = np.frombuffer(buf, dtype=np.float16, count=this_chunk_rows * elems_per_row)
            # Convert to float32
            arr_f32 = arr_f16.astype(np.float32, copy=False)
            # Write out as float32
            fout.write(arr_f32.tobytes(order='C'))
            rows_done += this_chunk_rows
            if verbose and (rows_done % (chunk_rows * 10) == 0 or rows_done == n_rows):
                print(f"Converted {rows_done}/{n_rows} rows ({rows_done / n_rows:.1%}).")

        if verbose:
            out_size = os.path.getsize(output_path)
            expected_out = HEADER_STRUCT.size + n_rows * dim * 4  # float32 4 bytes each
            print(f"Done. Output size = {out_size} bytes (expected {expected_out}).")
            if out_size != expected_out:
                raise FormatError(
                    f"Output size mismatch: got {out_size}, expected {expected_out}." )


def memory_map_convert_fp16_to_fp32(input_path: str, output_path: str, verbose: bool = True):
    """
    Alternative conversion using memory-mapped arrays (may require enough virtual address space).
    Loads the entire float16 matrix via memmap and writes float32 version.
    """
    file_size = os.path.getsize(input_path)
    with open(input_path, 'rb') as f:
        n_rows, dim = _read_header(f)
    expected_data_bytes = n_rows * dim * 2
    expected_total = expected_data_bytes + HEADER_STRUCT.size
    if file_size != expected_total:
        raise FormatError(
            f"Input size mismatch: file has {file_size} bytes, expected {expected_total}" )
    if verbose:
        print(f"Memmap header: rows={n_rows}, dim={dim}")

    # Memory-map the data portion only
    data_offset = HEADER_STRUCT.size
    mm = np.memmap(input_path, dtype=np.float16, mode='r', offset=data_offset, shape=(n_rows * dim,))
    arr_f32 = np.asarray(mm, dtype=np.float32)  # creates a new float32 ndarray

    with open(output_path, 'wb') as fout:
        fout.write(HEADER_STRUCT.pack(n_rows, dim))
        fout.write(arr_f32.tobytes(order='C'))

    if verbose:
        print("Memmap conversion complete.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert float16 matrix binary file to float32 format.")
    parser.add_argument('input', help='Input .bin path (float16)')
    parser.add_argument('output', help='Output .bin path (float32)')
    parser.add_argument('--chunk-rows', type=int, default=65536, help='Rows per streaming chunk')
    parser.add_argument('--no-verify', action='store_true', help='Skip input size verification')
    parser.add_argument('--memmap', action='store_true', help='Use memory-mapped whole-file method')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    args = parser.parse_args()

    if args.memmap:
        memory_map_convert_fp16_to_fp32(args.input, args.output, verbose=not args.quiet)
    else:
        convert_fp16_to_fp32(
            args.input,
            args.output,
            chunk_rows=args.chunk_rows,
            verify_size=not args.no_verify,
            verbose=not args.quiet,
        )

if __name__ == '__main__':
    # If run as a script inside the notebook environment, you can call main()
    # or directly call convert_fp16_to_fp32(). Example (uncomment and edit paths):
     convert_fp16_to_fp32('/mnt/ravi/SentenceChunk_OAILarge_query_normalized_6809.bin', '/mnt/ravi/openai_query.fbin')
    