import struct
import random
import argparse

def get_dtype_size(data_type):
    if data_type == 'float':
        return 4  # float32
    elif data_type == 'int8':
        return 1
    elif data_type == 'uint8':
        return 1
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

def split_u8bin(input_bin, input_label, output_bin_prefix, output_label_prefix, split_ratio, data_type):
    dtype_size = get_dtype_size(data_type)

    # Read labels
    with open(input_label, 'r') as f:
        labels = f.readlines()

    N = len(labels)
    print(f"Total points (from labels): {N}")

    # Generate shuffled indices and split
    indices = list(range(N))
    random.shuffle(indices)
    split_point = int(split_ratio * N)
    indices_train = sorted(indices[:split_point])
    indices_test = sorted(indices[split_point:])

    # Read vectors from .u8bin
    with open(input_bin, 'rb') as f:
        N_bin, D = struct.unpack('II', f.read(8))
        print(f"From binary file - points: {N_bin}, dim: {D}")

        assert N_bin == N, "Mismatch in number of points between label and vector file"

        # Total bytes to read = points * dim * bytes_per_value
        total_bytes = N * D * dtype_size
        data = f.read(total_bytes)

    def write_bin(file_path, indices_subset):
        with open(file_path, 'wb') as f_out:
            f_out.write(struct.pack('II', len(indices_subset), D))
            for idx in indices_subset:
                start = idx * D * dtype_size
                end = start + D * dtype_size
                f_out.write(data[start:end])

    # Write splits
    out_bin_train = f"{output_bin_prefix}.train.{data_type}bin"
    out_bin_test = f"{output_bin_prefix}.test.{data_type}bin"
    out_label_train = f"{output_label_prefix}.train.txt"
    out_label_test = f"{output_label_prefix}.test.txt"

    write_bin(out_bin_train, indices_train)
    write_bin(out_bin_test, indices_test)

    with open(out_label_train, 'w') as f:
        for idx in indices_train:
            f.write(labels[idx])

    with open(out_label_test, 'w') as f:
        for idx in indices_test:
            f.write(labels[idx])

    print(f"Done! Created:")
    print(f" - {out_bin_train}")
    print(f" - {out_bin_test}")
    print(f" - {out_label_train}")
    print(f" - {out_label_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split .u8bin vector file and label file into train/test splits")
    parser.add_argument('--input_bin', required=True, help='Input .bin file path')
    parser.add_argument('--input_label', required=True, help='Input label text file path')
    parser.add_argument('--output_bin_prefix', required=True, help='Output prefix for .u8bin split files')
    parser.add_argument('--output_label_prefix', required=True, help='Output prefix for label split files')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train split ratio (default 0.8)')
    parser.add_argument('--data_type', choices=['float', 'int8', 'uint8'], required=True, help='Data type of vectors')

    args = parser.parse_args()

    split_u8bin(args.input_bin, args.input_label, args.output_bin_prefix, args.output_label_prefix, args.split_ratio, args.data_type)
