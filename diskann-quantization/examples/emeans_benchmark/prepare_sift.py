# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Prepare reproducible fbin inputs from ANN Benchmarks' public SIFT HDF5 file."""

import hashlib
import json
from pathlib import Path
import struct
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    source, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    # Optional task-local wheels; no global installation is needed.
    sys.path.insert(0, str(output / 'python'))
    import h5py
    import numpy as np

    seed = 20261112
    with h5py.File(source, 'r') as dataset:
        database = np.asarray(dataset['train'], dtype='<f4')
        test = np.asarray(dataset['test'][:10000], dtype='<f4')
    test_rows = {row.tobytes() for row in test}
    ids, excluded = [], 0
    for row in np.random.default_rng(seed).permutation(len(database)):
        if database[row].tobytes() in test_rows:
            excluded += 1
            continue
        ids.append(row)
        if len(ids) == 100000:
            break
    if len(ids) != 100000:
        raise ValueError('fewer than 100,000 eligible training vectors')
    train = np.ascontiguousarray(database[ids])
    files = {}
    for name, data in [('sift-train-100k.fbin', train), ('sift-test-10k.fbin', test)]:
        path = output / name
        with path.open('xb') as stream:
            stream.write(struct.pack('<II', *data.shape))
            stream.write(data.tobytes())
        files[name] = {'shape': list(data.shape), 'sha256': sha256(path)}
    np.save(output / 'sift-training-indices.npy', np.asarray(ids, dtype=np.int64))
    metadata = {
        'url': 'https://ann-benchmarks.com/sift-128-euclidean.hdf5',
        'source_sha256': sha256(source),
        'database_rows': len(database),
        'sampling_seed': seed,
        'excluded_training_rows_identical_to_test': excluded,
        'numpy': np.__version__,
        'h5py': h5py.__version__,
        'files': files,
    }
    (output / 'dataset.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
