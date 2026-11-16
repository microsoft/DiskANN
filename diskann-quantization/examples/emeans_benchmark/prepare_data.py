# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Prepare reproducible SIFT and repository-YFCC cases without publishing the datasets."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    with path.open('rb') as stream:
        header = stream.read(8)
        if header == b'version ':
            raise ValueError(f'{path} is a Git LFS pointer; hydrate it or pass --yfcc-dir with real data')
        n, d = struct.unpack('<II', header)
        return np.frombuffer(stream.read(), dtype='<f4').reshape(n, d)


def write(path, data):
    payload = struct.pack('<II', *data.shape) + np.asarray(data, dtype='<f4').tobytes()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f'conflicting prepared data: {path}')
        return
    with path.open('xb') as stream:
        stream.write(payload)


def copy(source, target):
    if target.exists():
        if sha256(source) != sha256(target):
            raise FileExistsError(f'conflicting prepared data: {target}')
        return
    shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sift-dir', type=Path, required=True)
    parser.add_argument('--yfcc-dir', type=Path, default=Path('test_data/yfcc'))
    parser.add_argument('--output', type=Path, default=Path('target/emeans-benchmark'))
    args = parser.parse_args()
    output = args.output
    data = output / 'data'
    data.mkdir(parents=True, exist_ok=True)
    for name in ['sift-train-100k.fbin', 'sift-test-10k.fbin']:
        target = data / name
        copy(args.sift_dir / name, target)
    source_manifest = json.loads((args.sift_dir / 'dataset.json').read_text(encoding='utf-8'))
    for name, info in source_manifest['files'].items():
        assert sha256(data / name) == info['sha256']

    database_path = args.yfcc_dir / 'yfcc_10k.fbin'
    queries_path = args.yfcc_dir / 'yfcc_query_100.fbin'
    database, queries = read(database_path), read(queries_path)
    order = np.random.default_rng(20261116).permutation(len(database))
    test_ids = order[:2000]
    excluded = {row.tobytes() for row in database[test_ids]} | {row.tobytes() for row in queries}
    train_ids = [i for i in order[2000:] if database[i].tobytes() not in excluded]
    write(data / 'yfcc-train.fbin', database[train_ids])
    write(data / 'yfcc-test.fbin', database[test_ids])
    copy(database_path, data / 'yfcc-database.fbin')
    copy(queries_path, data / 'yfcc-queries.fbin')

    def path(name): return (data / name).as_posix()
    cases = [
        dict(name='SIFT clustering', mode='clustering', train=path('sift-train-100k.fbin'), test=path('sift-test-10k.fbin'),
             train_rows=100000, test_rows=10000, clusters=1024, partitions=1, database=None, queries=None, query_rows=0),
        dict(name='SIFT PQ', mode='pq', train=path('sift-train-100k.fbin'), test=path('sift-test-10k.fbin'),
             train_rows=50000, test_rows=10000, clusters=256, partitions=16,
             database=path('sift-train-100k.fbin'), queries=path('sift-test-10k.fbin'), query_rows=1000),
        dict(name='YFCC PQ', mode='pq', train=path('yfcc-train.fbin'), test=path('yfcc-test.fbin'),
             train_rows=len(train_ids), test_rows=2000, clusters=256, partitions=24,
             database=path('yfcc-database.fbin'), queries=path('yfcc-queries.fbin'), query_rows=len(queries)),
    ]
    runs = [dict(algorithm='lloyd', epochs=t) for t in [5, 12, 25]] + [dict(algorithm='emeans', epochs=t) for t in [1, 2, 3, 5, 10]]
    config = dict(seeds=[1, 17, 42, 99, 1234], repeats=3, output=(output / 'measurements.jsonl').as_posix(), runs=runs, cases=cases)
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    pilot = dict(config, seeds=[42], repeats=1, output=(output / 'pilot.jsonl').as_posix(),
                 runs=[dict(algorithm='lloyd', epochs=5), dict(algorithm='emeans', epochs=3)],
                 cases=[dict(c, train_rows=min(c['train_rows'], 2000), test_rows=100, clusters=32,
                             query_rows=10 if c['query_rows'] else 0) for c in cases])
    (output / 'pilot.json').write_text(json.dumps(pilot, indent=2), encoding='utf-8')
    manifest = dict(sift=source_manifest, yfcc=dict(database_sha256=sha256(database_path), queries_sha256=sha256(queries_path),
                    split_seed=20261116, training_rows=len(train_ids), held_out_rows=2000,
                    excluded_training_rows_matching_validation_or_query=8000-len(train_ids)), numpy=np.__version__,
                    inputs={p.name: dict(shape=list(read(p).shape), sha256=sha256(p)) for p in data.glob('*.fbin')})
    (output / 'datasets.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
