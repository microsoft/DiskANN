# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tempfile import mkdtemp
import time

import diskannpy as dap
import numpy as np

from .create_test_data import random_vectors


def build_random_vectors_and_disk_index(
    dtype, metric, index_prefix: str = "ann", seed: int = 12345, build_pq_bytes: int = 0
):
    query_vectors: np.ndarray = random_vectors(
        1000, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(
        10000, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    dap.build_disk_index(
        data=index_vectors,
        distance_metric=metric,
        index_directory=ann_dir,
        graph_degree=16,
        build_memory_maximum=10,
        complexity=32,
        num_threads=0,
        build_pq_bytes=build_pq_bytes,
        num_of_pq_chunks=10,
        index_prefix=index_prefix,
    )

    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        os.path.join(ann_dir, "vectors.bin"),
    )
