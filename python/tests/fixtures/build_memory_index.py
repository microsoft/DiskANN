# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from tempfile import mkdtemp

import diskannpy as dap
import numpy as np

from .create_test_data import random_vectors


def build_random_vectors_and_memory_index(dtype, metric):
    query_vectors: np.ndarray = random_vectors(1000, 10, dtype=dtype)
    index_vectors: np.ndarray = random_vectors(10000, 10, dtype=dtype)
    ann_dir = mkdtemp()
    dap.build_memory_index(
        data=index_vectors,
        metric=metric,
        index_directory=ann_dir,
        graph_degree=16,
        complexity=32,
        alpha=1.2,
        num_threads=0,
        use_pq_build=False,
        num_pq_bytes=8,
        use_opq=False,
        filter_complexity=32
    )
    return metric, dtype, query_vectors, index_vectors, ann_dir, os.path.join(ann_dir, "vectors.bin")