# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tempfile import mkdtemp
import time

import diskannpy as dap
import numpy as np

from .create_test_data import random_vectors

def build_random_vectors_and_async_index(
    dtype, metric, with_tags: bool = False, index_prefix: str = "ann", seed: int = 12345, 
    use_pq: bool = False, num_pq_bytes: int = 0, use_opq: bool = False, 
    query_vector_count: int = 1000, index_vector_count: int = 10000
):
    query_vectors: np.ndarray = random_vectors(query_vector_count, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(index_vector_count, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    if with_tags:
        rng = np.random.default_rng(seed)
        tags = np.arange(start=1, stop=10001, dtype=np.uint32)
        rng.shuffle(tags)
    else:
        tags = ""
    dap.build_async_index(
        data=index_vectors,
        distance_metric=metric,
        index_directory=ann_dir,
        graph_degree=64,
        complexity=100,
        alpha=1.2,
        num_start_pts=1,
        num_threads=1,
        graph_slack_factor=1.3,
        max_fp_vecs_per_prune=0,
        num_tasks=1,
        pq_seed=seed,
        insert_minibatch_size=1,
        num_pq_bytes=num_pq_bytes,
        tags=tags,
        index_prefix=index_prefix,
        use_pq=use_pq,
        use_opq=use_opq,
    )

    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        os.path.join(ann_dir, "vectors.bin"),
        # tags,
    )
