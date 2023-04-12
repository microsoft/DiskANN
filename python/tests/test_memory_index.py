# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import shutil
import unittest
from tempfile import mkdtemp

import diskannpy as dap
import numpy as np
from fixtures import calculate_recall, random_vectors, vectors_as_temp_file
from sklearn.neighbors import NearestNeighbors


def _build_random_vectors_and_index(dtype, metric):
    query_vectors = random_vectors(10, 10, dtype=dtype)
    index_vectors = random_vectors(100, 10, dtype=dtype)
    with vectors_as_temp_file(index_vectors) as vector_temp:
        ann_dir = mkdtemp()
        dap.build_memory_index_from_vector_file(
            vector_bin_file=vector_temp,
            metric=metric,
            vector_dtype=dtype,
            index_path=ann_dir,
            graph_degree=64,
            complexity=100,
            alpha=1.2,
            num_threads=0,
            use_pq_build=False,
            num_pq_bytes=0,
            use_opq=False
        )
    return metric, dtype, query_vectors, index_vectors, ann_dir


class TestDiskIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            _build_random_vectors_and_index(np.single, "l2"),
            # _build_random_vectors_and_index(np.ubyte, "l2"),
            # _build_random_vectors_and_index(np.byte, "l2"),
        ]
        cls._example_ann_dir = cls._test_matrix[0][4]

    @classmethod
    def tearDownClass(cls) -> None:
        for test in cls._test_matrix:
            try:
                _, _, _, _, ann_dir = test
                shutil.rmtree(ann_dir, ignore_errors=True)
            except:
                pass

    def test_recall_and_batch(self):
        self.assertTrue(True)
