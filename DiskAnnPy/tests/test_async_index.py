# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import unittest

from pathlib import Path
from tempfile import mkdtemp

#from DiskAnnPy.tests.fixtures.build_async_index import build_random_vectors_and_async_index
import diskannpy as dap
import numpy as np
from fixtures import build_random_vectors_and_async_index, calculate_recall, PQ_RECALL_CUTOFF
from fixtures import random_vectors
from sklearn.neighbors import NearestNeighbors
import asyncio

class TestAsyncIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._test_matrix = [
            build_random_vectors_and_async_index(np.float32, "l2"),
            build_random_vectors_and_async_index(np.uint8, "l2"),
            build_random_vectors_and_async_index(np.int8, "l2"),
            build_random_vectors_and_async_index(np.float32, "cosine"),
        ]
        cls._test_matrix_with_pq = [
            build_random_vectors_and_async_index(np.float32, "l2", use_pq=True, num_pq_bytes=5, use_opq=False),
            build_random_vectors_and_async_index(np.uint8, "l2", use_pq=True, num_pq_bytes=5, use_opq=False),
            build_random_vectors_and_async_index(np.int8, "l2", use_pq=True, num_pq_bytes=5, use_opq=False),
            build_random_vectors_and_async_index(np.int8, "cosine", use_pq=True, num_pq_bytes=5, use_opq=False),
        ]

    def test_search_pq_true(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix_with_pq:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        build_pq_bytes=5,
                        use_pq=True,
                    )
                    k = 5
                    response = index.search(
                        query_vectors[0], k_value=k, l_value=5, use_full_precision_to_search=False)
                    self.assertIsInstance(response, dap.QueryResponse)
                    ids, dists = response
                    self.assertEqual(ids.shape[0], k)
                    self.assertEqual(dists.shape[0], k)

    def test_search_pq_false(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                    )
                    k = 5
                    response = index.search(
                        query_vectors[0], k_value=k, l_value=5)
                    self.assertIsInstance(response, dap.QueryResponse)
                    ids, dists = response
                    self.assertEqual(ids.shape[0], k)
                    self.assertEqual(dists.shape[0], k)

    def test_average_degree(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.AsyncDiskIndex(
                    index_directory=ann_dir,
                    num_threads=1,
                )
                avg_degree = index.get_average_degree()
                self.assertTrue(avg_degree > 0, "Average degree should be greater than 0")

    def test_recall_and_batch(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.AsyncDiskIndex(
                    index_directory=ann_dir,
                    num_threads=1,
                )
                k = 5
                batch_response = index.batch_search(
                    query_vectors,
                    k_value=k,
                    l_value=5,
                    num_threads=8,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatchWithStats)

                diskann_neighbors, diskann_distances, diskann_stats = batch_response
                if metric in ["l2", "cosine"]:
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric=metric
                    )
                    knn.fit(index_vectors)
                    knn_distances, knn_indices = knn.kneighbors(query_vectors)
                    recall = calculate_recall(diskann_neighbors, knn_indices, k)
                    self.assertTrue(
                        recall > 0.70,
                        f"Recall [{recall}] was not over 0.7",
                    )
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
        ) in self._test_matrix_with_pq:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.AsyncDiskIndex(
                    index_directory=ann_dir,
                    num_threads=1,
                    use_pq=True,
                    build_pq_bytes=5,
                    use_opq=False,
                )

                k = 5
                batch_response = index.batch_search(
                    query_vectors,
                    k_value=k,
                    l_value=5,
                    num_threads=8,
                    use_full_precision_to_search=False,
                )
                self.assertIsInstance(batch_response, dap.QueryResponseBatchWithStats)

                diskann_neighbors, diskann_distances, diskann_stats = batch_response
                if metric in ["l2", "cosine"]:
                    knn = NearestNeighbors(
                        n_neighbors=100, algorithm="auto", metric=metric
                    )
                    knn.fit(index_vectors)
                    knn_distances, knn_indices = knn.kneighbors(query_vectors)
                    recall = calculate_recall(diskann_neighbors, knn_indices, k)
                    # Low bar for recall since we're using PQ
                    self.assertTrue(
                        recall > PQ_RECALL_CUTOFF,
                        f"Recall [{recall}] was not over {PQ_RECALL_CUTOFF} for metric {metric} and datatype {dtype}",
                    )

    def test_batch_range_search(self):
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
        ) in self._test_matrix:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.AsyncDiskIndex(
                    index_directory=ann_dir,
                    num_threads=1,
                )
                batch_response = index.batch_range_search(
                    query_vectors,
                    starting_l_value=5,
                    radius=1.0,
                    num_threads=1,
                )
                self.assertIsInstance(batch_response, dap.RangeQueryResponseBatchWithStats)

                diskann_lims, diskann_neighbors, diskann_distances, diskann_stats = batch_response
                print("DiskANN Range Search Results, full precision:", np.sum(diskann_lims))
        for (
            metric,
            dtype,
            query_vectors,
            index_vectors,
            ann_dir,
            vector_bin_file,
        ) in self._test_matrix_with_pq:
            with self.subTest(msg=f"Testing dtype {dtype}"):
                index = dap.AsyncDiskIndex(
                    index_directory=ann_dir,
                    num_threads=1,
                    use_pq=True,
                    build_pq_bytes=5,
                    use_opq=False,
                )

                batch_response = index.batch_range_search(
                    query_vectors,
                    starting_l_value=5,
                    radius=1.0,
                    num_threads=8,
                )
                self.assertIsInstance(batch_response, dap.RangeQueryResponseBatchWithStats)

                diskann_lims, diskann_neighbors, diskann_distances, diskann_stats = batch_response

                print("DiskANN Range Search Results, quantized:", np.sum(diskann_lims))
                



    def test_insert_and_delete(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix_with_pq:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        build_pq_bytes=5,
                        use_pq=True,
                        load_from_file=False,
                        max_points=10000,
                        dimensions=100,
                    )
                    insert_vectors: np.ndarray = random_vectors(5, 10, dtype=dtype)
                    # Test insert 5 random vectors
                    for vector_id, vector in enumerate(insert_vectors):
                        index.insert(vector_id, vector, False)
                    # Test soft delete for vector 1...10000
                    index.mark_deleted(np.array(list(range(1,10000)), dtype=np.uint32))
                    # Test hard delete for vector 1...10000
                    index.consolidate_deletes(num_threads=2)
                    # Test batch_insert
                    vectors = np.array(insert_vectors, dtype=dtype)
                    index.batch_insert(np.array(list(range(1,6)), dtype=np.uint32), vectors, False)
        
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        num_threads=1,
                        load_from_file=False,
                        max_points=10000,
                        dimensions=100,
                    )
                    insert_vectors: np.ndarray = random_vectors(5, 10, dtype=dtype)
                    # Test insert 5 random vectors
                    for vector_id, vector in enumerate(insert_vectors):
                        index.insert(vector_id, vector, True)
                    # Test soft delete for vector 1...10000
                    index.mark_deleted(np.array(list(range(1,10000)), dtype=np.uint32))
                    # Test hard delete for vector 1...10000
                    index.consolidate_deletes(num_threads=2)
                    # Test batch_insert
                    vectors = np.array(insert_vectors, dtype=dtype)
                    index.batch_insert(np.array(list(range(1,6)), dtype=np.uint32), vectors, True)

    def test_insert_and_inplace_delete(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix_with_pq:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        build_pq_bytes=5,
                        use_pq=True,
                        load_from_file=False,
                        max_points=10000,
                        dimensions=100,
                    )
                    insert_vectors: np.ndarray = random_vectors(5, 10, dtype=dtype)
                    # Test insert 5 random vectors
                    for vector_id, vector in enumerate(insert_vectors):
                        index.insert(vector_id, vector, False)
                    # Test soft delete for vector 1...10000
                    index.batch_inplace_delete(np.array(list(range(1,10000)), dtype=np.uint32), False)
                    # Test hard delete for vector 1...10000
                    index.consolidate_simple(num_threads=2)
                    # Test batch_insert
                    vectors = np.array(insert_vectors, dtype=dtype)
                    index.batch_insert(np.array(list(range(1,6)), dtype=np.uint32), vectors, False)
        
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        num_threads=1,
                        load_from_file=False,
                        max_points=10000,
                        dimensions=100,
                    )
                    insert_vectors: np.ndarray = random_vectors(5, 10, dtype=dtype)
                    # Test insert 5 random vectors
                    for vector_id, vector in enumerate(insert_vectors):
                        index.insert(vector_id, vector, True)
                    # Test soft delete for vector 1...10000
                    index.batch_inplace_delete(np.array(list(range(1,10000)), dtype=np.uint32), True, 2)
                    # Test hard delete for vector 1...10000
                    index.consolidate_simple(num_threads=2)
                    # Test batch_insert
                    vectors = np.array(insert_vectors, dtype=dtype)
                    index.batch_insert(np.array(list(range(1,6)), dtype=np.uint32), vectors, True)

    def test_insert_and_multi_inplace_delete(self):
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix_with_pq:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        build_pq_bytes=5,
                        use_pq=True,
                        load_from_file=False,
                        max_points=10000,
                        dimensions=100,
                    )
                    insert_vectors: np.ndarray = random_vectors(5, 10, dtype=dtype)
                    # Test insert 5 random vectors
                    for vector_id, vector in enumerate(insert_vectors):
                        index.insert(vector_id, vector, False)
                    # Test soft delete for vector 1...10000
                    index.multi_inplace_delete(np.array(list(range(1,10000)), dtype=np.uint32), False)
                    # Test hard delete for vector 1...10000
                    index.consolidate_simple(num_threads=2)
                    # Test batch_insert
                    vectors = np.array(insert_vectors, dtype=dtype)
                    index.batch_insert(np.array(list(range(1,6)), dtype=np.uint32), vectors, False)
        
        for (
                metric,
                dtype,
                query_vectors,
                index_vectors,
                ann_dir,
                vector_bin_file,
            ) in self._test_matrix:
                with self.subTest(msg=f"Testing dtype {dtype}"):
                    index = dap.AsyncDiskIndex(
                        index_directory=ann_dir,
                        num_threads=1,
                        load_from_file=False,
                        max_points=10000,
                        dimensions=100,
                    )
                    insert_vectors: np.ndarray = random_vectors(5, 10, dtype=dtype)
                    # Test insert 5 random vectors
                    for vector_id, vector in enumerate(insert_vectors):
                        index.insert(vector_id, vector, True)
                    # Test soft delete for vector 1...10000
                    index.multi_inplace_delete(np.array(list(range(1,10000)), dtype=np.uint32), True, 2)
                    # Test hard delete for vector 1...10000
                    index.consolidate_simple(num_threads=2)
                    # Test batch_insert
                    vectors = np.array(insert_vectors, dtype=dtype)
                    index.batch_insert(np.array(list(range(1,6)), dtype=np.uint32), vectors, True)


    @classmethod
    def tearDownClass(cls) -> None:
        for test in cls._test_matrix:
            try:
                ann_dir = test[4]
                shutil.rmtree(ann_dir, ignore_errors=True)
            except:
                pass