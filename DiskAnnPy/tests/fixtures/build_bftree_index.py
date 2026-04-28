# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from tempfile import mkdtemp
import time

import diskannpy as dap
import numpy as np

from .create_test_data import random_vectors

def save_array_as_bin_file(filename, array, dtype):
    array = np.ascontiguousarray(array)
    rows, cols = array.shape
    with open(filename, 'wb') as f:
        np.array([rows, cols], dtype=np.int32).tofile(f)
        array.tofile(f)

def build_random_vectors_and_bftree_index(
    dtype, metric,
    query_vector_count: int = 1000, index_vector_count: int = 10000, seed: int = 12345
):
    query_vectors: np.ndarray = random_vectors(query_vector_count, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(index_vector_count, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    # save the index_vectors to a temporary file
    index_vectors_file = os.path.join(ann_dir, "index_vectors.bin")
    save_array_as_bin_file(index_vectors_file, index_vectors, dtype)

    index = dap.BfTreeIndex(
        data_path=index_vectors_file,
        dimensions=10,
        num_threads=1,
        r=64,
        l=100,
        alpha=1.2,
        graph_slack_factor=1.3,
        metric=metric,
        vector_dtype=dtype,
    )
    
    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        os.path.join(ann_dir, "vectors.bin"),
        index,
    )

def build_random_vectors_and_bftree_index_pq(
    dtype, metric,
    query_vector_count: int = 1000, index_vector_count: int = 10000, seed: int = 12345, num_pq_bytes: int = 8
):
    query_vectors: np.ndarray = random_vectors(query_vector_count, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(index_vector_count, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    # save the index_vectors to a temporary file
    index_vectors_file = os.path.join(ann_dir, "index_vectors.bin")
    save_array_as_bin_file(index_vectors_file, index_vectors, dtype)

    index = dap.BfTreeIndex(
        data_path=index_vectors_file,
        dimensions=10,
        num_threads=1,
        r=64,
        l=100,
        alpha=1.2,
        graph_slack_factor=1.3,
        metric=metric,
        vector_dtype=dtype,
        build_pq_bytes=num_pq_bytes,
        max_fp_vecs_per_fill=None,
        pq_seed=seed,
    )
    
    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        os.path.join(ann_dir, "vectors.bin"),
        index,
    )

def build_empty_bftree_index(dtype, metric, max_points: int = 20, number_of_query_vectors: int = 100):
    index = dap.BfTreeIndex(
        data_path=None,
        dimensions=10,
        num_threads=1,
        r=64,
        l=100,
        alpha=1.2,
        graph_slack_factor=1.3,
        metric=metric,
        vector_dtype=dtype,
        create_empty_bftree=True,
        max_points=max_points,
    )
    query_vectors: np.ndarray = random_vectors(number_of_query_vectors, 10, dtype=dtype, seed=12345)
    
    return (
        query_vectors,
        metric,
        dtype,
        index,
    )

def build_on_disk_bftree_index(
    dtype, metric,
    query_vector_count: int = 1000, index_vector_count: int = 10000, seed: int = 12345
):
    """Build a BfTree index with on-disk (file-backed) BfTree storage.

    Unlike build_random_vectors_and_bftree_index which uses in-memory BfTrees,
    this creates file-backed BfTrees that support snapshot/save operations.
    """
    query_vectors: np.ndarray = random_vectors(query_vector_count, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(index_vector_count, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    # save the index_vectors to a temporary file
    index_vectors_file = os.path.join(ann_dir, "index_vectors.bin")
    save_array_as_bin_file(index_vectors_file, index_vectors, dtype)

    # Use on_disk_prefix to create file-backed BfTrees
    on_disk_prefix = os.path.join(ann_dir, "bftree_index")

    index = dap.BfTreeIndex(
        data_path=index_vectors_file,
        dimensions=10,
        num_threads=1,
        r=64,
        l=100,
        alpha=1.2,
        graph_slack_factor=1.3,
        metric=metric,
        vector_dtype=dtype,
        on_disk_prefix=on_disk_prefix,
    )
    
    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        on_disk_prefix,
        index,
    )

def build_on_disk_bftree_index_pq(
    dtype, metric,
    query_vector_count: int = 100, index_vector_count: int = 200, seed: int = 12345, num_pq_bytes: int = 8
):
    """Build a PQ BfTree index with on-disk (file-backed) BfTree storage."""
    query_vectors: np.ndarray = random_vectors(query_vector_count, 10, dtype=dtype, seed=seed)
    index_vectors: np.ndarray = random_vectors(index_vector_count, 10, dtype=dtype, seed=seed)
    ann_dir = mkdtemp()

    index_vectors_file = os.path.join(ann_dir, "index_vectors.bin")
    save_array_as_bin_file(index_vectors_file, index_vectors, dtype)

    on_disk_prefix = os.path.join(ann_dir, "bftree_pq_index")

    index = dap.BfTreeIndex(
        data_path=index_vectors_file,
        dimensions=10,
        num_threads=1,
        r=64,
        l=100,
        alpha=1.2,
        graph_slack_factor=1.3,
        metric=metric,
        vector_dtype=dtype,
        build_pq_bytes=num_pq_bytes,
        max_fp_vecs_per_fill=None,
        pq_seed=seed,
        on_disk_prefix=on_disk_prefix,
    )

    return (
        metric,
        dtype,
        query_vectors,
        index_vectors,
        ann_dir,
        on_disk_prefix,
        index,
    )