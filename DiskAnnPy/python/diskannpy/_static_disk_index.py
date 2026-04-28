# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Optional

import numpy as np

from . import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatchWithStats,
    VectorDType,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_2d,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_index_prefix,
    _valid_metric,
    valid_dtype,
    _valid_pq_params,
)
from ._files import vectors_to_file, delete_file
from ._defaults import *

__ALL__ = ["StaticDiskIndex"]


class StaticDiskIndex:
    """
    A StaticDiskIndex is an immutable on-disk DiskANN index.
    """

    def __init__(
        self,
        index_directory: str,
        num_threads: int,
        beam_width: int,
        search_io_limit: int = (1 << 32) - 1,
        num_nodes_to_cache: int = 0,
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
        index_prefix: str = "ann",
    ):
        """
        ### Parameters
        - **index_directory**: Path to the index directory.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system.
        - **beam_width**: The beam width used during search.
        - **search_io_limit**: The maximum number of I/O operations allowed during a search. The default value u32::MAX 4294967295
        - **num_nodes_to_cache**: Number of BFS nodes around medoid(s) to cache during query warm up.
        - **distance_metric**: A `str`, strictly one of {"l2", "cosine", "cosinenormalized", "innerproduct"}.
        - **vector_dtype**: The vector dtype this index has been built with.
        - **dimensions**: The vector dimensionality of this index. All new vectors inserted must be the same
          dimensionality. **This value is only used if a `{index_prefix}_metadata.bin` file does not exist.** If it
          does not exist, you are required to provide it.
        - **index_prefix**: The prefix of the index files. Defaults to "ann".
        """
        self._num_threads = num_threads
        self._beam_width = beam_width
        self._search_io_limit = search_io_limit
        self._num_nodes_to_cache = num_nodes_to_cache

        index_path_prefix = _valid_index_prefix(index_directory, index_prefix)
        vector_dtype, metric, _, dimensions = _ensure_index_metadata(
            index_path_prefix,
            vector_dtype,
            distance_metric,
            1,  # it doesn't matter because we don't need it in this context anyway
            dimensions,
        )
        dap_metric = _valid_metric(metric)

        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(num_nodes_to_cache, "num_nodes_to_cache")
        _assert_is_positive_uint32(beam_width, "beam_width")

        self._vector_dtype = vector_dtype
        self._dimensions = dimensions

        # Initialize the Rust StaticDiskIndex
        if vector_dtype == np.uint8:
            _index = _native_dap.StaticDiskIndexU8
        elif vector_dtype == np.int8:
            _index = _native_dap.StaticDiskIndexInt8
        else:
            _index = _native_dap.StaticDiskIndexF32

        self._index = _index(
            dist_fn=dap_metric,
            index_path_prefix=index_path_prefix,
            beam_width=beam_width,
            search_io_limit=search_io_limit,
            num_threads=num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
        )

    def search(
        self, query: VectorLike, k_neighbors: int, complexity: int
    ) -> QueryResponse:
        """
        Searches the index by a single query vector.

        ### Parameters
        - **query**: 1d numpy array of the same dimensionality and dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. Must be at least k_neighbors in size.
        """
        _query = _castable_dtype_or_raise(query, expected=self._vector_dtype)
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert(
            _query.shape[0] == self._dimensions,
            f"query vector must have the same dimensionality as the index; index dimensionality: {self._dimensions}, query dimensionality: {_query.shape[0]}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_nonnegative_uint32(complexity, "complexity")
        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        result = self._index.search(
            query=_query, recall_at=k_neighbors, l_value=complexity)
        neighbors = result.ids
        distances = result.distances
        neighbors, distances = np.array(neighbors), np.array(distances)
        return QueryResponse(identifiers=neighbors, distances=distances)

    def batch_search(
        self,
        queries: VectorLikeBatch,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
    ) -> QueryResponseBatchWithStats:
        """
        Searches the index by a batch of query vectors.

        This search is parallelized and far more efficient than searching for each vector individually.

        ### Parameters
        - **queries**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of queries intended to search for in parallel. Dtype must match dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
         - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        """

        _queries = _castable_dtype_or_raise(
            queries, expected=self._vector_dtype)
        _assert_2d(_queries, "queries")

        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, query dimensionality: {_queries.shape[1]}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        
        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        result = self._index.batch_search(
            queries=_queries, recall_at=k_neighbors, l_value=complexity, num_threads=num_threads)
        neighbors = result.ids
        distances = result.distances
        neighbors, distances = np.array(neighbors), np.array(distances)
        return QueryResponseBatchWithStats(identifiers=neighbors, distances=distances,search_stats=result.search_stats)
