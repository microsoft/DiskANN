# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Optional
import numpy as np

from . import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatchWithStats,
    RangeQueryResponseBatchWithStats,
    VectorDType,
    VectorLike,
    VectorLikeBatch,
)
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
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

__ALL__ = ["AsyncDiskIndex"]

class AsyncDiskIndex:
    """
    An AsyncDiskIndex is an immutable in-mem DiskANN index that supports asynchronous operations.
    """

    def __init__(
        self,
        index_directory: str,
        num_threads: int = 1,
        r: int = 64,
        l: int = 100,
        alpha: float = 1.2,
        build_pq_bytes: int = 0,
        graph_slack_factor: float = 1.3,
        index_prefix: str = "ann",
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
        use_pq: bool = False,
        use_opq: bool = False,
        load_from_file: bool = True,
        max_points: int = 10000,
    ):
        """
        ### Parameters
        - **index_path_prefix**: Path to the index file.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system.
        - **beam_width**: The beam width used during search.
        - **distance_metric**: A `str`, strictly one of {"l2", "cosine", "cosinenormalized", "innerproduct"}.
        - **vector_dtype**: The vector dtype this index has been built with.
        """

        _valid_pq_params(use_pq, build_pq_bytes, use_opq)
        dim = 100 if dimensions is None else dimensions
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)
        self._index_directory = index_directory
        vector_dtype, metric, _, dimensions = _ensure_index_metadata(
            index_prefix_path,
            vector_dtype,
            distance_metric,
            1,  # it doesn't matter because we don't need it in this context anyway
            dimensions,
        )
        dap_metric = _valid_metric(metric)

        self._num_threads = num_threads
        self._distance_metric = distance_metric
        self._vector_dtype = vector_dtype
        self._dimensions = dimensions

        # Initialize the Rust AsyncDiskIndex
        if vector_dtype == np.uint8:
            _index = _native_dap.AsyncMemoryIndexU8
        elif vector_dtype == np.int8:
            _index = _native_dap.AsyncMemoryIndexInt8
        else:
            _index = _native_dap.AsyncMemoryIndexF32

        dap_metric = _valid_metric(metric)

        self._index = _index(
            metric=dap_metric,
            index_path=index_prefix_path,
            r=r,
            l=l,
            alpha=alpha,
            num_threads=num_threads,
            build_pq_bytes=build_pq_bytes,
            graph_slack_factor=graph_slack_factor,
            load_from_file=load_from_file,
            max_points=max_points,
            dim=dim,
        )

    def search(
        self, query: VectorLike, k_value: int, l_value: int, use_full_precision_to_search: bool = True,
    ) -> QueryResponse:
        """
        Searches the index by a single query vector asynchronously.

        ### Parameters
        - **query**: 1d numpy array of the same dimensionality and dtype of the index.
        - **k_value**: Number of neighbors to be returned. Must be > 0.
        - **l_value**: Size of distance ordered list of candidate neighbors to use while searching. Must be at least k_neighbors in size.
        """
        _query = _castable_dtype_or_raise(query, expected=self._vector_dtype)
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert(
            _query.shape[0] == self._dimensions,
            f"query vector must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_query.shape[0]}",
        )
        _assert_is_positive_uint32(k_value, "k_value")
        _assert_is_nonnegative_uint32(l_value, "l_value")
        if k_value > l_value:
            warnings.warn(
                f"k_neighbors={k_value} asked for, but list_size={l_value} was smaller. Increasing {l_value} to {k_value}"
            )
            l_value = k_value

        result, cmps = self._index.search(query=_query, k_value=k_value, l_value=l_value, use_full_precision_to_search=use_full_precision_to_search)
        neighbors = result.ids
        distances = result.distances
        neighbors, distances = np.array(neighbors), np.array(distances)
        return QueryResponse(identifiers=neighbors, distances=distances)

    def insert(
        self, vector_id: int, vector: VectorLike, use_full_precision_to_search: bool
    ) -> None:
        """
        Inserts a vector into the index asynchronously.

        ### Parameters
        - **vector_id**: ID of the vector to insert.
        - **vector**: The vector to insert.
        - **use_full_precision_to_search**: Whether to use full precision to search.
        """
        _vector = _castable_dtype_or_raise(vector, expected=self._vector_dtype)
        self._index.insert(
            vector_id=vector_id,
            vector=_vector,
            use_full_precision_to_search=use_full_precision_to_search,
        )

    def batch_insert(
        self, vector_ids: VectorLike, vectors: VectorLikeBatch, use_full_precision_to_search: bool, num_threads: int = 1,
    ) -> None:
        """
        Inserts a batch of vectors

        ### Parameters
        - **vector_ids**: IDs of the vectors to insert.
        - **vectors**: The vectors to insert.
        - **use_full_precision_to_search**: Whether to use full precision to search.
        - **num_threads**: num threads to use for search
        """

        _vectors = _castable_dtype_or_raise(vectors, expected=self._vector_dtype)
        _assert(len(vectors.shape) == 2, "vectors must be a 2-d array")
        _vectors = _vectors.astype(dtype=self._vector_dtype, casting="safe", copy=False)
        _vector_ids = vector_ids.astype(dtype=np.uint32, casting="safe", copy=False)
        self._index.batch_insert(_vector_ids, _vectors, use_full_precision_to_search, num_threads)   

    def batch_inplace_delete(
        self, vector_ids: VectorLike, use_full_precision_to_search: bool, num_threads: int = 1, k_value: int = 50, l_value: int = 100, num_to_replace: int = 3, delete_method: int = 1,
    ) -> None:
        """
        Inplace deletes a batch of vectors

        ### Parameters
        - **vector_ids**: IDs of the vectors to delete
        - **use_full_precision_to_search**: Whether to use full precision to search.
        - **num_threads**: num threads to use for delete
        - **k_value**: k_value to use for inplace delete
        - **l_value**: l_value to use for inplace delete
        - **num_to_replace**: number of points to replace each deleted edge with for inplace delete
        """

        _vector_ids = vector_ids.astype(dtype=np.uint32, casting="safe", copy=False)
        self._index.batch_inplace_delete(_vector_ids, use_full_precision_to_search, num_threads, k_value, l_value, num_to_replace, delete_method)   

    def multi_inplace_delete(
        self, vector_ids: VectorLike, use_full_precision_to_search: bool, k_value: int = 50, l_value: int = 100, num_to_replace: int = 3, delete_method: int = 1,
    ) -> None:  
        """
        Inplace deletes a batch of vectors using multi-inplace delete. It will use the configured max_minibatch_par 

        ### Parameters
        - **vector_ids**: IDs of the vectors to delete
        - **use_full_precision_to_search**: Whether to use full precision to search.
        - **k_value**: k_value to use for inplace delete
        - **l_value**: l_value to use for inplace delete
        - **num_to_replace**: number of points to replace each deleted edge with for inplace delete
        """

        _vector_ids = vector_ids.astype(dtype=np.uint32, casting="safe", copy=False)
        self._index.multi_inplace_delete(_vector_ids, use_full_precision_to_search, k_value, l_value, num_to_replace, delete_method)


    def batch_search(
            self,
            queries: VectorLikeBatch,
            k_value: int,
            l_value: int,
            num_threads: int = 1,
            use_full_precision_to_search: bool = True,
    ) -> QueryResponseBatchWithStats:
        """
        Searches the index by a batch of query vectors.

        This search is parallelized and far more efficient than searching for each vector individually.

        ### Parameters
        - **queries**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of queries intended to search for in parallel. Dtype must match dtype of the index.
        - **k_value**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **l_value**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        """

        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert(len(_queries.shape) == 2, "queries must must be 2-d np array")
        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_queries.shape[1]}",
        )
        _assert_is_positive_uint32(k_value, "k_value")
        _assert_is_positive_uint32(l_value, "l_value")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")

        if k_value > l_value:
            warnings.warn(
                f"k_neighbors={k_value} asked for, but list_size={l_value} was smaller. Increasing {l_value} to {k_value}"
            )
            l_value = k_value

        _search = self._index.batch_search
        result = _search(queries=_queries, num_threads=num_threads, k_value=k_value, l_value=l_value, use_full_precision_to_search=use_full_precision_to_search)
        neighbors = result.ids
        distances = result.distances
        neighbors, distances = np.array(neighbors), np.array(distances)
        return QueryResponseBatchWithStats(identifiers=neighbors, distances=distances, search_stats=result.search_stats)

    def batch_range_search(
            self,
            queries: VectorLikeBatch,
            starting_l_value: int,
            radius: float,
            num_threads: int = 1,
    ) -> RangeQueryResponseBatchWithStats:
        """
        Searches the index by a batch of query vectors.

        This search is parallelized and far more efficient than searching for each vector individually.

        ### Parameters
        - **queries**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of queries intended to search for in parallel. Dtype must match dtype of the index.
        - **l_value**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        - **radius**: Radius for range search 
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        """

        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert(len(_queries.shape) == 2, "queries must must be 2-d np array")
        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_queries.shape[1]}",
        )
        _assert_is_positive_uint32(starting_l_value, "starting_l_value")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")


        result = self._index.batch_range_search(queries=_queries, num_threads=num_threads, starting_l_value=starting_l_value, radius=radius)
        return RangeQueryResponseBatchWithStats(lims = np.array(result.lims), identifiers= np.array(result.ids), distances = np.array(result.distances), search_stats=result.search_stats)

    def mark_deleted(
            self, vector_ids: VectorLike,
    ) -> None:
        """
        Delete a batch of vectors

        ### Parameters
        - **vector_ids**: IDs of the vectors to delete.
        """
        self._index.mark_deleted(vector_ids)


    def consolidate_deletes(
            self, num_threads: int = 1
    ) -> None:
        """
        Delete vectors to be marked as soft-deleted

        ### Parameters
        - **num_threads**: num threads to use for batch deletes.
        """
        self._index.consolidate_deletes(num_threads)

    
    def consolidate_simple(
            self, num_threads: int = 1
    ) -> None:
        """
        Delete vectors to be marked as soft-deleted

        ### Parameters
        - **num_threads**: num threads to use for batch deletes.
        """
        self._index.consolidate_simple(num_threads)

    # useful function for debugging
    def get_neighbors(self, vector_id: int) -> np.ndarray:
        """
        Get neighbors of a vector by its ID.

        ### Parameters
        - **vector_id**: ID of the vector whose neighbors are to be retrieved.
        """
        neighbors = self._index.get_neighbors(vector_id)
        return np.array(neighbors)

    # get the average degree of the index
    # note that depending on the specifics of the provider,
    # this function may not do exactly what you expect--it simply
    # computes the average degree across all ids in the dataset iterator,
    # which might NOT all actually be active at a time, might include soft-deleted indices, etc.
    # The user is responsible for interpreting the result correctly
    def get_average_degree(self) -> float:
        """
        Get the average degree of the index.
        """
        average_degree = self._index.get_average_degree()
        return average_degree


    # Add other async methods as needed