# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from typing import Optional
import numpy as np
import json

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

__ALL__ = ["BfTreeIndex"]

class BfTreeIndex:
    """
    A BfTreeIndex is an in-mem DiskANN index using the BFTree data provider
    """

    def __init__(
        self,
        data_path: Optional[str],
        dimensions: int,
        num_threads: int = 1,
        r: int = 64,
        l: int = 100,
        alpha: float = 1.2,
        graph_slack_factor: float = 1.3,
        metric: DistanceMetric = "l2",
        vector_dtype: VectorDType = np.float32,
        create_empty_bftree: bool = False,
        max_points: int = 10_000,
        build_pq_bytes: int = 0,
        max_fp_vecs_per_fill: Optional[int] = None,
        pq_seed: int = 42,
        on_disk_prefix: Optional[str] = None,
    ):
        """
        ### Parameters
        - **data_path**: Path to the data file to build the index from.
        - **num_threads**: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system.
        - **r**: The number of neighbors to consider for each node during index construction.
        - **l**: The size of the search list for each node during index construction.
        - **alpha**: The alpha pruning factor
        - **graph_slack_factor**: The graph slack factor
        - **metric**: A `str`, strictly one of {"l2", "cosine", "cosinenormalized", "innerproduct"}.
        - **vector_dtype**: The vector dtype this index has been built with.
        - **dimensions**: The dimensionality of the vectors in the index. 
        """
        assert not (data_path is not None and create_empty_bftree), "data_path must be None when create_empty_bftree is True"
        dap_metric = _valid_metric(metric)

        self._num_threads = num_threads
        self._distance_metric = metric
        self._vector_dtype = vector_dtype
        self._dimensions = dimensions

        # Initialize the Rust AsyncDiskIndex
        if vector_dtype == np.uint8:
            _index = _native_dap.BfTreeIndexU8
        elif vector_dtype == np.int8:
            _index = _native_dap.BfTreeIndexInt8
        else:
            _index = _native_dap.BfTreeIndexF32

        dap_metric = _valid_metric(metric)

        _assert_is_positive_uint32(max_points, "max_points")

        self._index = _index(
            metric=dap_metric,
            data_path=data_path if data_path else "",
            r=r,
            l=l,
            alpha=alpha,
            num_start_pts=1,  
            num_threads=num_threads,
            graph_slack_factor=graph_slack_factor,
            backedge_ratio=1.0,
            num_tasks=num_threads,
            insert_minibatch_size=1,
            create_empty_index=create_empty_bftree,
            max_points=max_points,
            dim=dimensions,
            build_pq_bytes=build_pq_bytes,
            max_fp_vecs_per_fill=max_fp_vecs_per_fill,
            pq_seed=pq_seed,
            on_disk_prefix=on_disk_prefix,
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
            num_threads: int = 1,
            starting_l_value = 5,
            radius: float = 1.0,
    ) -> RangeQueryResponseBatchWithStats:
        """
        Performs range search on the index by a batch of query vectors.

        Returns all neighbors within the specified radius for each query.
        """

        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype)
        _assert(len(_queries.shape) == 2, "queries must must be 2-d np array")
        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_queries.shape[1]}",
        )
        _assert(radius > 0, "radius must be positive")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_positive_uint32(starting_l_value, "starting_l_value")

        _range_search = self._index.batch_range_search
        result = _range_search(
            queries=_queries,
            num_threads=num_threads,
            starting_l_value=starting_l_value,
            radius=radius,    
        )

        lims = result.lims
        neighbors = result.ids
        distances = result.distances
        lims, neighbors, distances = np.array(lims), np.array(neighbors), np.array(distances)
        return RangeQueryResponseBatchWithStats(
            lims=lims,
            identifiers=neighbors, 
            distances=distances, 
            search_stats=result.search_stats
        )
    
    def insert(self, vector: VectorLike, identifier: int) -> None:
        """
        Inserts a single vector into the index.

        ### Parameters
        - **vector**: 1d numpy array of the same dimensionality and dtype of the index.
        - **identifier**: The identifier for this vector. Must be a non-negative integer.
        """
        _vector = _castable_dtype_or_raise(vector, expected=self._vector_dtype)
        _assert(len(_vector.shape) == 1, "vector must be 1-d")
        _assert(
            _vector.shape[0] == self._dimensions,
            f"vector must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"vector dimensionality: {_vector.shape[0]}",
        )
        _assert_is_nonnegative_uint32(identifier, "identifier")
        
        self._index.insert(vector=_vector, id=identifier)


    def batch_insert(self, vectors: VectorLikeBatch, identifiers: np.ndarray, num_tasks: int) -> None:
        """
        Inserts a batch of vectors into the index.

        ### Parameters
        - **vectors**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of vectors intended to insert in parallel. Dtype must match dtype of the index.
        - **identifiers**: 1d numpy array of non-negative integers, with length equal to the number of vectors.
        """
        _vectors = _castable_dtype_or_raise(vectors, expected=self._vector_dtype)
        _assert(len(_vectors.shape) == 2, "vectors must must be 2-d np array")
        _assert(
            _vectors.shape[1] == self._dimensions,
            f"vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"vector dimensionality: {_vectors.shape[1]}",
        )
        _assert(
            len(identifiers.shape) == 1,
            "identifiers must be 1-d np array",
        )
        _assert(
            identifiers.shape[0] == _vectors.shape[0],
            f"number of identifiers must match number of vectors; number of vectors: {_vectors.shape[0]}, "
            f"number of identifiers: {identifiers.shape[0]}",
        )
        for identifier in identifiers:
            _assert_is_nonnegative_uint32(identifier, "identifier")

        _assert_is_positive_uint32(num_tasks, "num_tasks")

        self._index.batch_insert(vectors=_vectors, vector_ids=identifiers, num_tasks=num_tasks)


    def multi_insert(self, vectors: VectorLikeBatch, identifiers: np.ndarray) -> None:
        """
        Inserts multiple vectors into the index.

        ### Parameters
        - **vectors**: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
          number of vectors intended to insert. Dtype must match dtype of the index.
        - **identifiers**: 1d numpy array of non-negative integers, with length equal to the number of vectors.
        """
        _vectors = _castable_dtype_or_raise(vectors, expected=self._vector_dtype)
        _assert(len(_vectors.shape) == 2, "vectors must must be 2-d np array")
        _assert(
            _vectors.shape[1] == self._dimensions,
            f"vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"vector dimensionality: {_vectors.shape[1]}",
        )
        _assert(
            len(identifiers.shape) == 1,
            "identifiers must be 1-d np array",
        )
        _assert(
            identifiers.shape[0] == _vectors.shape[0],
            f"number of identifiers must match number of vectors; number of vectors: {_vectors.shape[0]}, "
            f"number of identifiers: {identifiers.shape[0]}",
        )
        for identifier in identifiers:
            _assert_is_nonnegative_uint32(identifier, "identifier")

        self._index.multi_insert(vectors=_vectors, vector_ids=identifiers)

    def mark_deleted(self, vector_ids: np.ndarray) -> None:
        """
        Marks multiple vectors as deleted in the index.

        ### Parameters
        - **vector_ids**: 1d numpy array of non-negative integers representing the identifiers of vectors to mark as deleted.
        """
        _assert(
            len(vector_ids.shape) == 1,
            "vector_ids must be 1-d np array",
        )
        for vector_id in vector_ids:
            _assert_is_nonnegative_uint32(vector_id, "vector_id")
        
        # Ensure the array is uint32 type for Rust binding
        _vector_ids = np.array(vector_ids, dtype=np.uint32)
        self._index.mark_deleted(_vector_ids)

    def consolidate_deletes(self, num_tasks: int) -> None:
        """
        Consolidates the deletions marked in the index.

        ### Parameters
        - **num_tasks**: Number of parallel tasks to use for consolidation. Must be > 0.
        """
        _assert_is_positive_uint32(num_tasks, "num_tasks")
        self._index.consolidate_deletes(num_tasks)

    def get_neighbors(self, vector_id: int) -> np.ndarray:
        """
        Retrieves the neighbors of a given vector in the index.

        ### Parameters
        - **vector_id**: The identifier of the vector whose neighbors are to be retrieved. Must be a non-negative integer.

        ### Returns
        - A numpy array of neighbor identifiers.
        """
        _assert_is_nonnegative_uint32(vector_id, "vector_id")
        neighbors = self._index.get_neighbors(vector_id)
        return np.array(neighbors)

    def consolidate_simple(self, num_tasks: int) -> None:
        """
        Consolidates the index using a simple method.

        ### Parameters
        - **num_tasks**: Number of parallel tasks to use for consolidation. Must be > 0.
        """
        _assert_is_positive_uint32(num_tasks, "num_tasks")
        self._index.consolidate_simple(num_tasks)


    def multi_inplace_delete(
        self, 
        vector_ids: np.ndarray,
        k_value: int = 50, 
        l_value: int = 100, 
        num_to_replace: int = 3,
        delete_method: int = 1,
    ) -> None:  
        """
        Performs in-place deletion of multiple vectors from the index.

        ### Parameters
        - **vector_ids**: 1d numpy array of non-negative integers representing the identifiers of vectors to delete.
        - **k_value**: Number of neighbors to consider during deletion. Must be > 0. Default is 50.
        - **l_value**: Size of the candidate list during deletion. Must be > 0. Default is 100.
        - **num_to_replace**: Number of replacement points for each deleted edge. Must be > 0. Default is 3.
        - **delete_method**: Method to use for deletion. Must be > 0. Default is 1.
        """
        _assert(
            len(vector_ids.shape) == 1,
            "vector_ids must be 1-d np array",
        )
        for vector_id in vector_ids:
            _assert_is_nonnegative_uint32(vector_id, "vector_id")
        
        _assert_is_positive_uint32(k_value, "k_value")
        _assert_is_positive_uint32(l_value, "l_value")
        _assert_is_nonnegative_uint32(delete_method, "delete_method")
        
        _vector_ids = np.array(vector_ids, dtype=np.uint32)
        self._index.multi_inplace_delete(
            vector_ids=_vector_ids,
            k_value=k_value,
            l_value=l_value,
            num_to_replace=num_to_replace,
            delete_method=delete_method
        )
        
    def batch_inplace_delete(
        self, 
        vector_ids: np.ndarray,
        num_tasks: int,
        k_value: int = 50, 
        l_value: int = 100, 
        num_to_replace: int = 3,
        delete_method: int = 1,
    ) -> None:  
        """
        Performs in-place deletion of multiple vectors from the index in batches.

        ### Parameters
        - **vector_ids**: 1d numpy array of non-negative integers representing the identifiers of vectors to delete.
        - **num_tasks**: Number of parallel tasks to use for deletion. Must be > 0.
        - **k_value**: Number of neighbors to consider during deletion. Must be > 0. Default is 50.
        - **l_value**: Size of the candidate list during deletion. Must be > 0. Default is 100.
        - **num_to_replace**: Number of replacement points for each deleted edge. Must be > 0. Default is 3.
        - **delete_method**: Method to use for deletion. Must be > 0. Default is 1.
        """
        _assert(
            len(vector_ids.shape) == 1,
            "vector_ids must be 1-d np array",
        )
        for vector_id in vector_ids:
            _assert_is_nonnegative_uint32(vector_id, "vector_id")
        
        _assert_is_positive_uint32(num_tasks, "num_tasks")
        _assert_is_positive_uint32(k_value, "k_value")
        _assert_is_positive_uint32(l_value, "l_value")
        _assert_is_nonnegative_uint32(delete_method, "delete_method")
        
        _vector_ids = np.array(vector_ids, dtype=np.uint32)
        self._index.batch_inplace_delete(
            vector_ids=_vector_ids,
            num_tasks=num_tasks,
            k_value=k_value,
            l_value=l_value,
            num_to_replace=num_to_replace,
            delete_method=delete_method
        )

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

    def save(self, prefix: str) -> None:
        """
        Save the index to disk using the given file path prefix.

        Files will be created with the prefix followed by suffixes like
        `_params.json`, `_vectors.bftree`, `_neighbors.bftree`, etc.

        ### Parameters
        - **prefix**: The file path prefix for saved index files.
        """
        self._index.save(prefix)

    @classmethod
    def load(
        cls,
        prefix: str,
    ) -> "BfTreeIndex":
        """
        Load a previously saved BfTreeIndex from disk.

        ### Parameters
        - **prefix**: The file path prefix used when the index was saved.
        """
        # Read the params JSON to recover metadata
        params_file = f"{prefix}_params.json"
        with open(params_file, "r") as f:
            params = json.load(f)

        # Read vector_dtype from persisted graph_params
        gp = params["graph_params"]
        dtype_map = {"f32": np.float32, "u8": np.uint8, "i8": np.int8}
        vector_dtype = dtype_map[gp["vector_dtype"]]

        if vector_dtype == np.uint8:
            _index_cls = _native_dap.BfTreeIndexU8
        elif vector_dtype == np.int8:
            _index_cls = _native_dap.BfTreeIndexInt8
        else:
            _index_cls = _native_dap.BfTreeIndexF32

        native_index = _index_cls.load(
            prefix=prefix,
        )

        metric = _valid_metric(params["metric"])
        _assert_is_positive_uint32(params["dim"], "dim")

        instance = cls.__new__(cls)
        instance._index = native_index
        instance._num_threads = 1
        instance._distance_metric = metric
        instance._vector_dtype = vector_dtype
        instance._dimensions = params["dim"]
        return instance


