# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import warnings

import numpy as np

from pathlib import Path
from typing import Optional

from . import _diskannpy as _native_dap
from ._common import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatch,
    VectorDType,
    VectorIdentifier,
    VectorIdentifierBatch,
    VectorLike,
    VectorLikeBatch,
    _assert,
    _assert_2d,
    _assert_dtype,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _ensure_index_metadata,
    _valid_metric,
    _valid_index_prefix,
)
from ._diskannpy import defaults

__ALL__ = ["DynamicMemoryIndex"]


class DynamicMemoryIndex:

    @classmethod
    def from_file(
        cls,
        index_directory: str,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True,
        index_prefix: str = "ann",
        distance_metric: Optional[DistanceMetric] = None,
        vector_dtype: Optional[VectorDType] = None,
        dimensions: Optional[int] = None,
    ) -> "DynamicMemoryIndex":
        index_prefix_path = _valid_index_prefix(index_directory, index_prefix)

        # do tags exist?
        tags_file = index_prefix_path + ".tags"
        _assert(Path(tags_file).exists(), f"The file {tags_file} does not exist in {index_directory}")
        vector_dtype, dap_metric, num_vectors, dimensions = _ensure_index_metadata(
            index_prefix_path,
            vector_dtype,
            distance_metric,
            max_vectors,
            dimensions
        )

        index = cls(
            distance_metric=dap_metric,  # type: ignore
            vector_dtype=vector_dtype,
            dimensions=dimensions,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation
        )
        index._index.load(index_prefix_path)
        return index

    def __init__(
        self,
        distance_metric: DistanceMetric,
        vector_dtype: VectorDType,
        dimensions: int,
        max_vectors: int,
        complexity: int,
        graph_degree: int,
        saturate_graph: bool = defaults.SATURATE_GRAPH,
        max_occlusion_size: int = defaults.MAX_OCCLUSION_SIZE,
        alpha: float = defaults.ALPHA,
        num_threads: int = defaults.NUM_THREADS,
        filter_complexity: int = defaults.FILTER_COMPLEXITY,
        num_frozen_points: int = defaults.NUM_FROZEN_POINTS_DYNAMIC,
        initial_search_complexity: int = 0,
        search_threads: int = 0,
        concurrent_consolidation: bool = True
    ):
        """
        The diskannpy.DynamicMemoryIndex represents our python API into a dynamic DiskANN InMemory Index library.

        This dynamic index is unlike the DiskIndex and StaticMemoryIndex, in that after loading it you can continue
        to insert and delete vectors.

        Deletions are completed lazily, until the user executes `DynamicMemoryIndex.consolidate_deletes()`
        :param distance_metric: If it exists, must be one of {"l2", "mips", "cosine"}. L2 is supported for all 3 vector dtypes,
            but MIPS is only available for single point floating numbers (numpy.single). Default is ``None``.
        :type distance_metric: str
        :param vector_dtype: The vector dtype this index will be exposing.
        :type vector_dtype: Union[Type[numpy.single], Type[numpy.byte], Type[numpy.ubyte]]
        :param dimensions: The vector dimensionality of this index. All new vectors inserted must be the same
            dimensionality.
        :type dimensions: int
        :param max_vectors: Capacity of the data store including space for future insertions
        :type max_vectors: int
        :param graph_degree: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
            result in larger indices and longer indexing times, but better search quality.
        :type graph_degree: int
        :param saturate_graph:
        :type saturate_graph: bool
        :param max_occlusion_size:
        :type max_occlusion_size: int
        :param alpha:
        :type alpha: float
        :param num_threads:
        :type num_threads: int
        :param filter_complexity:
        :type filter_complexity: int
        :param num_frozen_points:
        :type num_frozen_points: int
        :param initial_search_complexity: The working scratch memory allocated is predicated off of
            initial_search_complexity * search_threads. If a larger list_size * num_threads value is
            ultimately provided by the individual action executed in `batch_query` than provided in this constructor,
            the scratch space is extended. If a smaller list_size * num_threads is provided by the action than the
            constructor, the pre-allocated scratch space is used as-is.
        :type initial_search_complexity: int
        :param search_threads: Should be set to the most common batch_query num_threads size. The working
            scratch memory allocated is predicated off of initial_search_list_size * initial_search_threads. If a
            larger list_size * num_threads value is ultimately provided by the individual action executed in
            `batch_query` than provided in this constructor, the scratch space is extended. If a smaller
            list_size * num_threads is provided by the action than the constructor, the pre-allocated scratch space
            is used as-is.
        :type search_threads: int
        :param concurrent_consolidation:
        :type concurrent_consolidation: bool

        """

        dap_metric = _valid_metric(distance_metric)
        _assert_dtype(vector_dtype)
        _assert_is_positive_uint32(dimensions, "dimensions")

        self._vector_dtype = vector_dtype
        self._dimensions = dimensions

        _assert_is_positive_uint32(max_vectors, "max_vectors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(graph_degree, "graph_degree")
        _assert(alpha >= 1, "alpha must be >= 1, and realistically should be kept between [1.0, 2.0)")
        _assert_is_nonnegative_uint32(max_occlusion_size, "max_occlusion_size")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
        _assert_is_nonnegative_uint32(num_frozen_points, "num_frozen_points")
        _assert_is_nonnegative_uint32(
            initial_search_complexity, "initial_search_complexity"
        )
        _assert_is_nonnegative_uint32(search_threads, "search_threads")

        if vector_dtype == np.single:
            _index = _native_dap.DynamicMemoryFloatIndex
        elif vector_dtype == np.ubyte:
            _index = _native_dap.DynamicMemoryUInt8Index
        else:
            _index = _native_dap.DynamicMemoryInt8Index
        self._index = _index(
            distance_metric=dap_metric,
            dimensions=dimensions,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            saturate_graph=saturate_graph,
            max_occlusion_size=max_occlusion_size,
            alpha=alpha,
            num_threads=num_threads,
            filter_complexity=filter_complexity,
            num_frozen_points=num_frozen_points,
            initial_search_complexity=initial_search_complexity,
            search_threads=search_threads,
            concurrent_consolidation=concurrent_consolidation
        )

    def search(
        self, query: VectorLike, k_neighbors: int, complexity: int
    ) -> QueryResponse:
        """
        Searches the disk index by a single query vector in a 1d numpy array.

        numpy array dtype must match index.

        :param query: 1d numpy array of the same dimensionality and dtype of the index.
        :type query: VectorLike
        :param k_neighbors: Number of neighbors to be returned. If query vector exists in index, it almost definitely
            will be returned as well, so adjust your ``k_neighbors`` as appropriate. (> 0)
        :type k_neighbors: int
        :param complexity: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type complexity: int
        :return: Returns a tuple of 1-d numpy ndarrays; the first including the indices of the approximate nearest
            neighbors, the second their distances. These are aligned arrays.
        """
        _query = _castable_dtype_or_raise(
            query,
            expected=self._vector_dtype,
            message=f"StaticMemoryIndex expected a query vector of dtype of {self._vector_dtype}"
        )
        _assert(len(_query.shape) == 1, "query vector must be 1-d")
        _assert(
            _query.shape[0] == self._dimensions,
            f"query vector must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_query.shape[0]}"
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_nonnegative_uint32(complexity, "complexity")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors
        return self._index.search(query=_query, knn=k_neighbors, complexity=complexity)

    def batch_search(
        self, queries: VectorLikeBatch, k_neighbors: int, complexity: int, num_threads: int
    ) -> QueryResponseBatch:
        """
        Searches the disk index for many query vectors in a 2d numpy array.

        numpy array dtype must match index.

        This search is parallelized and far more efficient than searching for each vector individually.

        :param queries: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
            number of queries intended to search for in parallel. Dtype must match dtype of the index.
        :type queries: VectorLike
        :param k_neighbors: Number of neighbors to be returned. If query vector exists in index, it almost definitely
            will be returned as well, so adjust your ``k_neighbors`` as appropriate. (> 0)
        :type k_neighbors: int
        :param complexity: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type complexity: int
        :param num_threads: Number of threads to use when searching this index. (>= 0), 0 = num_threads in system
        :type num_threads: int
        :return: Returns a tuple of 2-d numpy ndarrays; each row corresponds to the query vector in the same index,
            and elements in row corresponding from 1..k_neighbors approximate nearest neighbors. The second ndarray
            contains the distances, of the same form: row index will match query index, column index refers to
            1..k_neighbors distance. These are aligned arrays.
        """
        _queries = _castable_dtype_or_raise(queries, expected=self._vector_dtype, message=f"DynamicMemoryIndex expected a query vector of dtype of {self._vector_dtype}")
        _assert_2d(_queries, "queries")
        _assert(
            _queries.shape[1] == self._dimensions,
            f"query vectors must have the same dimensionality as the index; index dimensionality: {self._dimensions}, "
            f"query dimensionality: {_queries.shape[1]}"
        )

        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        num_queries, dim = queries.shape
        return self._index.batch_search(
            queries=_queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
        )

    def save(self, save_path: str, compact_before_save: bool = True):
        """
        Saves this index to file.
        :param save_path: The path to save these index files to.
        :type save_path: str
        :param compact_before_save:
        """
        if save_path == "":
            raise ValueError("save_path cannot be empty")
        self._index.save(save_path=save_path, compact_before_save=compact_before_save)

    def insert(self, vector: VectorLike, vector_id: VectorIdentifier):
        """
        Inserts a single vector into the index with the provided vector_id.
        :param vector: The vector to insert. Note that dtype must match.
        :type vector: VectorLike
        :param vector_id: The vector_id to use for this vector. 
        """
        _vector = _castable_dtype_or_raise(vector, expected=self._vector_dtype, message=f"DynamicMemoryIndex expected a query vector of dtype of {self._vector_dtype}")
        _assert(len(vector.shape) == 1, "insert vector must be 1-d")
        _assert_is_positive_uint32(vector_id, "vector_id")
        return self._index.insert(_vector, np.uintc(vector_id))

    def batch_insert(
        self, vectors: VectorLikeBatch, vector_ids: VectorIdentifierBatch, num_threads: int = 0
    ):
        """
        :param vectors: The 2d numpy array of vectors to insert.
        :type vectors: np.ndarray
        :param vector_ids: The 1d array of vector ids to use. This array must have the same number of elements as
            the vectors array has rows. The dtype of vector_ids must be ``np.uintc`` (or any alias that is your
            platform's equivalent)
        :param num_threads: Number of threads to use when inserting into this index. (>= 0), 0 = num_threads in system
        :type num_threads: int
        """
        _query = _castable_dtype_or_raise(vectors, expected=self._vector_dtype, message=f"DynamicMemoryIndex expected a query vector of dtype of {self._vector_dtype}")
        _assert(len(vectors.shape) == 2, "vectors must be a 2-d array")
        _assert(
            vectors.shape[0] == vector_ids.shape[0], "Number of vectors must be equal to number of ids"
        )
        _vectors = vectors.astype(dtype=self._vector_dtype, casting="safe", copy=False)
        _vector_ids = vector_ids.astype(dtype=np.uintc, casting="safe", copy=False)

        return self._index.batch_insert(
            _vectors, _vector_ids, _vector_ids.shape[0], num_threads
        )

    def mark_deleted(self, vector_id: VectorIdentifier):
        """
        Mark vector for deletion. This is a soft delete that won't return the vector id in any results, but does not
        remove it from the underlying index files or memory structure. To execute a hard delete, call this method and
        then call the much more expensive ``consolidate_delete`` method on this index.
        :param vector_id: The vector id to delete. Must be a uint32.
        :type vector_id: int
        """
        _assert_is_positive_uint32(vector_id, "vector_id")
        self._index.mark_deleted(np.uintc(vector_id))

    def consolidate_delete(self):
        """
        This method actually restructures the DiskANN index to remove the items that have been marked for deletion.
        """
        self._index.consolidate_delete()
