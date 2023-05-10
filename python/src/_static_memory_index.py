# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import warnings
from typing import Literal, Tuple

import numpy as np

from . import _diskannpy as _native_dap
from ._common import (
    _VALID_DTYPES,
    VectorDType,
    _assert,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _get_valid_metric,
    _assert_existing_directory,
    _assert_existing_file,
)

__ALL__ = ["StaticMemoryIndex"]


class StaticMemoryIndex:
    def __init__(
        self,
        metric: Literal["l2", "mips"],
        vector_dtype: VectorDType,
        data_path: str,
        index_directory: str,
        num_threads: int,
        initial_search_complexity: int,
        index_prefix: str = "ann",
    ):
        """
        The diskannpy.StaticMemoryIndex represents our python API into a static DiskANN InMemory Index library.

        This static index is treated exactly like the DiskIndex, in that it can only be loaded and searched.

        :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
            available for single point floating numbers (numpy.single)
        :type metric: str
        :param vector_dtype: The vector dtype this index will be exposing.
        :type vector_dtype: Type[numpy.single], Type[numpy.byte], Type[numpy.ubyte]
        :param data_path: The path to the vector bin file that created this index. Note that if you use a numpy
            array to build the index, you will still need to save this array as well via the
            ``diskannpy.numpy_to_diskann_file`` and provide the path to it here.
        :type data_path: str
        :param index_directory: The directory the index files reside in
        :type index_directory: str
        :param initial_search_complexity: A positive integer that tunes how much work should be completed in the
            conduct of a search. This can be overridden on a per search basis, but this initial value allows us
            to pre-allocate a search scratch space. It is suggested that you set this value to the P95 of your
            search complexity values.
        :type initial_search_complexity: int
        :param index_prefix: A shared prefix that all files in this index will use. Default is "ann".
        :type index_prefix: str
        """
        dap_metric = _get_valid_metric(metric)
        _assert(
            vector_dtype in _VALID_DTYPES,
            f"vector_dtype {vector_dtype} is not in list of valid dtypes supported: {_VALID_DTYPES}",
        )
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_positive_uint32(
            initial_search_complexity, "initial_search_complexity"
        )
        _assert_existing_file(data_path, "data_path")
        _assert_existing_directory(index_directory, "index_directory")

        _assert(index_prefix != "", "index_prefix cannot be an empty string")

        self._vector_dtype = vector_dtype
        if vector_dtype == np.single:
            _index = _native_dap.StaticMemoryFloatIndex
        elif vector_dtype == np.ubyte:
            _index = _native_dap.StaticMemoryUInt8Index
        else:
            _index = _native_dap.StaticMemoryInt8Index
        self._index = _index(
            metric=dap_metric,
            data_path=data_path,
            index_path=os.path.join(index_directory, index_prefix),
            num_threads=num_threads,
            initial_search_complexity=initial_search_complexity,
        )

    def search(self, query: np.ndarray, k_neighbors: int, complexity: int):
        """
        Searches the static in memory index by a single query vector in a 1d numpy array.

        numpy array dtype must match index.

        :param query: 1d numpy array of the same dimensionality and dtype of the index.
        :type query: numpy.ndarray
        :param k_neighbors: Number of neighbors to be returned. If query vector exists in index, it almost definitely
            will be returned as well, so adjust your ``k_neighbors`` as appropriate. (> 0)
        :type k_neighbors: int
        :param complexity: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type complexity: int
        :param beam_width: The beamwidth to be used for search. This is the maximum number of IO requests each query
            will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query,
            but might result in slightly higher total number of IO requests to SSD per query. For the highest query
            throughput with a fixed SSD IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
            Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will
            involve some tuning overhead.
        :type beam_width: int
        :return: Returns a tuple of 1-d numpy ndarrays; the first including the indices of the approximate nearest
            neighbors, the second their distances. These are aligned arrays.
        """
        _assert(len(query.shape) == 1, "query vector must be 1-d")
        _assert(
            query.dtype == self._vector_dtype,
            f"StaticMemoryIndex was built expecting a dtype of {self._vector_dtype}, but the query vector is of dtype "
            f"{query.dtype}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_nonnegative_uint32(complexity, "complexity")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors
        return self._index.search(query=query, knn=k_neighbors, complexity=complexity)

    def batch_search(
        self, queries: np.ndarray, k_neighbors: int, complexity: int, num_threads: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the static, in memory index for many query vectors in a 2d numpy array.

        numpy array dtype must match index.

        This search is parallelized and far more efficient than searching for each vector individually.

        :param queries: 2d numpy array, with column dimensionality matching the index and row dimensionality being the
            number of queries intended to search for in parallel. Dtype must match dtype of the index.
        :type queries: numpy.ndarray
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
        _assert(len(queries.shape) == 2, "queries must must be 2-d np array")
        _assert(
            queries.dtype == self._vector_dtype,
            f"StaticMemoryIndex was built expecting a dtype of {self._vector_dtype}, but the query vectors are of dtype "
            f"{queries.dtype}",
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
            queries=queries,
            num_queries=num_queries,
            knn=k_neighbors,
            complexity=complexity,
            num_threads=num_threads,
        )
