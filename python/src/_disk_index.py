# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import warnings
from pathlib import Path
from typing import Literal, Tuple

import numpy as np

from . import _diskannpy as _native_dap
from ._common import (
    VectorDType,
    _assert,
    _assert_2d,
    _assert_dtype,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _get_valid_metric,
)

__ALL__ = ["DiskIndex"]


class DiskIndex:
    def __init__(
        self,
        metric: Literal["l2", "mips"],
        vector_dtype: VectorDType,
        index_directory: str,
        num_threads: int,
        num_nodes_to_cache: int,
        cache_mechanism: int = 1,
        index_prefix: str = "ann",
    ):
        """
        The diskannpy.DiskIndex represents our python API into the DiskANN Product Quantization Flash Index library.

        This class is responsible for searching a DiskANN disk index.

        :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
            available for single point floating numbers (numpy.single)
        :type metric: str
        :param vector_dtype: The vector dtype this index will be exposing.
        :type vector_dtype: Type[numpy.single], Type[numpy.byte], Type[numpy.ubyte]
        :param index_directory: Path on disk where the disk index is stored
        :type index_directory: str
        :param num_threads: Number of threads used to load the index (>= 0)
        :type num_threads: int
        :param num_nodes_to_cache: Number of nodes to cache in memory (> -1)
        :type num_nodes_to_cache: int
        :param cache_mechanism: 1 -> use the generated sample_data.bin file for
            the index to initialize a set of cached nodes, up to ``num_nodes_to_cache``, 2 -> ready the cache for up to
            ``num_nodes_to_cache``, but do not initialize it with any nodes. Any other value disables node caching.
        :param index_prefix: A shared prefix that all files in this index will use. Default is "ann".
        :type index_prefix: str
        :raises ValueError: If metric is not a valid metric
        :raises ValueError: If vector dtype is not a supported dtype
        :raises ValueError: If num_threads or num_nodes_to_cache is an invalid range.
        """
        dap_metric = _get_valid_metric(metric)
        _assert_dtype(vector_dtype, "vector_dtype")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_nonnegative_uint32(num_nodes_to_cache, "num_nodes_to_cache")
        index_path = Path(index_directory)
        _assert(
            index_path.exists() and index_path.is_dir(),
            "index_directory must both exist and be a directory",
        )

        self._vector_dtype = vector_dtype
        if vector_dtype == np.single:
            _index = _native_dap.DiskFloatIndex
        elif vector_dtype == np.ubyte:
            _index = _native_dap.DiskUInt8Index
        else:
            _index = _native_dap.DiskInt8Index
        self._index = _index(
            metric=dap_metric,
            index_path_prefix=os.path.join(index_directory, index_prefix),
            num_threads=num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
            cache_mechanism=cache_mechanism,
        )

    def search(
        self, query: np.ndarray, k_neighbors: int, complexity: int, beam_width: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the disk index by a single query vector in a 1d numpy array.

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
        _assert_dtype(query.dtype, "query.dtype")
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_positive_uint32(beam_width, "beam_width")

        if k_neighbors > complexity:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={complexity} was smaller. Increasing {complexity} to {k_neighbors}"
            )
            complexity = k_neighbors

        return self._index.search(
            query=query,
            knn=k_neighbors,
            complexity=complexity,
            beam_width=beam_width,
        )

    def batch_search(
        self,
        queries: np.ndarray,
        k_neighbors: int,
        complexity: int,
        num_threads: int,
        beam_width: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the disk index for many query vectors in a 2d numpy array.

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
        :param beam_width: The beamwidth to be used for search. This is the maximum number of IO requests each query
            will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query,
            but might result in slightly higher total number of IO requests to SSD per query. For the highest query
            throughput with a fixed SSD IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
            Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will
            involve some tuning overhead.
        :type beam_width: int
        :return: Returns a tuple of 2-d numpy ndarrays; each row corresponds to the query vector in the same index,
            and elements in row corresponding from 1..k_neighbors approximate nearest neighbors. The second ndarray
            contains the distances, of the same form: row index will match query index, column index refers to
            1..k_neighbors distance. These are aligned arrays.
        """
        _assert_2d(queries, "queries")
        _assert(
            queries.dtype == self._vector_dtype,
            f"DiskIndex was built expecting a dtype of {self._vector_dtype}, but the query vectors are of dtype "
            f"{queries.dtype}",
        )
        _assert_is_positive_uint32(k_neighbors, "k_neighbors")
        _assert_is_positive_uint32(complexity, "complexity")
        _assert_is_nonnegative_uint32(num_threads, "num_threads")
        _assert_is_positive_uint32(beam_width, "beam_width")

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
            beam_width=beam_width,
            num_threads=num_threads,
        )
