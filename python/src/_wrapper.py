# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import tempfile
import warnings
from typing import BinaryIO, Literal, Tuple, Type, TypeVar, Union

import numpy as np

from . import _diskannpy as _native_dap

__ALL__ = [
    "build_disk_index_from_vector_file",
    "build_disk_index_from_vectors",
    "numpy_to_diskann_file",
    "VectorDType",
    "DiskIndex",
]


_VALID_DTYPES = [np.single, np.byte, np.ubyte]

_DTYPE_TO_NATIVE_INDEX = {
    np.single: _native_dap.DiskANNFloatIndex,
    np.ubyte: _native_dap.DiskANNUInt8Index,
    np.byte: _native_dap.DiskANNInt8Index,
}

_DTYPE_TO_NATIVE_INMEM_DYNAMIC_INDEX = {
    np.single: _native_dap.DiskANNDynamicInMemFloatIndex,
    np.ubyte: _native_dap.DiskANNDynamicInMemUint8Index,
    np.byte: _native_dap.DiskANNDynamicInMemInt8Index,
}

_DTYPE_TO_NATIVE_INMEM_STATIC_INDEX = {
    np.single: _native_dap.DiskANNStaticInMemFloatIndex,
    np.ubyte: _native_dap.DiskANNStaticInMemUint8Index,
    np.byte: _native_dap.DiskANNStaticInMemInt8Index,
}


VectorDType = TypeVar("VectorDType", Type[np.single], Type[np.ubyte], Type[np.byte])


def _get_valid_metric(metric: str) -> _native_dap.Metric:
    if not isinstance(metric, str):
        raise ValueError("metric must be a string")
    if metric.lower() == "l2":
        return _native_dap.L2
    elif metric.lower() == "mips":
        return _native_dap.INNER_PRODUCT
    else:
        raise ValueError("metric must be one of 'l2' or 'mips'")


def _validate_dtype(vectors: np.ndarray):
    if vectors.dtype not in _VALID_DTYPES:
        raise ValueError(
            f"vectors provided had dtype {vectors.dtype}, but must be single precision float "
            f"(numpy.single), unsigned 8bit integer (numpy.ubyte), or signed 8bit integer (numpy.byte)."
        )


def _validate_shape(vectors: np.ndarray):
    if len(vectors.shape) != 2:
        raise ValueError("vectors must be 2d numpy array")


def _numpy_to_diskann_file(
    vectors: np.ndarray,
    file_handler: BinaryIO,
):
    _validate_shape(vectors)
    _validate_dtype(vectors)
    _ = file_handler.write(np.array(vectors.shape, dtype=np.int32).tobytes())
    _ = file_handler.write(vectors.tobytes())


def numpy_to_diskann_file(vectors: np.ndarray, output_path: Union[str, BinaryIO]):
    """
    Utility function that writes a DiskANN binary vector formatted file to the location of your choosing.

    :param vectors: A 2d array of dtype ``numpy.single``, ``numpy.ubyte``, or ``numpy.byte``
    :type vectors: numpy.ndarray, dtype in set {numpy.single, numpy.ubyte, numpy.byte}
    :param output_path: Where to write the file. If a string is provided, a binary writer will be opened at that
        location. Otherwise it is presumed ``output_path`` is a BinaryIO file handler and will write to it.
    :type output_path: Union[str, io.BinaryIO]
    :raises ValueError: If vectors are the wrong shape or an unsupported dtype
    :raises ValueError: If output_path is not a str or ``io.BinaryIO``
    """
    if isinstance(output_path, BinaryIO):
        _numpy_to_diskann_file(vectors, output_path)
    elif isinstance(output_path, str):
        with open(output_path, "wb") as binary_out:
            _numpy_to_diskann_file(vectors, binary_out)
    else:
        raise ValueError(
            "output_path must be either a str or an open binary file handler (e.g. `handler = open('my_file_path', 'wb')`)"
        )


def build_disk_index_from_vector_file(
    vector_bin_file: str,
    metric: Literal["l2", "mips"],
    vector_dtype: VectorDType,
    index_path: str,
    max_degree: int,
    list_size: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int,
    index_prefix: str = "ann",
):
    """
    Builds a DiskANN disk index based on a provided DiskANN formatted binary file path.

    :param vector_bin_file: Must be a binary file formatted in the expected DiskANN file format.
        Use ``diskannpy.numpy_to_diskann_file`` to create it.
    :type vector_bin_file: str
    :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
        available for single point floating numbers (numpy.single)
    :type metric: str
    :param vector_dtype: The vector dtype this index will be exposing.
    :type vector_dtype: Type[numpy.single], Type[numpy.byte], Type[numpy.ubyte]
    :param index_path: The path on disk that the index will be created in.
    :type index_path: str
    :param max_degree: The degree of the graph index, typically between 60 and 50. A larger maximum degree will
        result in larger indices and longer indexing times, but better search quality.
    :type max_degree: int
    :param list_size: The size of queue to use when building the index for search. Values between 75 and 200 are
        typical. Larger values will take more time to build but result in indices that provide higher recall for
        the same search complexity. Use a value that is at least as large as R unless you are prepared to
        somewhat compromise on quality
    :type list_size: int
    :param search_memory_maximum: Build index with the expectation that the search will use at most
        ``search_memory_maximum``
    :type search_memory_maximum: float
    :param build_memory_maximum: Build index using at most ``build_memory_maximum``
    :type build_memory_maximum: float
    :param num_threads: Number of threads to use when creating this index.
    :type num_threads: int
    :param pq_disk_bytes:  Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100%
        recall. If your vectors are too large to store in SSD, this parameter provides the option to compress the
        vectors using PQ for storing on SSD. This will trade off recall. You would also want this to be greater
        than the number of bytes used for the PQ compressed data stored in-memory. Default is ``0``.
    :type pq_disk_bytes: int (default = 0)
    :param index_prefix: The prefix to give your index files. Defaults to ``ann``.
    :type index_prefix: str, default="ann"
    :raises ValueError: If any numeric parameter is in an invalid range.
    """
    dap_metric = _get_valid_metric(metric)
    if vector_dtype not in _VALID_DTYPES:
        raise ValueError(
            f"vector_dtype {vector_dtype} is not in list of valid dtypes supported: {_VALID_DTYPES}"
        )
    if list_size <= 0:
        raise ValueError("list_size must be a positive integer")
    if max_degree <= 0:
        raise ValueError("max_degree must be a positive integer")
    if search_memory_maximum <= 0:
        raise ValueError("search_memory_maximum must be larger than 0")
    if build_memory_maximum <= 0:
        raise ValueError("build_memory_maximum must be larger than 0")
    if num_threads < 0:
        raise ValueError("num_threads must be a nonnegative integer")
    if pq_disk_bytes < 0:
        raise ValueError("pq_disk_bytes must be nonnegative integer")

    index = _DTYPE_TO_NATIVE_INDEX[vector_dtype](dap_metric)
    index.build(
        data_file_path=vector_bin_file,
        index_prefix_path=os.path.join(index_path, index_prefix),
        R=max_degree,
        L=list_size,
        final_index_ram_limit=search_memory_maximum,
        indexing_ram_limit=build_memory_maximum,
        num_threads=num_threads,
        pq_disk_bytes=pq_disk_bytes,
    )


def build_disk_index_from_vectors(
    vectors: np.ndarray,
    metric: Literal["l2", "mips"],
    index_path: str,
    max_degree: int,
    list_size: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int,
    index_prefix: str = "ann",
):
    """
    This function is a convenience function for first converting the provided numpy 2-d array into the binary format
        expected by the DiskANN library, and then using that to generate the index as per
        ``DiskIndex.build_from_vector_file()``. After completion, this temporary file is deleted.

    :param vectors: A numpy.ndarray, of a supported dtype, in 2 dimensions
    :type vectors: numpy.ndarray
    :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
        available for single point floating numbers (numpy.single)
    :type metric: str
    :param index_path: The path on disk that the index will be created in.
    :type index_path: str
    :param max_degree: The degree of the graph index, typically between 60 and 50. A larger maximum degree will
        result in larger indices and longer indexing times, but better search quality.
    :type max_degree: int
    :param list_size: The size of queue to use when building the index for search. Values between 75 and 200 are
        typical. Larger values will take more time to build but result in indices that provide higher recall for
        the same search complexity. Use a value that is at least as large as R unless you are prepared to
        somewhat compromise on quality
    :type list_size: int
    :param search_memory_maximum: Build index with the expectation that the search will use at most
        ``search_memory_maximum``
    :type search_memory_maximum: float
    :param build_memory_maximum: Build index using at most ``build_memory_maximum``
    :type build_memory_maximum: float
    :param num_threads: Number of threads to use when creating this index.
    :type num_threads: int
    :param pq_disk_bytes:  Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100%
        recall. If your vectors are too large to store in SSD, this parameter provides the option to compress the
        vectors using PQ for storing on SSD. This will trade off recall. You would also want this to be greater
        than the number of bytes used for the PQ compressed data stored in-memory. Default is ``0``.
    :type pq_disk_bytes: int (default = 0)
    :param index_prefix: The prefix to give your index files. Defaults to ``ann``.
    :type index_prefix: str, default="ann"
    :raises ValueError: If vectors are not 2d numpy array or are not a supported dtype
    :raises ValueError: If any numeric value is in an invalid range
    """
    _validate_dtype(vectors)
    _validate_shape(vectors)

    _temp_work_dir = tempfile.mkdtemp()
    try:
        temp_vector_bin_path = os.path.join(_temp_work_dir, "vectors.bin")
        with open(os.path.join(_temp_work_dir, "vectors.bin"), "wb") as temp_vector_bin:
            numpy_to_diskann_file(vectors, temp_vector_bin)
            build_disk_index_from_vector_file(
                vector_bin_file=temp_vector_bin_path,
                metric=metric,
                vector_dtype=vectors.dtype,
                index_path=index_path,
                max_degree=max_degree,
                list_size=list_size,
                search_memory_maximum=search_memory_maximum,
                build_memory_maximum=build_memory_maximum,
                num_threads=num_threads,
                pq_disk_bytes=pq_disk_bytes,
                index_prefix=index_prefix,
            )
    finally:
        shutil.rmtree(_temp_work_dir)


class DiskIndex:
    def __init__(
        self,
        metric: Literal["l2", "mips"],
        vector_dtype: VectorDType,
        index_path: str,
        num_threads: int,
        num_nodes_to_cache: int,
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
        :param index_path: Path on disk where the disk index is stored
        :type index_path: str
        :param num_threads: Number of threads used to load the index (>= 0)
        :type num_threads: int
        :param num_nodes_to_cache: Number of nodes to cache in memory (> -1)
        :type num_nodes_to_cache: int
        :param index_prefix: A shared prefix that all files in this index will use. Default is "ann".
        :type index_prefix: str
        :raises ValueError: If metric is not a valid metric
        :raises ValueError: If vector dtype is not a supported dtype
        :raises ValueError: If num_threads or num_nodes_to_cache is an invalid range.
        """
        dap_metric = _get_valid_metric(metric)
        if vector_dtype not in _VALID_DTYPES:
            raise ValueError(
                f"vector_dtype {vector_dtype} is not in list of valid dtypes supported: {_VALID_DTYPES}"
            )
        if num_threads < 0:
            raise ValueError("num_threads must be a non-negative integer")
        if num_nodes_to_cache < 0:
            raise ValueError("num_nodes_to_cache must be a non-negative integer")
        self._vector_dtype = vector_dtype
        self._index = _DTYPE_TO_NATIVE_INDEX[vector_dtype](dap_metric)
        self._index.load_index(
            index_path_prefix=os.path.join(index_path, index_prefix),
            num_threads=num_threads,
            num_nodes_to_cache=num_nodes_to_cache,
        )

    def search(
        self, query: np.ndarray, k_neighbors: int, list_size: int, beam_width: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the disk index by a single query vector in a 1d numpy array.

        numpy array dtype must match index.

        :param query: 1d numpy array of the same dimensionality and dtype of the index.
        :type query: numpy.ndarray
        :param k_neighbors: Number of neighbors to be returned. If query vector exists in index, it almost definitely
            will be returned as well, so adjust your ``k_neighbors`` as appropriate. (> 0)
        :type k_neighbors: int
        :param list_size: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type list_size: int
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
        if len(query.shape) != 1:
            raise ValueError("query vector must be 1-d")
        if query.dtype != self._vector_dtype:
            raise ValueError(
                f"DiskIndex was built expecting a dtype of {self._vector_dtype}, but the query vector is "
                f"of dtype {query.dtype}"
            )
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer")
        if list_size <= 0:
            raise ValueError("list_size must be a positive integer")
        if beam_width <= 0:
            raise ValueError("beam_width must be a positive integer")

        if k_neighbors > list_size:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={list_size} was smaller. Increasing {list_size} to {k_neighbors}"
            )
            list_size = k_neighbors
        return self._index.search(
            query=query,
            knn=k_neighbors,
            l_search=list_size,
            beam_width=beam_width,
        )

    def batch_search(
        self,
        queries: np.ndarray,
        k_neighbors: int,
        list_size: int,
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
        :param list_size: Size of list to use while searching. List size increases accuracy at the cost of latency. Must
            be at least k_neighbors in size.
        :type list_size: int
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
        if len(queries.shape) != 2:
            raise ValueError("queries must must be 2-d np array")
        if queries.dtype != self._vector_dtype:
            raise ValueError(
                f"DiskIndex was built expecting a dtype of {self._vector_dtype}, but the query vectors "
                f"are of dtype {queries.dtype}"
            )
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer")
        if list_size <= 0:
            raise ValueError("list_size must be a positive integer")
        if num_threads < 0:
            raise ValueError("num_threads must be a nonnegative integer")
        if beam_width <= 0:
            raise ValueError("beam_width must be a positive integer")

        if k_neighbors > list_size:
            warnings.warn(
                f"k_neighbors={k_neighbors} asked for, but list_size={list_size} was smaller. Increasing {list_size} to {k_neighbors}"
            )
            list_size = k_neighbors

        num_queries, dim = queries.shape
        return self._index.batch_search(
            queries=queries,
            num_queries=num_queries,
            knn=k_neighbors,
            l_search=list_size,
            beam_width=beam_width,
            num_threads=num_threads,
        )
