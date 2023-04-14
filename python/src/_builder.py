# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import tempfile
from typing import BinaryIO, Literal, Union

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


def _numpy_to_diskann_file(
    vectors: np.ndarray,
    file_handler: BinaryIO,
):
    _assert_2d(vectors, "vectors")
    _assert_dtype(vectors, "vectors")

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
    complexity: int,
    graph_degree: int,
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
    :param complexity: The size of queue to use when building the index for search. Values between 75 and 200 are
        typical. Larger values will take more time to build but result in indices that provide higher recall for
        the same search complexity. Use a value that is at least as large as R unless you are prepared to
        somewhat compromise on quality
    :type complexity: int
    :param graph_degree: The degree of the graph index, typically between 60 and 50. A larger maximum degree will
        result in larger indices and longer indexing times, but better search quality.
    :type graph_degree: int
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
    _assert_dtype(vector_dtype, "vector_dtype")
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(search_memory_maximum > 0, "search_memory_maximum must be larger than 0")
    _assert(build_memory_maximum > 0, "build_memory_maximum must be larger than 0")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(pq_disk_bytes, "pq_disk_bytes")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    if vector_dtype is np.single:
        _builder = _native_dap.build_disk_float_index
    elif vector_dtype is np.ubyte:
        _builder = _native_dap.build_disk_uint8_index
    else:
        _builder = _native_dap.build_disk_int8_index

    _builder(
        metric=dap_metric,
        data_file_path=vector_bin_file,
        index_prefix_path=os.path.join(index_path, index_prefix),
        complexity=complexity,
        graph_degree=graph_degree,
        final_index_ram_limit=build_memory_maximum,
        indexing_ram_budget=build_memory_maximum,
        num_threads=num_threads,
        pq_disk_bytes=pq_disk_bytes,
    )


def build_disk_index_from_vectors(
    vectors: np.ndarray,
    metric: Literal["l2", "mips"],
    index_path: str,
    complexity: int,
    graph_degree: int,
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
    :param complexity: The size of queue to use when building the index for search. Values between 75 and 200 are
        typical. Larger values will take more time to build but result in indices that provide higher recall for
        the same search complexity. Use a value that is at least as large as R unless you are prepared to
        somewhat compromise on quality
    :type complexity: int
    :param graph_degree: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
        result in larger indices and longer indexing times, but better search quality.
    :type graph_degree int
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
    _assert_2d(vectors, "vectors")
    _assert_dtype(vectors, "vectors")

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
                complexity=complexity,
                graph_degree=graph_degree,
                search_memory_maximum=search_memory_maximum,
                build_memory_maximum=build_memory_maximum,
                num_threads=num_threads,
                pq_disk_bytes=pq_disk_bytes,
                index_prefix=index_prefix,
            )
    finally:
        shutil.rmtree(_temp_work_dir)


def build_memory_index_from_vector_file(
    vector_bin_file: str,
    metric: Literal["l2", "mips"],
    vector_dtype: VectorDType,
    index_path: str,
    complexity: int,
    graph_degree: int,
    alpha: float,
    num_threads: int,
    use_pq_build: bool,
    num_pq_bytes: int,
    use_opq: bool,
    label_file: str = "",
    universal_label: str = "",
    filter_complexity: int = 0,
    index_prefix: str = "ann",
):
    dap_metric = _get_valid_metric(metric)
    _assert_dtype(vector_dtype, "vector_dtype")
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(num_pq_bytes, "num_pq_bytes")
    _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")
    if vector_dtype is np.single:
        _builder = _native_dap.build_in_memory_float_index
    elif vector_dtype is np.ubyte:
        _builder = _native_dap.build_in_memory_uint8_index
    else:
        _builder = _native_dap.build_in_memory_int8_index

    _builder(
        metric=dap_metric,
        data_file_path=vector_bin_file,
        index_output_path=os.path.join(index_path, index_prefix),
        complexity=complexity,
        graph_degree=graph_degree,
        alpha=alpha,
        num_threads=num_threads,
        use_pq_build=use_pq_build,
        num_pq_bytes=num_pq_bytes,
        use_opq=use_opq,
        label_file=label_file,
        universal_label=universal_label,
        filter_complexity=filter_complexity,
    )
