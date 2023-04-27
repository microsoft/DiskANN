# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
from pathlib import Path
from typing import BinaryIO, Literal, Optional, Tuple, Union

import numpy as np

from . import _diskannpy as _native_dap
from ._common import (
    VectorDType,
    _assert,
    _assert_2d,
    _assert_dtype,
    _assert_existing_file,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _get_valid_metric,
)
from ._diskannpy import defaults


def numpy_to_diskann_file(vectors: np.ndarray, file_handler: BinaryIO):
    """
    Utility function that writes a DiskANN binary vector formatted file to the location of your choosing.

    :param vectors: A 2d array of dtype ``numpy.single``, ``numpy.ubyte``, or ``numpy.byte``
    :type vectors: numpy.ndarray, dtype in set {numpy.single, numpy.ubyte, numpy.byte}
    :param file_handler: An open binary file handler (typing.BinaryIO).
    :type file_handler: io.BinaryIO
    :raises ValueError: If vectors are the wrong shape or an unsupported dtype
    :raises ValueError: If output_path is not a str or ``io.BinaryIO``
    """
    _assert_2d(vectors, "vectors")
    _assert_dtype(vectors.dtype, "vectors.dtype")

    _ = file_handler.write(np.array(vectors.shape, dtype=np.intc).tobytes())
    _ = file_handler.write(vectors.tobytes())


def _valid_path_and_dtype(
    data: Union[str, np.ndarray], vector_dtype: Optional[VectorDType], index_path: str
) -> Tuple[str, VectorDType]:
    if isinstance(data, np.ndarray):
        _assert_2d(data, "data")
        _assert_dtype(data.dtype, "data.dtype")

        vector_bin_path = os.path.join(index_path, "vectors.bin")
        if Path(vector_bin_path).exists():
            raise ValueError(
                f"The path {vector_bin_path} already exists. Remove it and try again."
            )
        with open(vector_bin_path, "wb") as temp_vector_bin:
            numpy_to_diskann_file(data, temp_vector_bin)
        vector_dtype_actual = data.dtype
    else:
        vector_bin_path = data
        _assert(
            Path(data).exists() and Path(data).is_file(),
            "if data is of type `str`, it must both exist and be a file",
        )
        vector_dtype_actual = vector_dtype
    return vector_bin_path, vector_dtype_actual


def build_disk_index(
    data: Union[str, np.ndarray],
    metric: Literal["l2", "mips"],
    index_directory: str,
    complexity: int,
    graph_degree: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int = defaults.PQ_DISK_BYTES,
    vector_dtype: Optional[VectorDType] = None,
    index_prefix: str = "ann",
):
    """
    This function will construct a DiskANN Disk Index and save it to disk.

    If you provide a numpy array, it will save this array to disk in a temp location
    in the format DiskANN's PQ Flash Index builder requires. This temp folder is deleted upon index creation completion
    or error.

    :param data: Either a ``str`` representing a path to a DiskANN vector bin file, or a numpy.ndarray,
        of a supported dtype, in 2 dimensions. Note that vector_dtype must be provided if vector_path_or_np_array is a
        ``str``
    :type data: Union[str, numpy.ndarray]
    :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
        available for single point floating numbers (numpy.single)
    :type metric: str
    :param index_directory: The path on disk that the index will be created in.
    :type index_directory: str
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
    :param num_threads: Number of threads to use when creating this index.0 indicates we should use all available
        system threads.
    :type num_threads: int
    :param pq_disk_bytes:  Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100%
        recall. If your vectors are too large to store in SSD, this parameter provides the option to compress the
        vectors using PQ for storing on SSD. This will trade off recall. You would also want this to be greater
        than the number of bytes used for the PQ compressed data stored in-memory. Default is ``0``.
    :type pq_disk_bytes: int (default = 0)
    :param vector_dtype: Required if the provided ``vector_path_or_np_array`` is of type ``str``, else we use the
        ``vector_path_or_np_array.dtype`` if np array.
    :type vector_dtype: Optional[VectorDType], default is ``None``.
    :param index_prefix: The prefix to give your index files. Defaults to ``ann``.
    :type index_prefix: str, default="ann"
    :raises ValueError: If vectors are not 2d numpy array or are not a supported dtype
    :raises ValueError: If any numeric value is in an invalid range
    """

    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _get_valid_metric(metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(search_memory_maximum > 0, "search_memory_maximum must be larger than 0")
    _assert(build_memory_maximum > 0, "build_memory_maximum must be larger than 0")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(pq_disk_bytes, "pq_disk_bytes")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_prefix
    )

    if vector_dtype_actual == np.single:
        _builder = _native_dap.build_disk_float_index
    elif vector_dtype_actual == np.ubyte:
        _builder = _native_dap.build_disk_uint8_index
    else:
        _builder = _native_dap.build_disk_int8_index

    _builder(
        metric=dap_metric,
        data_file_path=vector_bin_path,
        index_prefix_path=os.path.join(index_directory, index_prefix),
        complexity=complexity,
        graph_degree=graph_degree,
        final_index_ram_limit=search_memory_maximum,
        indexing_ram_budget=build_memory_maximum,
        num_threads=num_threads,
        pq_disk_bytes=pq_disk_bytes,
    )


def build_memory_index(
    data: Union[str, np.ndarray],
    metric: Literal["l2", "mips"],
    index_directory: str,
    complexity: int,
    graph_degree: int,
    num_threads: int,
    alpha: float = defaults.ALPHA,
    use_pq_build: bool = defaults.USE_PQ_BUILD,
    num_pq_bytes: int = defaults.NUM_PQ_BYTES,
    use_opq: bool = defaults.USE_OPQ,
    vector_dtype: Optional[VectorDType] = None,
    label_file: str = "",
    universal_label: str = "",
    filter_complexity: int = defaults.FILTER_COMPLEXITY,
    index_prefix: str = "ann"
):
    """
    Builds a memory index and saves it to disk to be loaded into ``StaticMemoryIndex``.

    :param data: Either a ``str`` representing a path to a DiskANN vector bin file, or a numpy.ndarray,
        of a supported dtype, in 2 dimensions. Note that vector_dtype must be provided if vector_path_or_np_array is a
        ``str``
    :type data: Union[str, numpy.ndarray]
    :param metric: One of {"l2", "mips"}. L2 is supported for all 3 vector dtypes, but MIPS is only
        available for single point floating numbers (numpy.single)
    :type metric: str
    :param index_directory: The path on disk that the index will be created in.
    :type index_directory: str
    :param complexity: The size of queue to use when building the index for search. Values between 75 and 200 are
        typical. Larger values will take more time to build but result in indices that provide higher recall for
        the same search complexity. Use a value that is at least as large as R unless you are prepared to
        somewhat compromise on quality
    :type complexity: int
    :param graph_degree: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
        result in larger indices and longer indexing times, but better search quality.
    :type graph_degree int
    :param num_threads: Number of threads to use when creating this index. 0 indicates we should use all available
        system threads.
    :type num_threads: int
    :param alpha:
    :param use_pq_build:
    :param num_pq_bytes:
    :param use_opq:
    :param vector_dtype: Required if the provided ``vector_path_or_np_array`` is of type ``str``, else we use the
        ``vector_path_or_np_array.dtype`` if np array.
    :type vector_dtype: Optional[VectorDType], default is ``None``.
    :param label_file: Defaults to ""
    :type label_file: str
    :param universal_label: Defaults to ""
    :param filter_complexity: Complexity to use when using filters. Default is 0.
    :type filter_complexity: int
    :param index_prefix: The prefix to give your index files. Defaults to ``ann``.
    :type index_prefix: str, default="ann"
    :return:
    """
    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _get_valid_metric(metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(alpha >= 1, "alpha must be >= 1, and realistically should be kept between [1.0, 2.0)")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(num_pq_bytes, "num_pq_bytes")
    _assert_is_nonnegative_uint32(filter_complexity, "filter_complexity")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_directory
    )

    if vector_dtype_actual == np.single:
        _builder = _native_dap.build_in_memory_float_index
    elif vector_dtype_actual == np.ubyte:
        _builder = _native_dap.build_in_memory_uint8_index
    else:
        _builder = _native_dap.build_in_memory_int8_index

    _builder(
        metric=dap_metric,
        data_file_path=vector_bin_path,
        index_output_path=os.path.join(index_directory, index_prefix),
        complexity=complexity,
        graph_degree=graph_degree,
        alpha=alpha,
        num_threads=num_threads,
        use_pq_build=use_pq_build,
        num_pq_bytes=num_pq_bytes,
        use_opq=use_opq,
        label_file=label_file,
        universal_label=universal_label,
        filter_complexity=filter_complexity
    )
