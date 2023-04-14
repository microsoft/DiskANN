# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
import shutil
import tempfile
from typing import BinaryIO, Literal, Optional, Union

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

    _ = file_handler.write(np.array(vectors.shape, dtype=np.int32).tobytes())
    _ = file_handler.write(vectors.tobytes())


def _build_disk_index_from_vector_file(
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
    dap_metric = _get_valid_metric(metric)
    _assert_dtype(vector_dtype, "vector_dtype")
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(search_memory_maximum > 0, "search_memory_maximum must be larger than 0")
    _assert(build_memory_maximum > 0, "build_memory_maximum must be larger than 0")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(pq_disk_bytes, "pq_disk_bytes")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    if vector_dtype == np.single:
        _builder = _native_dap.build_disk_float_index
    elif vector_dtype == np.ubyte:
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


def build_disk_index(
    vector_path_or_np_array: Union[str, np.ndarray],
    metric: Literal["l2", "mips"],
    index_path: str,
    complexity: int,
    graph_degree: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int,
    vector_dtype: Optional[VectorDType] = None,
    index_prefix: str = "ann",
):
    """
    This function will construct a DiskANN Disk Index and save it to disk.

    If you provide a numpy array, it will save this array to disk in a temp location
    in the format DiskANN's PQ Flash Index builder requires. This temp folder is deleted upon index creation completion
    or error.

    :param vector_path_or_np_array: Either a ``str`` representing a path to a DiskANN vector bin file, or a numpy.ndarray,
        of a supported dtype, in 2 dimensions. Note that vector_dtype must be provided if vector_path_or_np_array is a
        ``str``
    :type vector_path_or_np_array: Union[str, numpy.ndarray]
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
    :param vector_dtype: Required if the provided ``vector_path_or_np_array`` is of type ``str``, else we use the
        ``vector_path_or_np_array.dtype`` if np array.
    :type vector_dtype: Optional[VectorDType], default is ``None``.
    :param index_prefix: The prefix to give your index files. Defaults to ``ann``.
    :type index_prefix: str, default="ann"
    :raises ValueError: If vectors are not 2d numpy array or are not a supported dtype
    :raises ValueError: If any numeric value is in an invalid range
    """

    _assert(
        (isinstance(vector_path_or_np_array, str) and vector_dtype is not None)
        or isinstance(vector_path_or_np_array, np.ndarray),
        "vector_dtype is required if vector_path_or_np_array is a str representing a path to the vector bin file",
    )

    def _build(path: str, v_dtype: VectorDType):
        _build_disk_index_from_vector_file(
            vector_bin_file=path,
            metric=metric,
            vector_dtype=v_dtype,
            index_path=index_path,
            complexity=complexity,
            graph_degree=graph_degree,
            search_memory_maximum=search_memory_maximum,
            build_memory_maximum=build_memory_maximum,
            num_threads=num_threads,
            pq_disk_bytes=pq_disk_bytes,
            index_prefix=index_prefix,
        )

    if isinstance(vector_path_or_np_array, np.ndarray):
        _assert_2d(vector_path_or_np_array, "vectors")
        _assert_dtype(vector_path_or_np_array.dtype, "vectors.dtype")

        _temp_work_dir = tempfile.mkdtemp()
        try:
            temp_vector_bin_path = os.path.join(_temp_work_dir, "vectors.bin")
            with open(
                os.path.join(_temp_work_dir, "vectors.bin"), "wb"
            ) as temp_vector_bin:
                numpy_to_diskann_file(vector_path_or_np_array, temp_vector_bin)
                _build(temp_vector_bin_path, vector_path_or_np_array.dtype)
        finally:
            shutil.rmtree(_temp_work_dir)
    else:
        _build(vector_path_or_np_array, vector_dtype)


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
    if vector_dtype == np.single:
        _builder = _native_dap.build_in_memory_float_index
    elif vector_dtype == np.ubyte:
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
