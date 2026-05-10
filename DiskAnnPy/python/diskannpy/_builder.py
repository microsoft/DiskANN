# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from . import DistanceMetric, VectorDType, VectorIdentifierBatch, VectorLikeBatch
from . import _diskannpy as _native_dap
from ._common import (
    _assert,
    _assert_is_nonnegative_uint32,
    _assert_is_positive_uint32,
    _castable_dtype_or_raise,
    _valid_metric,
    _write_index_metadata,
    valid_dtype,
    valid_dap_dtype,
    _valid_pq_params,
)
from ._files import tags_to_file, vectors_metadata_from_file, vectors_to_file
from ._defaults import *
def _valid_path_and_dtype(
    data: Union[str, VectorLikeBatch],
    vector_dtype: VectorDType,
    index_path: str,
    index_prefix: str,
) -> Tuple[str, VectorDType]:
    if isinstance(data, str):
        vector_bin_path = data
        _assert(
            Path(data).exists() and Path(data).is_file(),
            "if data is of type `str`, it must both exist and be a file",
        )
        vector_dtype_actual = valid_dtype(vector_dtype)
    else:
        vector_bin_path = os.path.join(index_path, f"{index_prefix}_vectors.bin")
        if Path(vector_bin_path).exists():
            raise ValueError(
                f"The path {vector_bin_path} already exists. Remove it and try again."
            )
        vector_dtype_actual = valid_dtype(data.dtype)
        vectors_to_file(vector_file=vector_bin_path, vectors=data)

    return vector_bin_path, vector_dtype_actual


def build_disk_index(
    data: Union[str, VectorLikeBatch],
    distance_metric: str,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    build_memory_maximum: float,
    num_threads: int,
    alpha: float = ALPHA,
    build_pq_bytes: int = BUILD_PQ_BYTES,
    num_of_pq_chunks:int = NUM_OF_PQ_CHUNKS,
    vector_dtype: Optional[VectorDType] = None,
    index_prefix: str = "ann",
) -> None:
    r"""
    This function will construct a DiskANN disk index. Disk indices are ideal for very large datasets that
    are too large to fit in memory. Memory is still used, but it is primarily used to provide precise disk
    locations for fast retrieval of smaller subsets of the index without compromising much on recall.

    If you provide a numpy array, it will save this array to disk in a temp location
    in the format DiskANN's PQ Flash Index builder requires. This temp folder is deleted upon index creation completion
    or error.

    ## Distance Metric and Vector Datatype Restrictions
    | Metric \ Datatype | np.float32 | np.uint8 | np.int8 |
    |-------------------|------------|----------|---------|
    | L2                |      ✅     |     ✅    |    ✅    |
    | CosineNormalized  |      ✅     |     ❌    |    ❌    |
    | Cosine            |      ✅     |     ❌    |    ❌    |

    [^bug-in-disk-cosine]: For StaticDiskIndex, Cosine distances are not currently supported.

    ### Parameters
    - **data**: Either a `str` representing a path to a DiskANN vector bin file, or a numpy.ndarray,
      of a supported dtype, in 2 dimensions. Note that `vector_dtype` must be provided if data is a `str`
    - **distance_metric**: A `str`, strictly one of {"l2", "cosine", "cosinenormalized"}. `l2` is supported for all 3
      vector dtypes.
    - **index_directory**: The index files will be saved to this **existing** directory path
    - **complexity**: The size of the candidate nearest neighbor list to use when building the index. Values between 75
      and 200 are typical. Larger values will take more time to build but result in indices that provide higher recall
      for the same search complexity. Use a value that is at least as large as `graph_degree` unless you are prepared
      to compromise on quality
    - **graph_degree**: The degree of the graph index, typically between 60 and 150. A larger maximum degree will
      result in larger indices and longer indexing times, but better search quality.
    - **build_memory_maximum**: Build index using at most `build_memory_maximum` in gb. Building processes typically
      require more memory, while search memory can be reduced.
    - **num_threads**: Number of threads to use when creating this index. `0` is used to indicate all available
      logical processors should be used.
    - **alpha**: The alpha parameter (>=1) is used to control the nature and number of points that are added to the
      graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) to convergence, but probably more
      distance comparisons compared to a lower alpha value. Default is 1.2.
    - **build_pq_bytes**: Set to a positive value less than the dimensionality of the data to enable faster index
      build with PQ based distance comparisons.. Default is `0`.
        - **num_of_pq_chunks**: Number of PQ chunks to split each vector into. Each chunk is compressed to 1 byte during PQ compression.
            It's the number of bytes used for the PQ compressed data stored in-memory. Default is `1`.
    - **vector_dtype**: Required if the provided `data` is of type `str`, else we use the `data.dtype` if np array.
    - **index_prefix**: The prefix of the index files. Defaults to "ann".
    """

    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _valid_metric(distance_metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert(build_memory_maximum > 0, "build_memory_maximum must be larger than 0")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(build_pq_bytes, "build_pq_bytes")
    _assert_is_nonnegative_uint32(num_of_pq_chunks, "num_of_pq_chunks")
    _assert(num_of_pq_chunks > 0, "num_of_pq_chunks must be larger than 0")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")


    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_directory, index_prefix
    )

    vector_dap_dtype = valid_dap_dtype(vector_dtype_actual)

    num_points, dimensions = vectors_metadata_from_file(vector_bin_path)
    
    index_prefix_path = os.path.join(index_directory, index_prefix)

    _native_dap.build_disk_index(
        data_type=vector_dap_dtype,
        distance_metric=dap_metric,
        data_path=vector_bin_path,
        index_path_prefix=index_prefix_path,
        complexity=complexity,
        graph_degree=graph_degree,
        build_dram_budget=build_memory_maximum,
        num_threads=num_threads,
        build_pq_bytes=build_pq_bytes,
        alpha=alpha,
        num_of_pq_chunks=num_of_pq_chunks
    )
    _write_index_metadata(
        index_prefix_path, vector_dtype_actual, dap_metric, num_points, dimensions
    )


def build_async_index(
    data: Union[str, VectorLikeBatch],
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    num_start_pts: int,
    num_threads: int,
    graph_slack_factor: float,
    max_fp_vecs_per_prune: int,
    num_tasks: int,
    pq_seed: int,
    insert_minibatch_size: int,
    alpha: float = ALPHA,
    num_pq_bytes: int = NUM_PQ_BYTES,
    vector_dtype: Optional[VectorDType] = None,
    tags: Union[str, VectorIdentifierBatch] = "",
    index_prefix: str = "ann",
    use_pq: bool = USE_PQ,
    use_opq: bool = USE_OPQ,
    backedge_ratio: float = BACKEDGE_RATIO,
) -> None:
    _assert(
        (isinstance(data, str) and vector_dtype is not None)
        or isinstance(data, np.ndarray),
        "vector_dtype is required if data is a str representing a path to the vector bin file",
    )
    dap_metric = _valid_metric(distance_metric)
    _assert_is_positive_uint32(complexity, "complexity")
    _assert_is_positive_uint32(graph_degree, "graph_degree")
    _assert_is_nonnegative_uint32(num_threads, "num_threads")
    _assert_is_nonnegative_uint32(num_pq_bytes, "num_pq_bytes")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")

    _valid_pq_params(use_pq, num_pq_bytes, use_opq)

    index_path = Path(index_directory)
    _assert(
        index_path.exists() and index_path.is_dir(),
        "index_directory must both exist and be a directory",
    )

    vector_bin_path, vector_dtype_actual = _valid_path_and_dtype(
        data, vector_dtype, index_directory, index_prefix
    )

    vector_dap_dtype = valid_dap_dtype(vector_dtype_actual)

    if dap_metric == _native_dap.Metric.CosineNormalized:
      _assert(
          vector_dtype_actual == np.float32,
          "Integral vector dtypes (np.uint8, np.int8) are not supported with distance metric cosineNormalized"
      )
    
    if dap_metric == _native_dap.Metric.Cosine:
      _assert(
          vector_dtype_actual != np.uint8,
          "Integral vector dtypes (np.uint8) is not supported with distance metric cosine"
      )

    num_points, dimensions = vectors_metadata_from_file(vector_bin_path)

    index_prefix_path = os.path.join(index_directory, index_prefix)

    if isinstance(tags, str) and tags != "":
        use_tags = True
        shutil.copy(tags, index_prefix_path + ".tags")
    elif not isinstance(tags, str):
        use_tags = True
        tags_as_array = _castable_dtype_or_raise(tags, expected=np.uint32)
        _assert(len(tags_as_array.shape) == 1, "Provided tags must be 1 dimensional")
        _assert(
            tags_as_array.shape[0] == num_points,
            "Provided tags must contain an identical population to the number of points, "
            f"{tags_as_array.shape[0]=}, {num_points=}",
        )
        tags_to_file(index_prefix_path + ".tags", tags_as_array)
    else:
        use_tags = False

    _native_dap.build_memory_index(
        data_type=vector_dap_dtype,
        metric=dap_metric,
        data_path=vector_bin_path,
        index_path=index_prefix_path,
        l=complexity,
        r=graph_degree,
        alpha=alpha,
        num_start_pts=num_start_pts,
        num_threads=num_threads,
        num_tasks=num_tasks,
        build_pq_bytes=num_pq_bytes,
        graph_slack_factor=graph_slack_factor,
        max_fp_vecs_per_prune=max_fp_vecs_per_prune,
        backedge_ratio=backedge_ratio,
        pq_seed=pq_seed,
        insert_minibatch_size=insert_minibatch_size,
    )

    _write_index_metadata(
        index_prefix_path, vector_dtype_actual, dap_metric, num_points, dimensions
    )