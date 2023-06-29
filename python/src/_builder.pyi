# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from typing import BinaryIO, overload, Optional

import numpy as np

from ._common import DistanceMetric, VectorDType, VectorLikeBatch, VectorIdentifierBatch

def numpy_to_diskann_file(vectors: np.ndarray, file_handler: BinaryIO): ...
@overload
def build_disk_index(
    data: str,
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int,
    vector_dtype: VectorDType,
    index_prefix: str,
): ...
@overload
def build_disk_index(
    data: VectorLikeBatch,
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    search_memory_maximum: float,
    build_memory_maximum: float,
    num_threads: int,
    pq_disk_bytes: int,
    index_prefix: str,
): ...
@overload
def build_memory_index(
    data: VectorLikeBatch,
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    alpha: float,
    num_threads: int,
    use_pq_build: bool,
    num_pq_bytes: int,
    use_opq: bool,
    label_file: str,
    universal_label: str,
    filter_complexity: int,
    tags: Optional[VectorIdentifierBatch],
    index_prefix: str,
): ...
@overload
def build_memory_index(
    data: str,
    distance_metric: DistanceMetric,
    index_directory: str,
    complexity: int,
    graph_degree: int,
    alpha: float,
    num_threads: int,
    use_pq_build: bool,
    num_pq_bytes: int,
    use_opq: bool,
    vector_dtype: VectorDType,
    label_file: str,
    universal_label: str,
    filter_complexity: int,
    tags: Optional[str],
    index_prefix: str,
): ...

