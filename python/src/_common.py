# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from enum import Enum
from pathlib import Path
from typing import List, Literal, NamedTuple, Tuple, Type, Union

import numpy as np
import numpy.typing as npt

from . import _diskannpy as _native_dap

__ALL__ = [
    "DistanceMetric",
    "QueryResponse",
    "QueryResponseBatch",
    "VectorDType",
    "VectorLike",
    "VectorLikeBatch",
    "VectorIdentifier",
    "VectorIdentifierBatch"
]

_VALID_DTYPES = [np.float32, np.int8, np.uint8]

DistanceMetric = Literal["l2", "mips", "cosine"]
VectorDType = Union[Type[np.float32], Type[np.int8], Type[np.uint8]]
VectorLike = Union[List[int], List[float], npt.NDArray[VectorDType]]
VectorLikeBatch = Union[List[List[int]], List[List[float]], npt.NDArray[VectorDType]]
VectorIdentifier = Union[int, np.uintc]
VectorIdentifierBatch = Union[List[int], List[np.uintc], npt.NDArray[np.uintc]]


class QueryResponse(NamedTuple):
    """
    Tuple with two values, distances and indices. Both are 1d arrays and positionally correspond
    """
    distances: np.ndarray
    indices: np.ndarray


class QueryResponseBatch(NamedTuple):
    """
    Tuple with two values, distances and indices. Both are 2d arrays, with dimensionality determined by the
    rows corresponding to the number of queries made, and the columns corresponding to the k neighbors
    requested. The two 2d arrays have an implicit, position-based relationship
    """
    distances: np.ndarray
    indices: np.ndarray


def valid_dtype(dtype: Type) -> VectorDType:
    _assert_dtype(dtype)
    if np.can_cast(dtype, np.uint8):
        return np.uint8
    if np.can_cast(dtype, np.int8):
        return np.int8
    if np.can_cast(dtype, np.float32):
        return np.float32


def _assert(statement_eval: bool, message: str):
    if not statement_eval:
        raise ValueError(message)


def _valid_metric(metric: str) -> _native_dap.Metric:
    if not isinstance(metric, str):
        raise ValueError("metric must be a string")
    if metric.lower() == "l2":
        return _native_dap.L2
    elif metric.lower() == "mips":
        return _native_dap.INNER_PRODUCT
    elif metric.lower() == "cosine":
        return _native_dap.COSINE
    else:
        raise ValueError("metric must be one of 'l2', 'mips', or 'cosine'")


def _assert_dtype(dtype: Type):
    _assert(
        any(np.can_cast(dtype, _dtype) for _dtype in _VALID_DTYPES),
        f"Vector dtype must be of one of type {{(np.single, np.float32), (np.byte, np.int8), (np.ubyte, np.uint8)}}",
    )


def _castable_dtype_or_raise(
    data: VectorLike,
    expected: np.dtype,
    message: str
) -> np.ndarray:
    _assert_dtype(expected)
    if isinstance(data, list):
        return np.array(data, dtype=expected)  # may result in an overflow and invalid data, but at least warns
    try:
        _vectors = data.astype(dtype=expected, casting="safe", copy=False)  # we would prefer no copy
    except TypeError as e:
        e.args = (message, *e.args)
        raise
    return _vectors


def _assert_2d(vectors: np.ndarray, name: str):
    _assert(len(vectors.shape) == 2, f"{name} must be 2d numpy array")


__MAX_UINT32_VAL = 4_294_967_295


def _assert_is_positive_uint32(test_value: int, parameter: str):
    _assert(
        test_value is not None and 0 < test_value < __MAX_UINT32_VAL,
        f"{parameter} must be a positive integer in the uint32 range",
    )


def _assert_is_nonnegative_uint32(test_value: int, parameter: str):
    _assert(
        test_value is not None and -1 < test_value < __MAX_UINT32_VAL,
        f"{parameter} must be a non-negative integer in the uint32 range",
    )


def _assert_is_nonnegative_uint64(test_value: int, parameter: str):
    _assert(
        -1 < test_value,
        f"{parameter} must be a non-negative integer in the uint64 range",
    )


def _assert_existing_directory(path: str, parameter: str):
    _path = Path(path)
    _assert(
        _path.exists() and _path.is_dir(), f"{parameter} must be an existing directory"
    )


def _assert_existing_file(path: str, parameter: str):
    _path = Path(path)
    _assert(_path.exists() and _path.is_file(), f"{parameter} must be an existing file")


class _DataType(Enum):
    FLOAT32 = 0
    INT8 = 1
    UINT8 = 2

    @classmethod
    def from_type(cls, vector_dtype: VectorDType) -> "DataType":
        if vector_dtype == np.single:
            return cls.FLOAT32
        if vector_dtype == np.byte:
            return cls.INT8
        if vector_dtype == np.ubyte:
            return cls.UINT8

    def to_type(self) -> VectorDType:
        if self is _DataType.FLOAT32:
            return np.float32
        if self is _DataType.INT8:
            return np.int8
        if self is _DataType.UINT8:
            return np.uint8


class _Metric(Enum):
    L2 = 0
    MIPS = 1
    COSINE = 2

    @classmethod
    def from_native(cls, metric: _native_dap.Metric) -> "_Metric":
        if metric == _native_dap.L2:
            return cls.L2
        if metric == _native_dap.INNER_PRODUCT:
            return cls.MIPS
        if metric == _native_dap.COSINE:
            return cls.COSINE

    def to_native(self) -> _native_dap.Metric:
        if self is _Metric.L2:
            return _native_dap.L2
        if self is _Metric.MIPS:
            return _native_dap.INNER_PRODUCT
        if self is _Metric.COSINE:
            return _native_dap.COSINE


def _build_metadata_path(index_path_and_prefix: str) -> str:
    return index_path_and_prefix + "_metadata.bin"


def _write_index_metadata(
        index_path_and_prefix: str,
        dtype: VectorDType,
        metric: _native_dap.Metric,
        num_points: int,
        dimensions: int
):
    np.array(
        [_DataType.from_type(dtype).value, _Metric.from_native(metric).value, num_points, dimensions],
        dtype=np.uint64
    ).tofile(_build_metadata_path(index_path_and_prefix))


def _read_index_metadata(index_path_and_prefix: str) -> Tuple[VectorDType, _native_dap.Metric, np.uint64, np.uint64]:
    path = _build_metadata_path(index_path_and_prefix)
    _assert_existing_file(path, "internal error: metadata path")
    metadata = np.fromfile(path, dtype=np.uint64, count=-1)
    return _DataType(int(metadata[0])).to_type(), _Metric(int(metadata[1])).to_native(), metadata[2], metadata[3]


def _valid_index_prefix(index_directory: str, index_prefix: str) -> str:
    _assert_existing_directory(index_directory, "index_directory")
    _assert(index_prefix != "", "index_prefix cannot be an empty string")
    return os.path.join(index_directory, index_prefix)
