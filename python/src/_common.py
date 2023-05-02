# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from pathlib import Path
from typing import List, NamedTuple, Type, TypeVar

import numpy as np
import numpy.typing as npt

from . import _diskannpy as _native_dap

__ALL__ = ["VectorDType"]

_VALID_DTYPES = [np.single, np.float32, np.byte, np.int8, np.ubyte, np.uint8]

VectorDType = TypeVar(
    "VectorDType",
    Type[np.single],
    Type[np.ubyte],
    Type[np.byte],
)

VectorLike = TypeVar(
    "VectorLike",
    Type[List[int]],
    Type[List[float]],
    Type[npt.NDArray[VectorDType]]
)

VectorLikeBatch = TypeVar(
    "VectorLikeBatch",
    Type[List[List[int]]],
    Type[List[List[float]]],
    Type[npt.NDArray[VectorDType]]
)

VectorIdentifier = TypeVar(
    "VectorIdentifier",
    Type[int],
    Type[np.uintc]
)

VectorIdentifierBatch = TypeVar(
    "VectorIdentifierBatch",
    Type[List[int]],
    Type[List[np.uintc]],
    Type[npt.NDArray[np.uintc]]
)


class QueryResponse(NamedTuple):
    distances: np.ndarray
    indices: np.ndarray


class QueryResponseBatch(NamedTuple):
    distances: np.ndarray
    indices: np.ndarray


def valid_dtype(dtype: str) -> VectorDType:
    _assert_dtype(dtype)
    if np.can_cast(dtype, np.ubyte):
        return np.ubyte
    if np.can_cast(dtype, np.byte):
        return np.byte
    if np.can_cast(dtype, np.single):
        return np.single


def _assert(statement_eval: bool, message: str):
    if not statement_eval:
        raise ValueError(message)


def _get_valid_metric(metric: str) -> _native_dap.Metric:
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


def _assert_dtype(dtype: np.dtype):
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


__MAX_UINT_VAL = 4_294_967_295


def _assert_is_positive_uint32(test_value: int, parameter: str):
    _assert(
        0 < test_value < __MAX_UINT_VAL,
        f"{parameter} must be a positive integer in the uint32 range",
    )


def _assert_is_nonnegative_uint32(test_value: int, parameter: str):
    _assert(
        -1 < test_value < __MAX_UINT_VAL,
        f"{parameter} must be a non-negative integer in the uint32 range",
    )


def _assert_existing_directory(path: str, parameter: str):
    _path = Path(path)
    _assert(
        _path.exists() and _path.is_dir(), f"{parameter} must be an existing directory"
    )


def _assert_existing_file(path: str, parameter: str):
    _path = Path(path)
    _assert(_path.exists() and _path.is_file(), f"{parameter} must be an existing file")
