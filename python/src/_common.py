# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from pathlib import Path
from typing import Type, TypeVar

import numpy as np

from . import _diskannpy as _native_dap

__ALL__ = ["VectorDType"]

_VALID_DTYPES = [np.single, np.float32, np.byte, np.int8, np.ubyte, np.uint8]

VectorDType = TypeVar(
    "VectorDType",
    Type[np.single],
    Type[np.float32],
    Type[np.ubyte],
    Type[np.uint8],
    Type[np.byte],
    Type[np.int8],
)


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
    else:
        raise ValueError("metric must be one of 'l2' or 'mips'")


def _assert_dtype(vectors: np.dtype, name: str):
    _assert(
        vectors in _VALID_DTYPES,
        name
        + " must be of one of type {(np.single, np.float32), (np.byte, np.int8), (np.ubyte, np.uint8)}",
    )


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
