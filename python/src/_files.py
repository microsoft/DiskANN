# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import numpy.typing as npt

from typing import NamedTuple

from ._common import VectorDType, _assert_existing_file


class Metadata(NamedTuple):
    num_vectors: int
    dimensions: int


def vector_file_metadata(vector_file: str) -> Metadata:
    _assert_existing_file(vector_file, "vector_file")
    points, dims = np.fromfile(file=vector_file, dtype=np.uintc, count=2)
    return Metadata(points, dims)


def vectors_from_binary(vector_file: str, dtype: VectorDType) -> npt.NDArray[VectorDType]:
    points, dims = vector_file_metadata(vector_file)
    return np.fromfile(file=vector_file, dtype=dtype, offset=8).reshape(points, dims)

