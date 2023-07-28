# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
# Documentation Overview
`diskannpy` is mostly structured around 2 distinct processes: [Index Builder Functions](#index-builders) and [Search Classes](#search-classes)

It also includes a few nascent [utilities](#utilities).

And lastly, it makes substantial use of type hints, with various shorthand [type aliases](#parameter-and-response-type-aliases) documented. 
When reading the `diskannpy` code we refer to the type aliases, though `pdoc` helpfully expands them.

## Index Builders
- `build_disk_index` - To build an index that cannot fully fit into memory when searching
- `build_memory_index` - To build an index that can fully fit into memory when searching

## Search Classes
- `StaticMemoryIndex` - for indices that can fully fit in memory and won't be changed during the search operations
- `StaticDiskIndex` - for indices that cannot fully fit in memory, thus relying on disk IO to search, and also won't be changed during search operations
- `DynamicMemoryIndex` - for indices that can fully fit in memory and will be mutated via insert/deletion operations as well as search operations

## Parameter Defaults
- `diskannpy.defaults` - Default values exported from the C++ extension for Python users

## Parameter and Response Type Aliases
- `DistanceMetric` - What distance metrics does `diskannpy` support?
- `VectorDType` - What vector datatypes does `diskannpy` support?
- `QueryResponse` - What can I expect as a response to my search?
- `QueryResponseBatch` - What can I expect as a response to my batch search?
- `VectorIdentifier` - What types do `diskannpy` support as vector identifiers? 
- `VectorIdentifierBatch` - A batch of identifiers of the exact same type. The type can change, but they must **all** change.
- `VectorLike` - How does a vector look to `diskannpy`, to be inserted or searched with.
- `VectorLikeBatch` - A batch of those vectors, to be inserted or searched with.
- `Metadata` - DiskANN vector binary file metadata (num_points, vector_dim)

## Utilities
- `vectors_to_file` - Turns a 2 dimensional `numpy.typing.NDArray[VectorDType]` with shape `(number_of_points, vector_dim)` into a DiskANN vector bin file.
- `vectors_from_file` - Reads a DiskANN vector bin file representing stored vectors into a numpy ndarray.
- `vectors_metadata_from_file` - Reads metadata stored in a DiskANN vector bin file without reading the entire file
- `tags_to_file` - Turns a 1 dimensional `numpy.typing.NDArray[VectorIdentifier]` into a DiskANN tags bin file.
- `tags_from_file` - Reads a DiskANN tags bin file representing stored tags into a numpy ndarray.
- `valid_dtype` - Checks if a given vector dtype is supported by `diskannpy`
"""

from ._builder import (
    build_disk_index,
    build_memory_index,
)
from ._common import (
    DistanceMetric,
    QueryResponse,
    QueryResponseBatch,
    VectorDType,
    VectorIdentifier,
    VectorIdentifierBatch,
    VectorLike,
    VectorLikeBatch,
    valid_dtype
)
from . import defaults
from ._dynamic_memory_index import DynamicMemoryIndex
from ._files import vectors_from_file, vectors_metadata_from_file, Metadata, vectors_to_file, tags_from_file, tags_to_file
from ._static_disk_index import StaticDiskIndex
from ._static_memory_index import StaticMemoryIndex


__all__ = [
    "build_disk_index", 
    "build_memory_index",
    "StaticDiskIndex",
    "StaticMemoryIndex",
    "DynamicMemoryIndex",
    "defaults",
    "DistanceMetric", 
    "VectorDType",
    "QueryResponse", 
    "QueryResponseBatch", 
    "VectorIdentifier",
    "VectorIdentifierBatch",
    "VectorLike",
    "VectorLikeBatch",
    "Metadata",
    "vectors_metadata_from_file",
    "vectors_to_file",
    "vectors_from_file",
    "tags_to_file",
    "tags_from_file",
    "valid_dtype",
]

