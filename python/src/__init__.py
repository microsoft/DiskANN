# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
# Documentation Overview
`diskannpy` is mostly structured around 2 distinct processes: [Building Functions](#index-builders) and [Search Classes](#search-classes)

It also includes a few nascent [utilities](#utilities).

And lastly, it makes substantial use of type hints, with various shorthand [type aliases](#parameter-and-response-type-aliases) documented. 
When reading the `diskannpy` code we refer to the type aliases, though `pydoc` helpfully expands them. 

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

## Utilities
- `numpy_to_diskann_file` - Turns a 2 dimensional `numpy.typing.NDArray[VectorDType]` with shape `(number_of_points, vector_dim)`
- `vectors_from_binary` - Reads a `diskann` bin file representing stored vectors into a numpy ndarray.
- `vector_file_metadata` - Reads a `diskannpy` metadata file, containing the index's distance metric, vector dtype, number of points, and vector dimensionality and returns them. Useful for inspecting what an index was saved with without loading the whole index. 
"""

from ._builder import (
    build_disk_index,
    build_memory_index,
    numpy_to_diskann_file,
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
from ._files import vectors_from_binary, vector_file_metadata
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
    "numpy_to_diskann_file", 
    "valid_dtype",
    "vectors_from_binary",
    "vector_file_metadata",
]

