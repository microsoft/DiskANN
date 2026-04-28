# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
# Parameter Defaults
These parameter defaults are re-exported from the C++ extension module, and used to keep the pythonic wrapper in sync with the C++.
"""

ALPHA = 1.2
""" 
Note that, as ALPHA is a `float32` (single precision float) in C++, when converted into Python it becomes a 
`float64` (double precision float). The actual value is 1.2f. The alpha parameter (>=1) is used to control the nature 
and number of points that are added to the graph. A higher alpha value (e.g., 1.4) will result in fewer hops (and IOs) 
to convergence, but probably more distance comparisons compared to a lower alpha value.
"""
NUM_THREADS = 0
""" Number of threads to use. `0` will use all available detected logical processors """
MAX_OCCLUSION_SIZE = 750
""" 
The maximum number of points that can be occluded by a single point. This is used to  prevent a single point from 
dominating the graph structure. If a point has more than `max_occlusion_size` neighbors closer to it than the current 
point, it will not be added to the graph. This is a tradeoff between index build time and search quality. 
"""
FILTER_COMPLEXITY = 0
""" 
Complexity (a.k.a. `L`) references the size of the list we store candidate approximate neighbors in while doing a 
filtered search. This value must be larger than `k_neighbors`, and larger values tend toward higher recall in the 
resultant ANN search at the cost of more time. 
"""
NUM_FROZEN_POINTS_STATIC = 0
""" Number of points frozen by default in a StaticMemoryIndex """
NUM_FROZEN_POINTS_DYNAMIC = 1
""" Number of points frozen by default in a DynamicMemoryIndex """
SATURATE_GRAPH = False
""" Whether to saturate the graph or not. Default is `True` """
GRAPH_DEGREE = 512
""" 
Graph degree (a.k.a. `R`) is the maximum degree allowed for a node in the index's graph structure. This degree will be 
pruned throughout the course of the index build, but it will never grow beyond this value. Higher R values require 
longer index build times, but may result in an index showing excellent recall and latency characteristics. 
"""
COMPLEXITY = 100
""" 
Complexity (a.k.a `L`) references the size of the list we store candidate approximate neighbors in while doing build
or search tasks. It's used during index build as part of the index optimization processes. It's used in index search 
classes both to help mitigate poor latencies during cold start, as well as on subsequent queries to conduct the search. 
Large values will likely increase latency but also may improve recall, and tuning these values for your particular 
index is certainly a reasonable choice.
"""
USE_PQ = False
"""
Whether to use product quantization in the loading/building memory index. Product quantization is an approximation technique 
that can vastly speed up vector computations and comparisons in a spatial neighborhood, but it is still an approximation
technique. It should be preferred when search times take longer than you can afford for your use case.
"""
NUM_PQ_BYTES = 0
""" 
The number of product quantization bytes to use. More bytes requires more resources in both memory and time, but is 
like to result in better approximations. 
"""
USE_OPQ = False
""" Whether to use Optimized Product Quantization or not. """
BUILD_PQ_BYTES = 0
""" 
Set to a positive value less than the dimensionality of the data to enable faster index build with PQ based distance comparisons.
"""
NUM_OF_PQ_CHUNKS = 1
"""
Number of PQ chunks to split each vector into. Each chunk is compressed to 1 byte during PQ compression. It's the number of bytes 
used for the PQ compressed data stored in-memory. Default is 1 so callers should set this based on their dataset characteristics. 
"""
BACKEDGE_RATIO = 1.0
"""
Ratio of back edges to foward edges added per insert. Should be between 0 and =1.0. 
"""