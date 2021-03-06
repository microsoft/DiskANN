# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
from vamanapy import Metric, Parameters, SinglePrecisionIndex


data_path = "/mnt/SIFT1M/sift_base.bin"
save_path = "/home/t-sjaiswal/diskann/build/tests/SIFT1M/test_build_memory_index.bin"

params = Parameters()
params.set("L", 125)
params.set("R", 32)
params.set("C", 750)
params.set("alpha", 1.2)
params.set("saturate_graph", False)
params.set("num_threads", 32)

start = time.time()
index = SinglePrecisionIndex(Metric.FAST_L2, data_path)
index.build(params, [])
end = time.time()

print("Indexing Time: " + str(end - start) + " seconds")

index.save(save_path)
