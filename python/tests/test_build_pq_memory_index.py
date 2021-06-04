# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
from diskannpy import Metric, Parameters, SinglePrecisionIndex


data_path = "/mnt/SIFT1M/sift_base.bin"
pq_path = "/home/t-sjaiswal/diskann/build/tests/PQ_SIFT1M/test_pq_memory_index"
save_path = "/home/t-sjaiswal/diskann/build/tests/PQ_SIFT1M/test_build_pq_memory_index.bin"

params = Parameters()
params.set("L", 125)
params.set("R", 32)
params.set("C", 750)
params.set("alpha", 1.2)
params.set("saturate_graph", False)
params.set("num_chunks", 32)
params.set("num_threads", 32)

start = time.time()
index = SinglePrecisionIndex(Metric.FAST_L2, data_path)
index.pq_build(data_path, pq_path, params)
end = time.time()

print("Indexing Time: " + str(end - start) + " seconds")

index.save(save_path)
