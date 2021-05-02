# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
import argparse
from vamanapy import Metric, Parameters, SinglePrecisionIndex


parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='Path to the input base set of vectors.')
parser.add_argument('pq_path', type=str, help='Prefix for the pq-quantization generated files.')
parser.add_argument('save_path', type=str, help='Path to the built index.')
args = parser.parse_args()

params = Parameters()
params.set("L", 125)
params.set("R", 32)
params.set("C", 750)
params.set("alpha", 1.2)
params.set("saturate_graph", False)
params.set("num_chunks", 32)
params.set("num_threads", 32)

start = time.time()
index = SinglePrecisionIndex(Metric.FAST_L2, args.data_path)
index.pq_build(args.data_path, args.pq_path, params)
end = time.time()

print("Indexing Time: " + str(end - start) + " seconds")

index.save(args.save_path)
