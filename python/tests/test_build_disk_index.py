# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
import argparse
from diskannpy import Metric, Parameters, DiskANNFloatIndex


parser = argparse.ArgumentParser()
parser.add_argument('data_path', type=str, help='Path to the input base set of vectors.')
parser.add_argument('save_path', type=str, help='Path to the built index.')
parser.add_argument('R', type=int, help='Graph degree.')
parser.add_argument('L', type=int, help='Index build complexity.')
parser.add_argument('B', type=float, help='Memory budget in GB for the final index.')
parser.add_argument('M', type=float, help='Memory budget in GB for the index construction.')
parser.add_argument('T', type=int, help='Number of threads for index construction.')

args = parser.parse_args()

start = time.time()
index = DiskANNFloatIndex()
index.build(args.data_path, args.save_path, args.R, args.L, args.B, args.M, args.T)
end = time.time()

print("Indexing Time: " + str(end - start) + " seconds")