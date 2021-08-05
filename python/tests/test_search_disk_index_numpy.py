# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT license.
import time
import argparse
import numpy as np
import diskannpy


parser = argparse.ArgumentParser()
parser.add_argument('query_path', type=str, help='Path to the input query set of vectors.')
parser.add_argument('ground_truth_path', type=str, help='Path to the input groundtruth set.')
parser.add_argument('index_path_prefix', type=str, help='Path prefix for index files.')
parser.add_argument('K', type=int, help='k value for recall@K.')
parser.add_argument('W', type=int, help='Beamwidth for search.')
parser.add_argument('T', type=int, help='Number of threads to use for search.')

args = parser.parse_args()
args = parser.parse_args()

recall_at = args.K
W = args.W
# Use multi-threaded search only for batch mode.
num_threads = args.T
l_search = [40, 50, 60, 70, 80, 90, 100, 110, 120]


query_data = diskannpy.VectorFloat()
ground_truth_ids = diskannpy.VectorUnsigned()
ground_truth_dists = diskannpy.VectorFloat()

num_queries, query_dims, query_aligned_dims = diskannpy.load_aligned_bin_float(args.query_path, query_data)
num_ground_truth, ground_truth_dims = diskannpy.load_truthset(args.ground_truth_path, ground_truth_ids, ground_truth_dists)

query_data_numpy = np.zeros((num_queries,query_aligned_dims), dtype=np.float32)
for i in range(0, num_queries):
    for d in range(0, query_dims):
        query_data_numpy[i,d] = query_data[i * query_aligned_dims + d]
        
index = diskannpy.DiskANNFloatIndex()
index.load_index(args.index_path_prefix, num_threads)
print("Index Loaded")


print("Ls     QPS     Recall@10")
print("========================")
for i, L in enumerate(l_search):
    diskannpy.omp_set_num_threads(num_threads)

    qs = time.time()
    ids, dists = index.batch_search_numpy_input(query_data_numpy, query_aligned_dims, 
                                                num_queries, recall_at, L, W, num_threads)
    qe = time.time()
    latency_stats = float((qe - qs) * 1000000)
    qps = (num_queries / (qe - qs))

    recall = diskannpy.calculate_recall_numpy_input(num_queries, ground_truth_ids,
                                                    ground_truth_dists, ground_truth_dims,
                                                    ids, recall_at, recall_at)
    mean_latency = latency_stats / num_queries
    print(str(L) + "{:>10}".format("{:.2f}".format(qps)) + "{:>15}".format("{:.2f}".format(recall)))
