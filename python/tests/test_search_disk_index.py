# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
import argparse
import numpy as np
import diskannpy


parser = argparse.ArgumentParser()
parser.add_argument('query_path', type=str, help='Path to the input query set of vectors.')
parser.add_argument('ground_truth_path', type=str, help='Path to the input groundtruth set.')
parser.add_argument('index_path_prefix', type=str, help='Path prefix for index files.')
parser.add_argument('output_path_prefix', type=str, help='Prefix for the generated output files.')
args = parser.parse_args()

recall_at = 10
# Use multi-threaded search only for batch mode.
num_threads = 1
single_query_mode = True
l_search = [40, 50, 60, 70, 80, 90, 100, 110, 120]
W = 4

query_data = diskannpy.VectorFloat()
ground_truth_ids = diskannpy.VectorUnsigned()
ground_truth_dists = diskannpy.VectorFloat()

num_queries, query_dims, query_aligned_dims = diskannpy.load_aligned_bin_float(args.query_path, query_data)
num_ground_truth, ground_truth_dims = diskannpy.load_truthset(args.ground_truth_path, ground_truth_ids, ground_truth_dists)

index = diskannpy.DiskANNFloatIndex()
index.load_index(index_path_prefix = args.index_path_prefix)
print("Index Loaded")

#index.optimize_graph()
#print("Graph Optimization Completed")

if single_query_mode:
    print("Ls     QPS     Mean Latency (mus)     99.9 Latency     Recall@10")
    print("================================================================")
    for i, L in enumerate(l_search):
        latency_stats = []
        query_result_ids = diskannpy.VectorUnsigned()
        query_result_dists = diskannpy.VectorUnsigned()
        s = time.time()

        for j in range(num_queries):
            qs = time.time()
            index.search(query_data, query_aligned_dims, recall_at, L, W,
                         query_result_ids + j * recall_at, query_result_dists + j * recall_at)
            qe = time.time()
            latency_stats.append(float((qe - qs) * 1000000))

        e = time.time()
        qps = (num_queries / (e - s))
        recall = diskannpy.calculate_recall(num_queries, ground_truth_ids,
                                    ground_truth_dists, ground_truth_dims,
                                    query_result_ids, recall_at,
                                    recall_at)
        latency_stats.sort()
        mean_latency = sum(latency_stats) / num_queries
        print(str(L) + "{:>10}".format("{:.2f}".format(qps)) +
            "{:>15}".format("{:.2f}".format(mean_latency)) +
            "{:>20}".format("{:.2f}".format(latency_stats[int((0.999 * num_queries))]))
            + "{:>15}".format("{:.2f}".format(recall)))

        result_path = args.output_path_prefix + "_" + str(L) + "_idx_uint32.bin"
        diskannpy.save_bin_u32(result_path, query_result_ids, num_queries, recall_at)
else:
    print("Ls     QPS     Mean Latency (mus)   Recall@10")
    print("=============================================")
    for i, L in enumerate(l_search):
        diskannpy.set_num_threads(num_threads)

        qs = time.time()
        query_result_ids = index.batch_search(query_data, num_queries, query_aligned_dims, recall_at, L, W, query_result_ids, query_result_dists)
        qe = time.time()
        latency_stats = float((qe - qs) * 1000000)

        query_result_ids = diskannpy.VectorUnsigned(query_result_ids)
        qps = (num_queries / (qe - qs))
        recall = diskannpy.calculate_recall(num_queries, ground_truth_ids,
                                     ground_truth_dists, ground_truth_dims,
                                     query_result_ids, recall_at,
                                     recall_at)
        mean_latency = latency_stats / num_queries
        print(str(L) + "{:>10}".format("{:.2f}".format(qps)) +
              "{:>15}".format("{:.2f}".format(mean_latency)) +
              "{:>15}".format("{:.2f}".format(recall)))

        result_path = args.output_path_prefix + "_" + str(L) + "_idx_uint32.bin"
        diskannpy.save_bin_u32(result_path, query_result_ids, num_queries, recall_at)