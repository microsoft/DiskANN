# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
import numpy as np
import vamanapy as vp


data_path = "/mnt/SIFT1M/sift_base.bin"
query_path = "/mnt/SIFT1M/sift_query.bin"
ground_truth_path = "/home/t-sjaiswal/diskann/build/tests/SIFT1M/sift_groundtruth.bin"
memory_index_path = "/home/t-sjaiswal/diskann/build/tests/PQ_SIFT1M/test_build_pq_memory_index.bin"
pq_path = "/home/t-sjaiswal/diskann/build/tests/PQ_SIFT1M/test_pq_memory_index"
output_path_prefix = "/home/t-sjaiswal/diskann/build/tests/PQ_SIFT1M/test_search_pq_memory_index"

recall_at = 10
# Use multi-threaded search only for batch mode.
num_threads = 1
single_query_mode = True
l_search = [40, 50, 60, 70, 80, 90, 100, 110, 120]

query_data = vp.VectorFloat()
ground_truth_ids = vp.VectorUnsigned()
ground_truth_dists = vp.VectorFloat()

num_queries, query_dims, query_aligned_dims = vp.load_aligned_bin_float(query_path, query_data)
num_ground_truth, ground_truth_dims = vp.load_truthset(ground_truth_path, ground_truth_ids, ground_truth_dists)

index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
index.load(file_name = memory_index_path)
print("Index Loaded")
index.pq_load(pq_prefix_path = pq_path)
print("PQ Data Loaded")

index.optimize_graph()
print("Graph Optimization Completed")

if single_query_mode:
    print("Ls     QPS     Mean Latency (mus)     99.9 Latency     Recall@10")
    print("================================================================")
    for i, L in enumerate(l_search):
        latency_stats = []
        query_result_ids = vp.VectorUnsigned()
        s = time.time()

        for j in range(num_queries):
            qs = time.time()
            index.pq_search(query_data, j * query_aligned_dims, recall_at,
                            num_queries, L, query_result_ids, j * recall_at)
            qe = time.time()
            latency_stats.append(float((qe - qs) * 1000000))

        e = time.time()
        qps = (num_queries / (e - s))
        recall = vp.calculate_recall(num_queries, ground_truth_ids,
                                     ground_truth_dists, ground_truth_dims,
                                     query_result_ids, recall_at,
                                     recall_at)
        latency_stats.sort()
        mean_latency = sum(latency_stats) / num_queries
        print(str(L) + "{:>10}".format("{:.2f}".format(qps)) +
              "{:>15}".format("{:.2f}".format(mean_latency)) +
              "{:>20}".format("{:.2f}".format(latency_stats[int((0.999 * num_queries))]))
              + "{:>15}".format("{:.2f}".format(recall)))

        result_path = output_path_prefix + "_" + str(L) + "_idx_uint32.bin"
        vp.save_bin_u32(result_path, query_result_ids, num_queries, recall_at)
else:
    query_data = np.load('/mnt/SIFT1M/sift_query.npy')
    print("Ls     QPS     Mean Latency (mus)   Recall@10")
    print("=============================================")
    for i, L in enumerate(l_search):
        vp.set_num_threads(num_threads)

        qs = time.time()
        query_result_ids = index.pq_batch_numpy_query(query_data, recall_at,
                                                      num_queries, L)
        qe = time.time()
        latency_stats = float((qe - qs) * 1000000)

        query_result_ids = vp.VectorUnsigned(query_result_ids)
        qps = (num_queries / (qe - qs))
        recall = vp.calculate_recall(num_queries, ground_truth_ids,
                                     ground_truth_dists, ground_truth_dims,
                                     query_result_ids, recall_at,
                                     recall_at)
        mean_latency = latency_stats / num_queries
        print(str(L) + "{:>10}".format("{:.2f}".format(qps)) +
              "{:>15}".format("{:.2f}".format(mean_latency)) +
              "{:>15}".format("{:.2f}".format(recall)))

        result_path = output_path_prefix + "_" + str(L) + "_idx_uint32.bin"
        vp.save_bin_u32(result_path, query_result_ids, num_queries, recall_at)
