# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import time
import vamanapy as vp


data_path = "/mnt/SIFT1M/sift_base.bin"
query_path = "/mnt/SIFT1M/sift_query.bin"
ground_truth_path = "/home/t-sjaiswal/diskann/build/tests/SIFT1M/sift_groundtruth.bin"
memory_index_path = "/home/t-sjaiswal/diskann/build/tests/SIFT1M/test_build_memory_index.bin"
output_path_prefix = "/home/t-sjaiswal/diskann/build/tests/SIFT1M/test_search_memory_index"

recall_at = 10
num_threads = 32
l_search = [40, 50, 60, 70, 80, 90, 100, 110, 120]

query_data = vp.VectorFloat()
ground_truth_ids = vp.VectorUnsigned()
ground_truth_dists = vp.VectorFloat()

num_queries, query_dims, query_aligned_dims = vp.load_aligned_bin_float(query_path, query_data)
num_ground_truth, ground_truth_dims = vp.load_truthset(ground_truth_path, ground_truth_ids, ground_truth_dists)

index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
index.load(file_name = memory_index_path)
print("Index Loaded")

index.optimize_graph()
print("Graph Optimization Completed")

print("Ls     QPS     Mean Latency (mus)     99.9 Latency     Recall@10")
print("================================================================")

for i, L in enumerate(l_search):
    latency_stats = []
    query_result_ids = vp.VectorUnsigned()
    vp.set_num_threads(num_threads)
    s = time.time()

    for j in range(num_queries):
        qs = time.time()
        index.search_with_optimized_graph(query_data, j * query_aligned_dims,
                                          recall_at, num_queries, L,
                                          query_result_ids,
                                          j * recall_at)
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
