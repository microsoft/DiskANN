# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from xml.dom.pulldom import default_bufsize

import diskannpy
import numpy as np
import utils


def build_and_search(
    metric,
    dtype_str,
    index_directory,
    indexdata_file,
    querydata_file,
    Lb,
    graph_degree,
    K,
    Ls,
    num_threads,
    gt_file,
    index_prefix,
    search_only
):
    if dtype_str == "float":
        dtype = np.single
    elif dtype_str == "int8":
        dtype = np.byte
    elif dtype_str == "uint8":
        dtype = np.ubyte
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    # build index
    if not search_only:
        diskannpy.build_memory_index(
            data=indexdata_file,
            distance_metric=metric,
            vector_dtype=dtype,
            index_directory=index_directory,
            complexity=Lb,
            graph_degree=graph_degree,
            num_threads=num_threads,
            index_prefix=index_prefix,
            alpha=1.2,
            use_pq_build=False,
            num_pq_bytes=8,
            use_opq=False,
        )

    # ready search object
    index = diskannpy.StaticMemoryIndex(
        distance_metric=metric,
        vector_dtype=dtype,
        index_directory=index_directory,
        num_threads=num_threads,  # this can be different at search time if you would like
        initial_search_complexity=Ls,
        index_prefix=index_prefix
    )

    queries = utils.bin_to_numpy(dtype, querydata_file)

    timer = utils.timer()
    ids, dists = index.batch_search(queries, 10, Ls, num_threads)
    query_time = timer.elapsed()
    qps = round(queries.shape[0]/query_time, 1)
    print('Batch searched', queries.shape[0], 'in', query_time, 's @', qps, 'QPS')

    if gt_file != "":
        recall = utils.calculate_recall_from_gt_file(K, ids, gt_file)
        print(f"recall@{K} is {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="in-mem-static",
        description="Static in-memory build and search from vectors in a file",
    )

    parser.add_argument("-m", "--metric", required=False, default="l2")
    parser.add_argument("-d", "--data_type", required=True)
    parser.add_argument("-id", "--index_directory", required=False, default=".")
    parser.add_argument("-i", "--indexdata_file", required=True)
    parser.add_argument("-q", "--querydata_file", required=True)
    parser.add_argument("-Lb", "--Lbuild", default=50, type=int)
    parser.add_argument("-Ls", "--Lsearch", default=50, type=int)
    parser.add_argument("-R", "--graph_degree", default=32, type=int)
    parser.add_argument("-T", "--num_threads", default=8, type=int)
    parser.add_argument("-K", default=10, type=int)
    parser.add_argument("-G", "--gt_file", default="")
    parser.add_argument("-ip", "--index_prefix", required=False, default="ann")
    parser.add_argument("--search_only", required=False, default=False)
    args = parser.parse_args()

    build_and_search(
        args.metric,
        args.data_type,
        args.index_directory.strip(),
        args.indexdata_file.strip(),
        args.querydata_file.strip(),
        args.Lbuild,
        args.graph_degree,  # Build args
        args.K,
        args.Lsearch,
        args.num_threads,  # search args
        args.gt_file,
        args.index_prefix,
        args.search_only
    )
