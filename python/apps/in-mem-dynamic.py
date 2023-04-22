# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse

import diskannpy
import numpy as np
import utils


def insert_and_search(
    dtype_str,
    indexdata_file,
    querydata_file,
    Lb,
    graph_degree,
    K,
    Ls,
    num_threads,
    gt_file,
):
    npts, ndims = utils.get_bin_metadata(indexdata_file)

    if dtype_str == "float":
        index = diskannpy.DynamicMemoryIndex(
            "l2", np.float32, ndims, npts, Lb, graph_degree, False
        )
        queries = utils.bin_to_numpy(np.float32, querydata_file)
        data = utils.bin_to_numpy(np.float32, indexdata_file)
    elif dtype_str == "int8":
        index = diskannpy.DynamicMemoryIndex(
            "l2", np.int8, ndims, npts, Lb, graph_degree
        )
        queries = utils.bin_to_numpy(np.int8, querydata_file)
        data = utils.bin_to_numpy(np.int8, indexdata_file)
    elif dtype_str == "uint8":
        index = diskannpy.DynamicMemoryIndex(
            "l2", np.uint8, ndims, npts, Lb, graph_degree
        )
        queries = utils.bin_to_numpy(np.uint8, querydata_file)
        data = utils.bin_to_numpy(np.uint8, indexdata_file)
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    tags = np.zeros(npts, dtype=int)
    for i in range(npts):
        tags[i] = i+1
    index.batch_insert(data, tags, npts, num_threads)
    print("batch_insert complete")

    delete_tags = np.random.choice(range(1,npts+1,1), size=int(0.5*npts), replace=False)
    for tag in delete_tags:
        index.mark_deleted(tag)
    print("mark deletion completed")

    index.consolidate_delete()
    print("consolidation completed")

    for tag in delete_tags:
        index.insert(data[tag-1, :], tag)    
    print("re-insertion completed")

    tags, dists = index.batch_search(queries, K, Ls, num_threads)
    res_ids = tags - 1

    if gt_file != "":
        recall = utils.calculate_recall_from_gt_file(K, res_ids, gt_file)
        print(f"recall@{K} is {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="in-mem-dynamic",
        description="Inserts points dynamically in a clustered order and search from vectors in a file.",
    )

    parser.add_argument("-d", "--data_type", required=True)
    parser.add_argument("-i", "--indexdata_file", required=True)
    parser.add_argument("-q", "--querydata_file", required=True)
    parser.add_argument("-Lb", "--Lbuild", default=50)
    parser.add_argument("-Ls", "--Lsearch", default=50)
    parser.add_argument("-R", "--graph_degree", default=32)
    parser.add_argument("-T", "--num_threads", default=8)
    parser.add_argument("-K", default=10)
    parser.add_argument("--gt_file", default="")
    args = parser.parse_args()

    insert_and_search(
        args.data_type,
        args.indexdata_file,
        args.querydata_file,
        args.Lbuild,
        args.graph_degree,  # Build args
        args.K,
        args.Lsearch,
        args.num_threads,  # search args
        args.gt_file,
    )