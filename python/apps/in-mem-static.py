# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import argparse
from xml.dom.pulldom import default_bufsize
import diskannpy
import numpy as np
import utils

def build_and_search(dtype_str, indexdata_file, querydata_file,
                     Lb, graph_degree,
                     K, Ls, num_threads,
                     gt_file):
    if dtype_str == "float":
        index = diskannpy.StaticMemoryIndex("l2", np.float32, indexdata_file, Lb, graph_degree)
        queries = utils.bin_to_numpy(np.float32, querydata_file)
    elif dtype_str == "int8":
        index = diskannpy.StaticMemoryIndex("l2", np.int8, indexdata_file, Lb, graph_degree)
        queries = utils.bin_to_numpy(np.int8, querydata_file)
    elif dtype_str == "uint8":
        index = diskannpy.StaticMemoryIndex("l2", np.uint8, indexdata_file, Lb, graph_degree)
        queries = utils.bin_to_numpy(np.uint8, querydata_file)
    else:
        raise ValueError("data_type must be float, int8 or uint8")
   
    ids, dists = index.batch_search(queries, 10, Ls, num_threads)

    if gt_file != '':
        recall = utils.calculate_recall_from_gt_file(K, ids, gt_file)
        print(f'recall@{K} is {recall}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='in-mem-static', 
                                     description='Static in-memory build and search from vectors in a file')

    parser.add_argument('-d', '--data_type', required=True)
    parser.add_argument('-i', '--indexdata_file', required=True)
    parser.add_argument('-q', '--querydata_file', required=True)
    parser.add_argument('-Lb', '--Lbuild', default=50)
    parser.add_argument('-Ls', '--Lsearch', default=50)
    parser.add_argument('-R', '--graph_degree', default=32)
    parser.add_argument('-T', '--num_threads', default=8)
    parser.add_argument('-K', default=10)
    parser.add_argument('--gt_file', default='')
    args = parser.parse_args()

    build_and_search(args.data_type, args.indexdata_file, args.querydata_file,
                     args.Lbuild,  args.graph_degree, # Build args
                     args.K, args.Lsearch, args.num_threads, # search args
                     args.gt_file) 
