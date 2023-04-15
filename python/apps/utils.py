# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np
from scipy.cluster.vq import vq, kmeans2


def get_bin_metadata(bin_file):
    array = np.fromfile(file=bin_file, dtype=np.uint32, count=2)
    return array[0], array[1]


def bin_to_numpy(dtype, bin_file):
    npts, ndims = get_bin_metadata(bin_file)
    return np.fromfile(file=bin_file, dtype=dtype, offset=8).reshape(npts, ndims)


def numpy_to_bin(array, out_file):
    shape = np.shape(array)
    npts = shape[0].astype(np.uint32)
    ndims = shape[1].astype(np.uint32)
    f = open(out_file, 'wb')
    f.write(npts.tobytes())
    f.write(ndims.tobytes())
    f.write(array.tobytes())
    f.close()

def read_gt_file(gt_file):
    nq, K = get_bin_metadata(gt_file)
    ids = np.fromfile(file=gt_file, dtype=np.uint32, offset=8, count=nq * K).reshape(
        nq, K
    )
    dists = np.fromfile(
        file=gt_file, dtype=np.float32, offset=8 + nq * K * 4, count=nq * K
    ).reshape(nq, K)
    return ids, dists


def calculate_recall(
    result_set_indices: np.ndarray, truth_set_indices: np.ndarray, recall_at: int = 5
) -> float:
    """
    result_set_indices and truth_set_indices correspond by row index. the columns in each row contain the indices of
    the nearest neighbors, with result_set_indices being the approximate nearest neighbor results and truth_set_indices
    being the brute force nearest neighbor calculation via sklearn's NearestNeighbor class.
    :param result_set_indices:
    :param truth_set_indices:
    :param recall_at:
    :return:
    """
    found = 0
    for i in range(0, result_set_indices.shape[0]):
        result_set_set = set(result_set_indices[i][0:recall_at])
        truth_set_set = set(truth_set_indices[i][0:recall_at])
        found += len(result_set_set.intersection(truth_set_set))
    return found / (result_set_indices.shape[0] * recall_at)


def calculate_recall_from_gt_file(K, ids, gt_file) -> float:
    gt_ids, gt_dists = read_gt_file(gt_file)
    return calculate_recall(ids, gt_ids, K)


def cluster_and_permute(dtype_str, indexdata_file, num_clusters):
    npts, ndims = get_bin_metadata(indexdata_file)

    if dtype_str == "float":
        data = bin_to_numpy(np.float32, indexdata_file)
    elif dtype_str == "int8":
        data = bin_to_numpy(np.int8, indexdata_file)
    elif dtype_str == "uint8":
        data = bin_to_numpy(np.uint8, indexdata_file)
    else:
        raise ValueError("data_type must be float, int8 or uint8")

    sample_size = min(100000, npts)
    sample_indices = np.random.choice(range(npts), size = sample_size, replace=False)
    sampled_data = data[sample_indices,:]
    centroids, sample_labels = kmeans2(sampled_data, num_clusters, minit='++', iter=10)
    labels, dist = vq(data, centroids)
    
    count = np.zeros(num_clusters)
    for i in range(npts):
        count[labels[i]] += 1
    print("Cluster counts")
    print(count)

    offsets = np.zeros(num_clusters, dtype=int)
    counters = np.zeros(num_clusters, dtype=int)
    for i in range(1,num_clusters,1):
        offsets[i] = offsets[i-1] + count[i-1]

    permutation = np.zeros(npts)
    for i in range(npts):
        label = labels[i]
        row = offsets[label] + counters[label]
        counters[label] +=1 
        permutation[row] = i

    return offsets, permutation