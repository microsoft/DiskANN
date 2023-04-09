# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

def get_bin_metadata(bin_file):
    array = np.fromfile(file=bin_file, dtype=np.uint32, count=2)
    return array[0], array[1]

def bin_to_numpy(dtype, bin_file):
    npts, ndims = get_bin_metadata(bin_file)
    return np.fromfile(file=bin_file, dtype=dtype, offset=8).reshape(npts,ndims)

def read_gt_file(gt_file):
    nq, K = get_bin_metadata(gt_file)
    ids = np.fromfile(file=gt_file, dtype=np.uint32, offset=8, count=nq*K).reshape(nq,K)
    dists = np.fromfile(file=gt_file, dtype=np.float32, offset=8+nq*K*4, count=nq*K).reshape(nq,K)
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
    gt_ids, gt_dists = utils.read_gt_file(gt_file)
    return utils.calculate_recall(ids, gt_ids, K)