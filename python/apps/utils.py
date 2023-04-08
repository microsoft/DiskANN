# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np

def get_bin_metadata(bin_file):
    array = np.fromfile(file=bin_file, dtype=np.uint32, count=2)
    return array[0], array[1]

def bin_to_numpy(dtype, bin_file):
    npts, ndims = get_bin_metadata(bin_file)
    return np.fromfile(file=bin_file, dtype=dtype, offset=8).reshape(npts,ndims)       