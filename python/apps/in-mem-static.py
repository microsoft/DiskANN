# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import sys, getopt
import numpy
import diskannpy

def get_bin_metadata(bin_file):
    array = numpy.fromfile(file=bin_file, dtype=numpy.uint32, count=2)
    return array[0], array[1]

def bin_to_numpy(dtype, bin_file):
    npts, ndims = get_bin_metadata(bin_file)
    return numpy.fromfile(file=bin_file, dtype=dtype, offset=8).reshape(npts,ndims)         


def main(argv):
   indexdata_file = ''
   querydata_file = ''
   dtype_str = ''
   opts, args = getopt.getopt(argv,"hd:i:q:",["indexdata_file","querydata_file="])
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <indexdata_file> -q <querydata_file>')
         sys.exit()
      elif opt in ("-i", "--indexdata_file"):
         indexdata_file = arg
      elif opt in ("-q", "--querydata_file"):
         querydata_file = arg
      elif opt in ("-d", "--data_type"):
          dtype_str = arg 

   if dtype_str == "float32":
       index = diskannpy.StaticMemoryIndex("l2", numpy.float32, indexdata_file, 32, 32)
       queries = bin_to_numpy(numpy.float32, querydata_file)
       index.batch_search(queries, 10, 50, 8)
   elif dtype_str == "int8":
       index = diskannpy.StaticMemoryIndex("l2", numpy.int32, indexdata_file, 32, 32)
       queries = bin_to_numpy(numpy.int8, querydata_file)
       index.batch_search(queries, 10, 50, 8)
   elif dtype_str == "uint8":
       index = diskannpy.StaticMemoryIndex("l2", numpy.uint8, indexdata_file, 32, 32)
       queries = bin_to_numpy(numpy.uint8, querydata_file)
       index.batch_search(queries, 10, 50, 8)