// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <string.h>

namespace program_options_utils
{
const std::string make_program_description(const char *executable_name, const char *description)
{
    return std::string("\n")
        .append(description)
        .append("\n\n")
        .append("Usage: ")
        .append(executable_name)
        .append(" [OPTIONS]");
}

// Required parameters
const char *DATA_TYPE_DESCRIPTION = "data type, one of {int8, uint8, float} - float is single precision (32 bit)";
const char *DISTANCE_FUNCTION_DESCRIPTION =
    "distance function <l2/mips/fast_l2/cosine>.  'fast l2' and 'mips' only support data_type float";
const char *INDEX_PATH_PREFIX_DESCRIPTION = "Path prefix to the index, e.g. '/mnt/data/my_ann_index'";
const char *RESULT_PATH_DESCRIPTION =
    "Path prefix for saving results of the queries, e.g. '/mnt/data/query_file_X.bin'";
const char *QUERY_FILE_DESCRIPTION = "Query file in binary format, e.g. '/mnt/data/query_file_X.bin'";
const char *NUMBER_OF_RESULTS_DESCRIPTION = "Number of neighbors to be returned";
const char *SEARCH_LIST_DESCRIPTION =
    "List of L values of search.  L value is the number of neighbor/distance pairs to keep in-memory at the same time "
    "during a query.  The higher the value the more accurate the results and the slower the search.  This value MUST "
    "be at least K (number of neighbors to return).";
const char *INPUT_DATA_PATH = "Input data file in bin format.  This is the file you want to build the index over.";

// Optional parameters
const char *FILTER_LABEL_DESCRIPTION =
    "Filter to use when running a query.  'filter_label' and 'query_filters_file' are mutually exclusive.";
const char *FILTERS_FILE_DESCRIPTION =
    "Filter file for Queries for Filtered Search.  File format is text with one filter per line.  File must "
    "have exactly one filter OR the same number of filters as there are queries in the 'query_file'.";
const char *LABEL_TYPE_DESCRIPTION =
    "Storage type of Labels <uint/ushort>, default value is uint which will consume memory 4 bytes per filter";
const char *GROUND_TRUTH_FILE_DESCRIPTION =
    "ground truth file for the queryset"; // what's the format, what's the requirements? does it need to include an
                                          // entry for every item or just a small subset? I have so many questions about
                                          // this file
const char *NUMBER_THREADS_DESCRIPTION = "Number of threads used for building index.  Defaults to number of logical "
                                         "processor cores on your this machine returned by omp_get_num_procs()";
const char *FAIl_IF_RECALL_BELOW = "If set to a value >0 and <100%, program returns -1 if best recall found is below "
                                   "this threshold. "; // does it continue running or die immediately?  Will I still get
                                                       // my results even if the return code is -1?
const char *NUMBER_OF_NODES_TO_CACHE = "Number of BFS nodes around medoid(s) to cache.  Default value: 0";
const char *BEAMWIDTH = "Beamwidth for search. Set 0 to optimize internally.  Default value: 2";
const char *MAX_BUILD_DEGREE = "Maximum graph degree";
const char *GRAPH_BUILD_COMPLEXITY = "Build complexity, higher value results in better graphs";
const char *GRAPH_BUILD_ALPHA = "Alpha controls density and diameter of graph, set 1 for sparse graph, 1.2 or 1.4 for "
                                "denser graphs with lower diameter";
const char *BUIlD_GRAPH_PQ_BYTES = "Number of PQ bytes to build the index; 0 for full precision build";
const char *USE_OPQ = "Use Optimized Product Quantization (OPQ).";
const char *LABEL_FILE = "Input label file in txt format for Filtered Index build. The file should contain comma "
                         "separated filters for each node with each line corresponding to a graph node";
const char *UNIVERSAL_LABEL =
    "Universal label, Use only in conjunction with label file for filtered index build. If a "
    "graph node has all the labels against it, we can assign a special universal filter to the "
    "point instead of comma separated filters for that point";
const char *FILTERED_LBUILD = "Build complexity for filtered points, higher value results in better graphs";

} // namespace program_options_utils
