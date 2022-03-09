**Usage for in-memory indices**
================================

To generate index, use the `tests/build_memory_index` program. 
--------------------------------------------------------------

```
./tests/build_memory_index  data_type<int8/uint8/float>  dist_fn<l2/mips>   data_file.bin  output_index_file  R(graph degree)   L(build complexity)  alpha(graph diameter control)   T(num_threads)
```

The arguments are as follows:

(i) **data_type**: The type of dataset you wish to build an index. float(32 bit), signed int8 and unsigned uint8 are supported. 

(ii) **dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2)) and maximum inner product (mips).

(iii) **data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following n*d*sizeof(T) bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.

(iv) **output_index_file**: The constructed index will be saved here.

(v) **R**: the degree of the graph index, typically between 60 and 150. Larger R will result in larger indices and longer indexing times, but better search quality. Try to ensure that the L value is at least the R value unless you need to build indices really quickly and can somewhat compromise on quality. 

(vi) **L**: the size of search list we maintain during index building. Typical values are between 75 to 200. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity.

(vii) **alpha**: A float value between 1.0 and 1.5 which approximately determines the diameter of the graph, which will be *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield sparsest graph, 1.5 will yield denser graphs. Use 1.2 if you are not sure.

(viii) **T**: number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine).


To search the generated index, use the `tests/search_memory_index` program:
---------------------------------------------------------------------------

```
./tests/search_memory_index  index_type<float/int8/uint8>   dist_fn<l2/mips/fast_l2>   data_file.bin   memory_index_path   T(num_threads)   query_file.bin   truthset.bin(\"null\" for none)   K   result_output_prefix   L1   L2 ... 
```

The arguments are as follows:

(i) **data_type**: The type of dataset you wish to build an index. float(32 bit), signed int8 and unsigned uint8 are supported. 

(ii) **dist_fn**: There are two primary metric types of distance supported: l2 and mips. There is an additional *fast_l2* implementation that could provide faster results for small (about a million-sized) indices.

(iii) **memory_index_path**: index built above in argument (iv).

(iv) **T**: The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but may lead to higher per-query latency, especially if the DRAM bandwidth is a bottleneck. So find the balance depending on throughput and latency required for your application.

(v) **query_bin**: The queries to be searched on in same binary file format as the data file (ii) above. The query file must be the same type as in argument (i).

(vi)  **truthset.bin**: The ground truth file for the queries in arg (vii) and data file used in index construction.  The binary file must start with *n*, the number of queries (4 bytes), followed by *d*, the number of ground truth elements per query (4 bytes), followed by n*d entries per query representing the d closest IDs per query in integer format,  followed by n*d entries representing the corresponding distances (float). Total file size is 8 + 4*n*d + 4*n*d. The groundtruth file, if not available, can be calculated using the program, tests/utils/compute_groundtruth. Use "null" if you do not have this file and if you do not want to compute recall.

(vii) **K**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.

(viii) **result_output_prefix**: search results will be stored in files, one per L value (see next arg), with specified prefix, in binary format.

(ix, x, ...) various search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be atleast the value of *K* in (vii).
