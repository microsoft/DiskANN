**Usage for filtered indices**
================================

To generate an SSD-friendly index, use the `tests/build_disk_index` program. 
----------------------------------------------------------------------------

## Building a SSD based filtered Index

### filtered-vamana SSD Index

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as an integer. The next 4 bytes represent the dimension of data as an integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. `sizeof(T)` is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--index_path_prefix**: the index will span a few files, all beginning with the specified prefix path. For example, if you provide `~/index_test` as the prefix path, build  generates files such as `~/index_test_pq_pivots.bin, ~/index_test_pq_compressed.bin, ~/index_test_disk.index, ...`. There may be between 8 and 10 files generated with this prefix depending on how the index is constructed.
5. **-R (--max_degree)**  (default is 64): the degree of the graph index, typically between 60 and 150. Larger R will result in larger indices and longer indexing times, but better search quality. 
6. **-L (--Lbuild)**  (default is 100): the size of search listduring index build. Typical values are between 75 to 200. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Use a value for L value that is at least the value of R unless you need to build indices really quickly and can somewhat compromise on quality. Note that this is to be used only for building an unfiltered index. The corresponding search list parameter for a filtered index is managed by `--FilteredLbuild`.
7. **-B (--search_DRAM_budget)**: bound on the memory footprint of the index at search time in GB. Once built, the index will use up only the specified RAM limit, the rest will reside on disk. This will dictate how aggressively we compress the data vectors to store in memory. Larger will yield better performance at search time. For an n point index, to use b byte PQ compressed representation in memory, use `B = ((n * b) / 2^30  + (250000*(4*R + sizeof(T)*ndim)) / 2^30)`. The second term in the summation is to allow some buffer for caching about 250,000 nodes from the graph in memory while serving.  If you are not sure about this term, add 0.25GB to the first term. 
8. **-M (--build_DRAM_budget)**: Limit on the memory allowed for building the index in GB. If you specify a value less than what is required to build the index in one pass, the index is  built using a divide and conquer approach so that  sub-graphs will fit in the RAM budget. The sub-graphs are overlayed to build the overall index. This approach can be upto 1.5 times slower than building the index in one shot. Allocate as much memory as your RAM allows.
9. **-T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
10. **--PQ_disk_bytes**  (default is 0): Use 0 to store uncompressed data on SSD. This allows the index to asymptote to 100% recall. If your vectors are too large to store in SSD, this parameter provides the option to compress the vectors using PQ for storing on SSD. This will trade off recall. You would also want this to be greater than the number of bytes used for the PQ compressed data stored in-memory
11. **--build_PQ_bytes** (default is 0): Set to a positive value less than the dimensionality of the data to enable faster index build with PQ based distance comparisons. 
12. **--use_opq**: use the flag to use OPQ rather than PQ compression. OPQ is more space efficient for some high dimensional datasets, but also needs a bit more build time.
13. **--label_file**: Filter data for each point, in `.txt` format. Line `i` of the file consists of a comma-separated list of filters corresponding to point `i` in the file passed via `--data_file`.
14. **--universal_label**: Optionally, the the filter data may contain a "wild-card" filter corresponding to all filters. This is referred to as a universal label. Note that if a point has the universal label, then the filter data must only have the universal label on the line corresponding to said point.
15. **--FilteredLbuild**: If building a filtered index, we maintain a separate search list from the one provided by `--Lbuild`. 
16. **--filter_threshold**: Threshold to break up the existing nodes to generate new graph internally by breaking dense points where each node will have a maximum F labels. Default value is zero where no break up happens for the dense points.


## Computing a groundtruth file for a filtered index
In order to evaluate the performance of our algorithms, we can compare its results (i.e. the top `k` neighbors found for each query) against the results found by an exact nearest neighbor search. We provide the program `tests/utils/compute_groundtruth.cpp` to provide the results for the latter:

1. **`--data_type`** The type of dataset you built an index with. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **`--dist_fn`**: There are two distance functions supported: l2 and mips.
3. **`--base_file`**: The input data over which to build an index, in .bin format. Corresponds to the `--data_path` argument from above.
4. **`--query_file`**: The queries to be searched on, which are stored in the same .bin format.
5. **`--label_file`**: Filter data for each point, in `.txt` format. Line `i` of the file consists of a comma-separated list of filters corresponding to point `i` in the file passed via `--data_file`. 
6. **`--filter_label`**: Filter for each query. For each query, a search is performed with this filter.
7. **`--universal_label`**: Corresponds to the universal label passed when building an index with filter support.
8. **`--gt_file`**: File to output results to. The binary file starts with `n`, the number of queries (4 bytes), followed by `d`, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the `d` closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes.
9. **`-K`**: The number of nearest neighbors to compute for each query. 

## Searching a Filtered Index

Searching a filtered index uses the `tests/search_memory_index.cpp`:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. Use the same data type as in arg (1) above used in building the index.
2.  **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips). Use the same distance as in arg (2) above used in building the index.
3. **--index_path_prefix**: same as the prefix used in building the index (see arg 4 above).
4. **--num_nodes_to_cache** (default is 0): While serving the index, the entire graph is stored on SSD. For faster search performance, you can cache a few frequently accessed nodes in memory. 
5. **-T (--num_threads)** (default is to get_omp_num_procs()): The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but will also use more IOs/second across the system, which may lead to higher per-query latency. So find the balance depending on the maximum number of IOPs supported by the SSD.
6. **-W (--beamwidth)** (default is 2): The beamwidth to be used for search. This is the maximum number of IO requests each query will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query, but might result in slightly higher total number of IO requests to SSD per query. For the highest query throughput with a fixed SSD IOps rating, use `W=1`. For best latency, use `W=4,8` or higher complexity search. Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will involve some tuning overhead. 
7. **--query_file**: The queries to be searched on in same binary file format as the data file in arg (2) above. The query file must be the same type as argument (1).
8. **--gt_file**: The ground truth file for the queries in arg (7) and data file used in index construction.  The binary file must start with *n*, the number of queries (4 bytes), followed by *d*, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the d closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes. The groundtruth file, if not available, can be calculated using the program `tests/utils/compute_groundtruth`. Use "null" if you do not have this file and if you do not want to compute recall.
9. **-K**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.
10. **--result_path**: Search results will be stored in files with specified prefix, in bin format.
11. **-L (--search_list)**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be atleast the value of *K* in arg (9).
12. **--filter_label**: The filter to be used when searching an index with filters. For each query, a search is performed with this filter.