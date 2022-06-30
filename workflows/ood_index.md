**Usage for in-memory indices**
================================

To generate index, use the `tests/build_with_query_data` program.  This program builds an index that takes a base dataset and a sample query dataset, and embeds the query dataset in the graph and uses it to perform optimizations that make searches more accurate. The returned graph object contains the query points as data points in the graph, but they are unreachable from the start node so they will never be returned by a query.
--------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--save_path**: The constructed index components will be saved to this path prefix.
5. **--query_path**: The path to the query data that will be embedded in the graph.
6. **-R (--max_degree)** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality. 
7. **-L (--Lbuild)** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. 
8. **--alpha** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs. 
9. **T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).


To search the generated index, use the `tests/search_memory_index` program as described in other workflows. Be aware that the flag for dynamic indices must be set to true.


Example with BIGANN:
--------------------

This example demonstrates the use of the commands above on a 100K slice of the [BIGANN dataset](http://corpus-texmex.irisa.fr/) with 128 dimensional SIFT descriptors applied to images. It assumes that the base and query sets are downloaded and converted to binary format, as described in other workflows. In order to see improvement in recall, this index should be queried using query data points drawn from the same distribution as the query set embedded in the index.

```bash
./tests/build_memory_index  --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --save_path data/sift/index_sift_learn_R32_L50_A1.2 --query_path data/sift/sift_query.fbin -R 32 -L 50 --alpha 1.2
 ```
 


