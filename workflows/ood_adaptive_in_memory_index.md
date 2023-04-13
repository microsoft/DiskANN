**Usage for ood-adaptive in-memory indices**
================================

To generate index, you will need an initial graph built over the base points using Vamana algorithm without using any query information. 
Use the `tests/search_memory_index` program to search for approximate nearest neighbors for training query points using the initial graph.
Use the training query points and their approximate nearest neighbor ids to build the query-adaptive graph. 
--------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--query_file**: The training query data used to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices. The number of query points used in indexing is limited by num_base_points / 100.
5. **--nnid_path**: The approximate nearest neighbor ids of the training query points. The first 4 bytes represent number of points as integer. The next 4 bytes represent the number of nearest neighbors per query. The following `n*d*4` bytes contain the d approximate nearest neighbor IDs per query point.
6. **--index_path_prefix**: The constructed index components will be saved to this path prefix.
7. **-R (--max_degree)** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality. 
8. **-L (--Lbuild)** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. 
9. **--alpha** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs.
10. **--lambda** (default is 0.75): A float value between 0 and 1 which decides to what extent indexing is adaptive to query distribution. 0 will yield baseline vamana graph disregarding any query information. 
11. **T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
12. **--build_PQ_bytes** (default is 0): Set to a positive value less than the dimensionality of the data to enable faster index build with PQ based distance comparisons. Defaults to using full precision vectors for distance comparisons.
13. **--use_opq**: use the flag to use OPQ rather than PQ compression. OPQ is more space efficient for some high dimensional datasets, but also needs a bit more build time.


To search the generated index, use the `tests/search_memory_index` program:
---------------------------------------------------------------------------


The arguments are as follows:

1. **data_type**: The type of dataset you built the index on. float(32 bit), signed int8 and unsigned uint8 are supported. Use the same data type as in arg (1) above used in building the index.
2. **dist_fn**: There are two distance functions supported: l2 and mips. There is an additional *fast_l2* implementation that could provide faster results for small (about a million-sized) indices. Use the same distance as in arg (2) above used in building the index.
3. **memory_index_path**: index built above in argument (4).
4. **T**: The number of threads used for searching. Threads run in parallel and one thread handles one query at a time. More threads will result in higher aggregate query throughput, but may lead to higher per-query latency, especially if the DRAM bandwidth is a bottleneck. So find the balance depending on throughput and latency required for your application.
5. **query_bin**: The queries to be searched on in same binary file format as the data file (ii) above. The query file must be the same type as in argument (1).
6. **truthset.bin**: The ground truth file for the queries in arg (7) and data file used in index construction.  The binary file must start with *n*, the number of queries (4 bytes), followed by *d*, the number of ground truth elements per query (4 bytes), followed by `n*d` entries per query representing the d closest IDs per query in integer format,  followed by `n*d` entries representing the corresponding distances (float). Total file size is `8 + 4*n*d + 4*n*d` bytes. The groundtruth file, if not available, can be calculated using the program `tests/utils/compute_groundtruth`. Use "null" if you do not have this file and if you do not want to compute recall.
7. **K**: search for *K* neighbors and measure *K*-recall@*K*, meaning the intersection between the retrieved top-*K* nearest neighbors and ground truth *K* nearest neighbors.
8. **result_output_prefix**: search results will be stored in files, one per L value (see next arg), with specified prefix, in binary format.
9. **-L (--search_list)**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be atleast the value of *K* in (7).


Example with BIGANN:
--------------------

This example demonstrates the use of the commands above on a 100K slice of the [BIGANN dataset](http://corpus-texmex.irisa.fr/) with 128 dimensional SIFT descriptors applied to images. 

Download the base and query set and convert the data to binary format
```bash
mkdir -p DiskANN/build/data && cd DiskANN/build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
./tests/utils/fvecs_to_bin float data/sift/sift_learn.fvecs data/sift/sift_learn.fbin
./tests/utils/fvecs_to_bin float data/sift/sift_query.fvecs data/sift/sift_query.fbin
```

Now build and search the index and measure the recall using ground truth computed using bruteforce. 
```bash
./tests/build_memory_index  --data_type float --dist_fn l2 --data_path data/sift/sift_base.fbin --index_path_prefix data/sift/index_sift_base_R32_L50_A1.2 -R 32 -L 50 --alpha 1.2
 ./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/index_sift_base_R32_L50_A1.2 --query_file data/sift/sift_learn.fbin -K 10 -L 100 --result_path data/sift/train_id
./tests/build_memory_index  --data_type float --dist_fn l2 --data_path data/sift/sift_base.fbin --query_path data/sift/sift_learn.fbin --nnid_path data/sift/train_id_100_idx_uint32.bin --index_path_prefix data/sift/index_sift_base_R32_L50_A1.2_lamb0.75 -R 32 -L 50 --alpha 1.2 --lambda 0.75
 ./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/index_sift_base_R32_L50_A1.2_lamb0.75 --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_base_gt100 -K 10 -L 10 20 30 40 50 100 --result_path data/sift/res
 ```
 

 The output of search lists the throughput (Queries/sec) as well as mean and 99.9 latency in microseconds for each `L` parameter provided. (We measured on a 64-core Intel(R) Xeon(R) Gold 5218 CPU)
 ```
 Baseline Vamana results: 
  Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
=================================================================================
  10    53973.68            440.20             1161.15       19354.94       75.29
  20    35658.90            633.58             1783.73        4000.10       86.69
  30    27310.13            812.48             2330.14        4957.13       91.43
  40    24590.82            986.31             2586.45       11013.70       94.04
  50    22236.00           1155.30             2806.70       18312.87       95.54
 100    15070.39           1947.84             4221.98       20893.03       98.48

 Query-adaptive Vamana results:
   Ls         QPS     Avg dist cmps  Mean Latency (mus)   99.9 Latency   Recall@10
=================================================================================
  10    56853.06            403.46             1097.40       15754.79       88.37
  20    38910.86            585.14             1635.11        3080.47       94.77
  30    29094.23            760.58             2186.95        3865.86       96.73
  40    25031.22            931.91             2542.23       11071.94       97.66
  50    23508.23           1097.78             2706.31        4294.26       98.21
 100    15400.35           1874.25             4135.48       17101.10       99.13
 ```


