**Usage for dynamic indices**
================================

A "dynamic" index refers to an index which supports insertion of new points into a previously built index as well as deletions of points in an index. The program found in `tests/test_streaming_scenario` tests this functionality. It allows the user to specify which points from the data file will be used
to initially build the index, which points will be deleted from the index, and which points will be inserted into the index. Insertions and deletions can be performed sequentially or concurrently.

When modifying the index sequentially, the user has the ability to take *snapshots*--that is, save the index to memory for every *m* insertions or deletions instead of only at the end of the build.

--------------------------------------------------------------

The arguments are as follows:

1. **--data_type**: The type of dataset you wish to build an index on. float(32 bit), signed int8 and unsigned uint8 are supported. 
2. **--dist_fn**: There are two distance functions supported: minimum Euclidean distance (l2) and maximum inner product (mips).
3. **--data_file**: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following `n*d*sizeof(T)` bytes contain the contents of the data one data point in time. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by the program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.
4. **--index_path_prefix**: The constructed index components will be saved to this path prefix.
5. **-R (--max_degree)** (default is 64): the degree of the graph index, typically between 32 and 150. Larger R will result in larger indices and longer indexing times, but might yield better search quality. 
6. **-L (--Lbuild)** (default is 100): the size of search list we maintain during index building. Typical values are between 75 to 400. Larger values will take more time to build but result in indices that provide higher recall for the same search complexity. Ensure that value of L is at least that of R value unless you need to build indices really quickly and can somewhat compromise on quality. 
7. **--alpha** (default is 1.2): A float value between 1.0 and 1.5 which determines the diameter of the graph, which will be approximately *log n* to the base alpha. Typical values are between 1 to 1.5. 1 will yield the sparsest graph, 1.5 will yield denser graphs. 
8. **T (--num_threads)** (default is to get_omp_num_procs()): number of threads used by the index build process. Since the code is highly parallel, the  indexing time improves almost linearly with the number of threads (subject to the cores available on the machine and DRAM bandwidth).
9. **--points_to_skip**: number of points to skip from the beginning of the data file. 
10. **--max_points_to_insert**: the maximum size of the index. 
11. **--beginning_index_size**: how many points to build the initial index with. The number of points inserted dynamically will be max_points_to_insert - beginning_index_size. 
12. **--points_per_checkpoint**: when inserting and deleting sequentially, each update is handled in points_per_checkpoint batches. When updating concurrently, insertions are handled in points_per_checkpoint batches but deletions are always processed in a single batch.
13. **--checkpoints_per_snapshot**: when inserting and deleting sequentially, the graph is saved to memory every checkpoints_per_snapshot checkpoints. This is not currently supported for concurrent updates.
14. **--points_to_delete_from_beginning**: how many points to delete from the index, starting in order of insertion. If deletions are concurrent with insertions, points_to_delete_from_beginning cannot be larger than beginning_index_size. 
14. **--do_concurrent** (default false): whether to perform insertions and deletions concurrently or sequentially. If concurrent is specified, half the threads are used for insertions and half the threads are used for deletions. Note that insertions are performed before deletions if this flag is set to false, so in this case is possible to delete more than beginning_index_size points.


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
9. **-L (--search_list)**: A list of search_list sizes to perform search with. Larger parameters will result in slower latencies, but higher accuracies. Must be at least the value of *K* in (7).
10. **--dynamic** (default false): whether the index being searched is dynamic or not.
11. **--tags** (default false): whether to search with tags. This should be used if point *i* in the ground truth file does not correspond the point in the *i*th position in the loaded index.


Example with BIGANN:
--------------------

This example demonstrates the use of the commands above on a 100K slice of the [BIGANN dataset](http://corpus-texmex.irisa.fr/) with 128 dimensional SIFT descriptors applied to images. 

Download the base and query set and convert the data to binary format
```bash
mkdir -p DiskANN/build/data && cd DiskANN/build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
./tests/utils/fvecs_to_bin data/sift/sift_learn.fvecs data/sift/sift_learn.fbin
./tests/utils/fvecs_to_bin data/sift/sift_query.fvecs data/sift/sift_query.fbin
```

The example below tests the following scenario: using a file with 100000 points, the first 50000 points are used to initially build the index. Then, the first 25000 points are deleted from the index, while the next 25000 points (i.e. points 50001 to 75000) are concurrently inserted into the index. Note that the memory index should be built **before** calculating the ground truth, since the memory index returns the slice of the sift100K dataset that was used to build the final graph (that is, points 25001-75000 in the original index.)
```bash
./tests/build_memory_index  --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --index_path_prefix data/sift/index_sift_learn_dynamic -R 32 -L 50 --alpha 1.2 --T 16 --points_to_skip 0 --max_points_to_insert 75000 --beginning_index_size 25000 --points_per_checkpoint 10000 --checkpoints_per_snapshot 0 --points_to_delete_from_beginning 25000 --do_concurrent true
./tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/sift/index_sift_learn_dynamic.data --query_file  data/sift/sift_query.fbin --gt_file data/sift/sift_query_learn_dynamic_gt100 --K 100
 ./tests/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/index_sift_learn_dynamic --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_learn_dynamic_gt100 -K 10 -L 10 20 30 40 50 100 --result_path data/sift/res --dynamic true
 ```




