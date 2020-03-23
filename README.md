##Linux build:

Install the following packages through apt-get, and Intel MKL either by downloading the installer or using [apt](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) (we tested with build 2019.4-070).
```
sudo apt install cmake g++ libaio-dev libgoogle-perftools-dev clang-format-4.0
```

Build
```
mkdir build && cd build && cmake .. && make -j 
```

##Windows build:

The Windows version has been tested with the enterprise editions of Visual Studio 2017 and Visual Studio 2019

**Prerequisites:**

* Install CMAKE (v3.15.2 or later) from https://cmake.org
* Install MKL from https://software.intel.com/en-us/mkl
* Download boost (v1.71.0, later versions are not tested) from boost.org 

* Environment variables: 
    * Set a new System environment variable, called INTEL_ROOT to the "windows" folder under your MKL installation
	   (For instance, if your install folder is "C:\Program Files (x86)\IntelSWtools", set INTEL_ROOT to "C:\Program Files (x86)\IntelSWtools\compilers_and_libraries\windows")
    * Set BOOST_ROOT to your boost download folder

**Build steps:**
-	Open a new developer command prompt
-	Create a "build" directory under diskann
-	Change to the "build" directory and run  
```
cmake -B. -A x64 ..
```
**Note: Since VS comes with its own (older) version of cmake, you have to specify the full path to cmake to ensure that the right version is used.**
-	This will create a “diskann” solution file.
-	Open the "diskann" solution and build the "diskpriority_io" and “nsg_dll” projects in order. 
- 	Then build all the other binaries using the ALL_BUILD project that is part of the solution
- 	Generated binaries are stored in the diskann/x64/Debug or diskann/x64/Release directories.

To build from command line, change to the "build" directory and use msbuild to first build the "diskpriority_io" and "nsg_dll" projects. And then build the entire solution, as shown below.
```
msbuild src\dll\diskpriority_io.vcxproj
msbuild src\dll\nsg_dll.vcxproj
msbuild diskann.sln
```
Check msbuild docs for additional options including choosing between debug and release builds.


##Usage:

We now detail the main binaries using which one can build and search indices which reside in memory as well as SSD-resident indices.

**Usage for SSD-based indices**
===============================

To generate an SSD-friendly index, use the `tests/build_disk_index` program. 
----------------------------------------------------------------------------

```
./tests/build_disk_index  [data_type<float/int8/uint8>]  [data_file.bin]  [index_prefix_path]  [R]  [L]  [B]  [M]  [T]. 
```

The arguments are as follows:

(i) data_type:  The datatype is the type of dataset you wish to build an index. We support byte indices (signed int8 or unsigned uint8) or float indices. 

(ii) data_file: The input data over which to build an index, in .bin format. The first 4 bytes represent number of points as integer. The next 4 bytes represent the dimension of data as integer. The following n*d*sizeof(T) bytes contain the contents of the data in row major format. sizeof(T) is 1 for byte indices, and 4 for float indices. This will be read by our program as int8_t for signed indices, uint8_t for unsigned indices or float for float indices.

(iii) index_prefix_path: the index will generate a few files, all beginning with the specified prefix path. For example, if you provide ~/index_test as the prefix path, we will generate files such as ~/index_test_pq_pivots.bin, ~/index_test_pq_compressed.bin, ~/index_test_disk.index, etc. There may be between 8 and 10 files generated with this prefix depending on how we construct the index.

(iv) R: the degree of our graph index, typically between 60 and 150. Again, larger values will result in bigger indices (with longer indexing times), but better search quality. Try to ensure that the L value is at least the R value unless you need to build indices really quickly, but can somewhat compromise on quality. 

(v) L: the size of search list we maintain during index building. Typical values are between 75 to 200. Larger values will take more time but result in better quality indices.

(vi) B: Please set the desired memory footprint of the resulting index. Once built, our index will use up only the specified RAM limit, the rest will reside on disk. This will dictate how aggressively we compress the data vectors to store in memory. Larger will yield better performance at search time.

(vii) M: indexing time memory limit. How much of the total RAM would you like to allocate to our process? If you specify less RAM, we will build our index using a divide and conquer approach so that our sub-graphs will fit in the budget at all times, and stitch together our overall index. Can be upto 1.5 times slower to use a divide-and-conquer approach. Try to provide as much RAM as possible subject to your constraints.

(viii) T: number of threads our process is allowed to use during indexing. Since our code is highly parallel, the overall indexing time improves almost linearly with the number of threads (subject to the virtual cores available on the machine).

To search the SSD-index, use the `tests/search_disk_index` program. 
----------------------------------------------------------------------------

```
./tests/search_disk_index  [index_type<float/int8/uint8>]  [index_prefix_path]  [num_nodes_to_cache]  [num_threads]  [beamwidth (use 0 to optimize internally)]  [query_file.bin]  [truthset.bin (use "null" for none)]  [K]  [result_output_prefix]  [L1]  [L2] etc.
```

The arguments are as follows:

(i) data type: same as (i) above in building index.

(ii) index_prefix_path: same as (iii) above in building index.

(iii) num_nodes_to_cache: our program stores the entire graph on disk. For faster search performance, we provide the support to cache a few nodes (which are closest to the starting point) in memory. 

(iv) num_threads: search using specified number of threads in parallel, one thread per query. More will result in more IOs, so find the balance depending on the bandwidth of the SSD.

(v) beamwidth: maximum number of IO requests each query will issue per iteration of search code. Larger beamwidth will mean fewer IO round-trips per query, but might result in more contention across threads for the disk bandwidth. Specifying 0 will optimize the beamwidth depending on the number of threads performing search.

(vi) query_file.bin: search on these queries, same format as data file (ii) above. The query file must be the same type as specified in (i).

(vii) truthset.bin file. Must be in the following format, or specify "null": n, the number of queries (4 bytes) followed by d, the number of ground truth elements per query (4 bytes), followed by n*d entries per query representing the d closest IDs per query in integer format,  followed by n*d entries representing the corresponding distances (float). Total file size is 8 + 4*n*d + 4*n*d. The groundtruth file, if not available, can be calculated using our program, tests/utils/compute_groundtruth. If you just want to measure the latency numbers of search and output the nearest neighbors without calculating recall, enter "null".

(viii) K: search for recall@k, meaning accuracy of retrieving top-k nearest neighbors.

(ix) result output prefix: will search and store the computed results in the files with specified prefix in bin format.

(x, xi, ...) various search_list sizes to perform search with. Larger will result in slower latencies, but higher accuracies. Must be atleast the recall@ value in (vi).


**Usage for in-memory indices**
================================

To generate index, use the `tests/build_memory_index` program. 
--------------------------------------------------------------

```
./tests/build_memory_index  [data_type<int8/uint8/float>]  [data_file.bin]  [output_index_file]  [R]  [L]  [alpha]  [num_threads_to_use]
```

The arguments are as follows:

(i) data_type: same as (i) above in building disk index.

(ii) data_file: same as (ii) above in building disk index, the input data file in .bin format of type int8/uint8/float.

(iii) output_index_file: memory index will be saved here.

(iv) R: max degree of index: larger is typically better, range (50-150). Preferrably ensure that L is at least R.

(v) L: candidate_list_size for building index, larger is better (typical range: 75 to 200)

(vi) alpha: float value which determines how dense our overall graph will be, and diameter will be log of n base alpha (roughly). Typical values are between 1 to 1.5. 1 will yield sparsest graph, 1.5 will yield denser graphs.

(vii) number of threads to use: indexing uses specified number of threads.


To search the generated index, use the `tests/search_memory_index` program:
---------------------------------------------------------------------------

```
./tests/search_memory_index  [index_type<float/int8/uint8>]  [data_file.bin]  [memory_index_path]  [query_file.bin]  [truthset.bin (use "null" for none)] [K]  [result_output_prefix]  [L1]  [L2] etc. 
```

The arguments are as follows:

(i) data type: same as (i) above in building index.

(ii) memory_index_path: enter path of index built (argument (iii) above in building memory index).

(iii) query_bin: search on these queries, same format as data file (ii) above. The query file must be the same type as specified in (i).

(iv) Truthset file. Must be in the following format: n, the number of queries (4 bytes) followed by d, the number of ground truth elements per query (4 bytes), followed by n*d entries per query representing the d closest IDs per query in integer format,  followed by n*d entries representing the corresponding distances (float). Total file size is 8 + 4*n*d + 4*n*d. The groundtruth file, if not available, can be calculated using our program, tests/utils/compute_groundtruth.

(v) K: search for recall@k, meaning accuracy of retrieving top-k nearest neighbors.

(vi) result output prefix: will search and store the computed results in the files with specified prefix in bin format.

(vii, viii, ...) various search_list sizes to perform search with. Larger will result in slower latencies, but higher accuracies. Must be atleast the recall@ value in (vi).


