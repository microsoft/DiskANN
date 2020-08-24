# DiskANN

The goal of the project is to build scalable, performant and cost-effective approximate nearest neighbor search algorithms.
The initial release has the in-memory version of the [DiskANN paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf) published in NeurIPS 2019. The SSD based index will be released later.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.



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

* Environment variables: 
    * Set a new System environment variable, called INTEL_ROOT to the "windows" folder under your MKL installation
	   (For instance, if your install folder is "C:\Program Files (x86)\IntelSWtools", set INTEL_ROOT to "C:\Program Files (x86)\IntelSWtools\compilers_and_libraries\windows")

**Build steps:**
-	Open a new developer command prompt
-	Create a "build" directory under diskann
-	Change to the "build" directory and run  
```
<full-path-to-cmake>\cmake -B. -A x64 ..
```
**Note: Since VS comes with its own (older) version of cmake, you have to specify the full path to cmake to ensure that the right version is used.**
-	This will create a “diskann” solution file in the "build" directory
-	Open the "diskann" solution and build the “nsg_dll” project. 
- 	Then build all the other binaries using the ALL_BUILD project that is part of the solution
- 	Generated binaries are stored in the diskann/x64/Debug or diskann/x64/Release directories.

To build from command line, change to the "build" directory and use msbuild to first build the "diskpriority_io" and "nsg_dll" projects. And then build the entire solution, as shown below.
```
msbuild src\dll\nsg_dll.vcxproj
msbuild diskann.sln
```
Check msbuild docs for additional options including choosing between debug and release builds.


##Usage:


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
