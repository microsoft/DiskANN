##Linux build and usage (incomplete):

Install the following packages through apt-get, and Intel MKL either by downloading the installer or using [apt](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) (we tested with build 2019.4-070).
```
sudo apt install cmake g++ libaio-dev libgoogle-perftools-dev clang-format-4.0
```

Build
```
mkdir build && cd build && cmake .. && make -j 
```

**Usage for SSD-based indices**
To generate an SSD-friendly index, use the `tests/create_disk_index.sh` script. 
For floating point data file SIFT1M, to generate an index with 32 bytes in-
memory fooptrint per vector, you might want to use (assuming `pwd` is project root):
```
export BUILD_PATH=./build
${BUILD_PATH}/tests/utils/fvecs_to_bin data/SIFT1M/sift_base.fvecs data/SIFT1M/sift_base.bin
./tests/create_disk_index.sh -t float -i data/SIFT1M/sift_base.bin -o data/SIFT1M/tmp -L 30 -R 64 -b 32
```

To search the generated index
```
```

**Usage for in-memory indices**



##Windows CMake Build

The Windows version has been tested with the enterprise editions of Visual Studio 2017 and Visual Studio 2019

Install CMAKE (v3.15.2 or later) from https://cmake.org

Install MKL:
-	Install MKL from https://software.intel.com/en-us/mkl
-	Set a new System environment variable, called INTEL_ROOT to the "windows" folder under your MKL installation
	(For instance, if your install folder is "C:\Program Files (x86)\IntelSWtools", set INTEL_ROOT to "C:\Program Files (x86)\IntelSWtools\compilers_and_libraries\windows")

**Build steps:**
-	Open a new developer command prompt
-	Create a "build" directory under nsg
-	Change to the "build" directory and run  
```
cmake -B. -A x64 ..
```
**Note: Since VS comes with its own (older) version of cmake, you have to specify the full path to cmake to ensure that the right version is used.**
-	This will create a “diskann” solution file.
-	Open the "diskann" solution and build the “nsg_dll” project first. 
- 	Then build all the other binaries using the ALL_BUILD project that is part of the solution
- 	Generated binaries are stored in the nsg/x64/Debug or nsg/x64/Release directories.

To build from command line, use msbuild to first build the "nsg_dll" project. And then build the entire solution, as shown below.
```
msbuild src\dll\nsg_dll.vcxproj
msbuild diskann.sln
```
Check msbuild docs for additional options including choosing between debug and release builds.

###Sanity checks (paths specific to nn-z840): 

**Creating the graph**

1. Generate compressed vectors using Product Quantization
```
generate_pq.exe float "E:\sift1m_u8\sift1m_float_harsha\sift_base.bin" E:\cmake-sift\ravi-index 32 0.01
```

2. Build the graph using the compressed vectors
```
build_memory_index.exe float E:\sift1m_u8\sift1m_float_harsha\sift_base.bin 50 64 750 2 1.2 E:\cmake-sift\ravi-index_memory.index
```

3. Optimize for disk layout
```
create_disk_layout.exe float E:\sift1m_u8\sift1m_float_harsha\sift_base.bin E:\cmake-sift\ravi-index_memory.index E:\cmake-sift\ravi-index_disk.index
```

**Search**

At this stage we have a NSG graph on disk optimized for search. To conduct search for vectors given in the file sift_query.bin, 
```
search_disk_index.exe float E:\cmake-sift\ravi-index_pq_pivots.bin E:\cmake-sift\ravi-index_compressed.bin E:\cmake-sift\ravi-index_disk.index null null E:\sift1m_u8\sift1m_float_harsha\sift_query.bin 5 16 4 E:\cmake-sift\ravi-index_results 10 20 30 40
```
Results of the query are stored in the file ravi-index_resultsXX_idx_uint32.bin where XX is the L-size [10, 20, 30, 40]
To calculate recall @ 5 for L = 20, 
```
calculate_recall.exe E:\sift1m_u8\sift1m_float_harsha\sift_gs100_idx.bin E:\cmake-sift\ravi-index_results20_idx_uint32.bin 5
```

**Note:** You can simply type any of the commands given above by itself to get a description of the command line arguments.


