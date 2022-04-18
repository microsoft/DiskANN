# DiskANN

The goal of the project is to build scalable, performant and cost-effective approximate nearest neighbor search algorithms.
This release has the code from the [DiskANN paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf) published in NeurIPS 2019, and improvements. 
This code reuses and builds upon some of the [code for NSG](https://github.com/ZJULearning/nsg) algorithm.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.



## Linux build:

Install the following packages through apt-get

```bash
sudo apt install cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-dev 
```

For building the REST API, also install [cpprestsdk](https://github.com/microsoft/cpprestsdk)
```bash
sudo apt-get install libcpprest-dev
```

### Install Intel MKL
#### Ubuntu 20.04
```bash
sudo apt install libmkl-full-dev
```

#### Earlier versions of Ubuntu
Install Intel MKL either by downloading the [oneAPI MKL installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) or using [apt](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) (we tested with build 2019.4-070 and 2022.1.2.146).

```
# OneAPI MKL Installer
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146.sh
sudo sh l_BaseKit_p_2022.1.2.146.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
```

### Build
```bash
mkdir build && cd build && cmake .. && make -j 
```

## Windows build:

The Windows version has been tested with the Enterprise editions of Visual Studio 2017 and Visual Studio 2019. It should work with the Community and Professional editions as well without any changes. 

**Prerequisites:**

* Install CMAKE (v3.15.2 or later) from https://cmake.org
* Install MKL from https://software.intel.com/en-us/mkl
* Install/download Boost from https://www.boost.org

* Environment variables: 
    * Set a new System environment variable, called INTEL_ROOT to the "windows" folder under your MKL installation
	   (For instance, if your install folder is "C:\Program Files (x86)\IntelSWtools", set INTEL_ROOT to "C:\Program Files (x86)\IntelSWtools\compilers_and_libraries\windows")
    * Set environment variable BOOST_ROOT to your boost folder.

**Build steps:**
-	Open a new command prompt window
-	Create a "build" directory under diskann
-	Change to the "build" directory and run  
```
<full-path-to-cmake>\cmake -G "Visual Studio 16 2019" -B. -A x64 ..
```
OR 
```
<full-path-to-cmake>\cmake -G "Visual Studio 15 2017" -B. -A x64 ..
```

**Note: Since VS comes with its own (older) version of cmake, you have to specify the full path to cmake to ensure that the right version is used.**
-	This will create a “diskann” solution file in the "build" directory
-	Open the "diskann" solution and build the “diskann” project. 
- 	Then build all the other binaries using the ALL_BUILD project that is part of the solution
- 	Generated binaries are stored in the diskann/x64/Debug or diskann/x64/Release directories.

To build from command line, change to the "build" directory and use msbuild to first build the "diskpriority_io" and "diskann_dll" projects. And then build the entire solution, as shown below.
```
msbuild src\dll\diskann.vcxproj
msbuild diskann.sln
```
Check msbuild docs for additional options including choosing between debug and release builds.


## Usage:

Please see the following pages on using the compiled code:

- [Commandline interface for building and search SSD based indices](workflows/SSD_index.md)  
- [Commandline interface for building and search in memory indices](workflows/in_memory_index.md) 
- [REST service set up for serving DiskANN indices and query interface](workflows/rest_api.md)
