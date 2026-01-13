# DiskANN

[![DiskANN Main](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml/badge.svg?branch=main)](https://github.com/microsoft/DiskANN/actions/workflows/push-test.yml)
[![PyPI version](https://img.shields.io/pypi/v/diskannpy.svg)](https://pypi.org/project/diskannpy/)
[![Downloads shield](https://pepy.tech/badge/diskannpy)](https://pepy.tech/project/diskannpy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![DiskANN Paper](https://img.shields.io/badge/Paper-NeurIPS%3A_DiskANN-blue)](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Arxiv%3A_Fresh--DiskANN-blue)](https://arxiv.org/abs/2105.09613)
[![DiskANN Paper](https://img.shields.io/badge/Paper-Filtered--DiskANN-blue)](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf)


DiskANN is a suite of scalable, accurate and cost-effective approximate nearest neighbor search algorithms for large-scale vector search that support real-time changes and simple filters.
This code is based on ideas from the [DiskANN](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf), [Fresh-DiskANN](https://arxiv.org/abs/2105.09613) and the [Filtered-DiskANN](https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf) papers with further improvements. 
This code forked off from [code for NSG](https://github.com/ZJULearning/nsg) algorithm.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.

## Linux build:

Install the following packages through apt-get

```bash
sudo apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```

### Install Intel MKL
#### Ubuntu 20.04 or newer
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
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j 
```

### AVX-512 BF16 (optional acceleration)

DiskANN includes an optional AVX-512 BF16-accelerated kernel for `bf16` distance computations.

- Compile-time: the AVX-512 BF16 kernel is enabled only when the compiler supports the required flags; it is compiled for a single source file (`src/bf16_simd_kernels.cpp`) so the rest of the project is not forced to use AVX-512.
- Runtime: `bf16` distance code automatically dispatches to the AVX-512 BF16 kernel only when the running CPU/OS supports AVX-512 BF16; otherwise it falls back to the scalar implementation.

You can control this with the following CMake options (non-MSVC builds):

- Default (try to enable when supported):
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDISKANN_AVX512BF16=ON
    cmake --build build -j
    ```
- Force disable:
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDISKANN_AVX512BF16=OFF
    cmake --build build -j
    ```
- Force enable (fail configure if compiler does not support AVX-512 BF16 flags):
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDISKANN_FORCE_AVX512BF16=ON
    cmake --build build -j
    ```

### AMX BF16 (optional acceleration)

DiskANN includes an optional AMX BF16-accelerated kernel for `bf16` inner-product computations.

- Compile-time: the AMX BF16 kernel is enabled only when the compiler supports the required flags; it is compiled for a single source file (`src/bf16_amx_kernels.cpp`) so the rest of the project is not forced to use AMX.
- Runtime: `bf16` distance code automatically dispatches to the AMX kernel only when the running CPU/OS supports AMX and the current thread is permitted to use AMX tile state (Linux `arch_prctl` request). If unavailable, it falls back to AVX-512 BF16 (if enabled) and then scalar.

You can control this with the following CMake options (non-MSVC builds):

- Default (try to enable when supported):
    ```bash
    cmake -S . -B build-amx -DCMAKE_BUILD_TYPE=Release -DDISKANN_AMXBF16=ON
    cmake --build build-amx -j
    ```
- Force disable:
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDISKANN_AMXBF16=OFF
    cmake --build build -j
    ```
- Force enable (fail configure if compiler does not support AMX flags):
    ```bash
    cmake -S . -B build-amx -DCMAKE_BUILD_TYPE=Release -DDISKANN_FORCE_AMXBF16=ON
    cmake --build build-amx -j
    ```

### AVX-512 vs AMX (build one or the other)

If you want to do a strict A/B build where only one ISA path is compiled/used, configure two separate build directories.

- AVX-512 BF16 only (no AMX codegen):
        ```bash
        cmake -S . -B build-avx512 -DCMAKE_BUILD_TYPE=Release \
            -DDISKANN_FORCE_AVX512BF16=ON \
            -DDISKANN_AMXBF16=OFF
        cmake --build build-avx512 -j
        ```

- AMX BF16 only (no AVX-512 BF16 code path):
        ```bash
        cmake -S . -B build-amx -DCMAKE_BUILD_TYPE=Release \
            -DDISKANN_FORCE_AMXBF16=ON \
            -DDISKANN_AVX512BF16=OFF
        cmake --build build-amx -j
        ```

Note: some toolchains/build scripts add global `-march=native`. When AMX is disabled (`-DDISKANN_AMXBF16=OFF`), DiskANN explicitly compiles the AMX translation unit with `-mno-amx-tile`/`-mno-amx-bf16` (when supported) to avoid accidentally emitting AMX instructions.

### RaBitQ main-search approximate scoring (optional, runtime-gated)

DiskANN also supports using RaBitQ multi-bit codes as the *main traversal approximate scorer* in SSD search (inside `PQFlashIndex::cached_beam_search`).

- Default behavior is unchanged: traversal uses the existing PQ distance lookup.
- When enabled, traversal scoring uses RaBitQ approximate inner product (converted to a “distance” as `-ip`) while keeping the rest of the search logic intact.

#### Runtime enable

Set:

```bash
export DISKANN_USE_RABITQ_MAIN_APPROX=1
```

If the environment variable is set but RaBitQ main codes are missing or incompatible, DiskANN prints a one-time message and automatically falls back to PQ.

#### Main code file naming

Preferred sidecar file name:

```text
<index_filepath>_rabitq_main.bin
```

For example, if your SSD index file is `foo_disk.index`, the RaBitQ main code file should be `foo_disk.index_rabitq_main.bin`.

#### Generating main codes during disk index build

You can generate the main-search sidecar automatically as part of disk index build:

```bash
./build/apps/build_disk_index \
    ... \
    --dist_fn mips \
    --build_rabitq_main_codes \
    --rabitq_nb_bits 4
```

This produces:

```text
<index_path_prefix>_disk.index_rabitq_main.bin
```

#### Constraints

- Currently supported only for `dist_fn=mips` / `Metric::INNER_PRODUCT`.
- The RaBitQ code `dim` must match the index `_data_dim` (post any preprocessing/augmentation), otherwise main-search RaBitQ is disabled and the search falls back to PQ.
- Ensure you run the updated `search_disk_index`/`build_disk_index` binaries from the same build directory that contains this feature.

## Windows build:

The Windows version has been tested with Enterprise editions of Visual Studio 2022, 2019 and 2017. It should work with the Community and Professional editions as well without any changes. 

**Prerequisites:**

* CMake 3.15+ (available in VisualStudio 2019+ or from https://cmake.org)
* NuGet.exe (install from https://www.nuget.org/downloads)
    * The build script will use NuGet to get MKL, OpenMP and Boost packages.
* DiskANN git repository checked out together with submodules. To check out submodules after git clone:
```
git submodule init
git submodule update
```

* Environment variables: 
    * [optional] If you would like to override the Boost library listed in windows/packages.config.in, set BOOST_ROOT to your Boost folder.

**Build steps:**
* Open the "x64 Native Tools Command Prompt for VS 2019" (or corresponding version) and change to DiskANN folder
* Create a "build" directory inside it
* Change to the "build" directory and run
```
cmake ..
```
OR for Visual Studio 2017 and earlier:
```
<full-path-to-installed-cmake>\cmake ..
```
**This will create a diskann.sln solution**. Now you can:

- Open it from VisualStudio and build either Release or Debug configuration.
- `<full-path-to-installed-cmake>\cmake --build build`
- Use MSBuild:
```
msbuild.exe diskann.sln /m /nologo /t:Build /p:Configuration="Release" /property:Platform="x64"
```

* This will also build gperftools submodule for libtcmalloc_minimal dependency.
* Generated binaries are stored in the x64/Release or x64/Debug directories.

## Usage:

Please see the following pages on using the compiled code:

- [Commandline interface for building and search SSD based indices](workflows/SSD_index.md)  
- [Commandline interface for building and search in memory indices](workflows/in_memory_index.md) 
- [Commandline examples for using in-memory streaming indices](workflows/dynamic_index.md)
- [Commandline interface for building and search in memory indices with label data and filters](workflows/filtered_in_memory.md)
- [Commandline interface for building and search SSD based indices with label data and filters](workflows/filtered_ssd_index.md)
- [diskannpy - DiskANN as a python extension module](python/README.md)

Please cite this software in your work as:

```
@misc{diskann-github,
   author = {Simhadri, Harsha Vardhan and Krishnaswamy, Ravishankar and Srinivasa, Gopal and Subramanya, Suhas Jayaram and Antonijevic, Andrija and Pryce, Dax and Kaczynski, David and Williams, Shane and Gollapudi, Siddarth and Sivashankar, Varun and Karia, Neel and Singh, Aditi and Jaiswal, Shikhar and Mahapatro, Neelam and Adams, Philip and Tower, Bryan and Patel, Yash}},
   title = {{DiskANN: Graph-structured Indices for Scalable, Fast, Fresh and Filtered Approximate Nearest Neighbor Search}},
   url = {https://github.com/Microsoft/DiskANN},
   version = {0.6.1},
   year = {2023}
}
```
