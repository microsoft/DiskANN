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



