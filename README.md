Build:

Windows:

Install MKL:
-	Install MKL from https://software.intel.com/en-us/mkl
-	After installation, run the 'set' comment to check if the ICPP_COMPILER19 is set. 
- 	If the variable is not set, set it manually to the "windows" folder under your MKL installation directory
	(For instance, if your install folder is "C:\Program Files (x86)\IntelSWtools", set ICPP_COMPILER19 to "C:\Program Files (x86)\IntelSWtools\compilers_and_libraries\windows"

Install CMAKE (v3.15.2 or later)

Build steps:
-	Open developer command prompt
-	Create build directory under nsg
-	Change directory to build and run  "<path to your cmake>" -B. -A x64 -G "Visual Studio 15 2017" ..
		(Do specify the full path to cmake, as VS comes with its own (older) version of cmake, which will not work)
-	This will create a “rand-nsg” solution
-	Open the rand-nsg solution and build the “nsg_lib”, “build_disk_index” and “search_disk_index” projects
-	To build from command line, use "msbuild rand-nsg.sln". Check msbuild for options around targets.

Sanity checks (paths specific to nn-z840): 

build_disk_index.exe float "E:\sift1m_u8\sift1m_float_harsha\sift_base.bin" E:\cmake-sift\disk-index_L50_R64_C750 50 64 750 32 50000

search_disk_index float E:\cmake-sift\disk-index_L50_R64_C750 E:\sift1m_u8\sift1m_float_harsha\sift_query.bin E:\sift1m_u8\sift1m_float_harsha\sift_query_gs100.bin E:\sift1m_u8\sift1m_float_harsha\sift_query_gs100_dist.bin 5


UNIX (NOT COMPLETE):
-	Install the following packages
o	cmake
o	g++ 
o	Intel MKL:  https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
o	libaio-dev
o	libgoogle-perftools-dev
o	clang-format-4.0
