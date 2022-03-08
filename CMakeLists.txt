# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

if(MSVC)
	cmake_minimum_required(VERSION 3.15)
else()
	cmake_minimum_required(VERSION 2.8)
endif()
set(CMAKE_STANDARD 14)


#set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/CompilerOptions.cmake")
if(MSVC)
	set(CMAKE_CXX_COMPILER $ENV{VCToolsInstallDir}/bin/Hostx64/x64/cl.exe)
	set(CMAKE_CXX_LINK_EXECUTABLE $ENV{VCToolsInstallDir}/bin/Hostx64/x64/link.exe)
else()
	set(CMAKE_CXX_COMPILER g++)
endif()

project(diskann)
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/dll)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()


function(checkEnvAndSetLocalVar env_var msg local_var)
	if (NOT EXISTS "$ENV{${env_var}}" )
		message (FATAL_ERROR ${msg})
	else()
		if ($ENV{${env_var}} MATCHES "\\$"  OR $ENV{${env_var}} MATCHES "/$" )
			set(${local_var} $ENV{${env_var}} PARENT_SCOPE)
		else()
			message(STATUS "Appending trailing backslash to ${env_var}")
			set(${local_var} "$ENV{${env_var}}\\" PARENT_SCOPE)
		endif()
	endif()
endfunction()

	
include_directories(include)
#MKL Config
if (MSVC)
	checkEnvAndSetLocalVar("INTEL_ROOT" "Please install Intel MKL libraries and set the env variable INTEL_ROOT to the intel software directory. Should be similar to: C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\. " "INTEL_ROOT")
	set(MKL_ROOT ${INTEL_ROOT}/mkl)
	add_compile_options(/arch:AVX2 /Qpar)
	link_libraries("${INTEL_ROOT}/mkl/lib/intel64/mkl_core_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_thread_dll.lib" "${INTEL_ROOT}/compiler/lib/intel64/libiomp5md.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_ilp64_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_sequential_dll.lib")
	checkEnvAndSetLocalVar("BOOST_ROOT" "Please install Boost (1.71 or greater) from www.boost.org and set the env var BOOST_ROOT to the boost directory." "BOOST_ROOT")
	include_directories(${INTEL_ROOT}/include)
	include_directories(${MKL_ROOT}/include)
else()
	# expected path for manual intel mkl installs
	set(OMP_PATH /opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin/)
	set(MKL_ROOT /opt/intel/oneapi/mkl/latest)
	link_directories(${OMP_PATH} ${MKL_ROOT}/lib/intel64)
	include_directories(${MKL_ROOT}/include)

	# expected path for apt packaged intel mkl installs
	link_directories(/usr/lib/x86_64-linux-gnu/mkl)
	include_directories(/usr/include/mkl)

	# compile flags and link libraries
        add_compile_options(-m64 -Wl,--no-as-needed) 
	link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
endif()
include_directories(${BOOST_ROOT})

add_definitions(-DMKL_ILP64)

#Main compiler/linker settings 
if(MSVC)
	#language options
	add_compile_options(/permissive- /openmp:experimental /Zc:wchar_t /Zc:twoPhase- /Zc:forScope /Zc:inline /WX- /std:c++14 /Gd /W3 /MP /Zi /FC /nologo /diagnostics:classic) 
	#code generation options
	add_compile_options(/Qpar /fp:fast /Zp8 /fp:except- /EHsc /GS- /Gm- /Gy )
	#optimization options
	add_compile_options(/Ot /Oy /Oi)
	#path options
	#add_compile_options(/Fdx64/Release/vc141.pdb /Fox64/Release/)
	add_definitions(-DUSE_AVX2 -DUSE_ACCELERATED_PQ -D_WINDOWS -DNOMINMAX -DUNICODE)
	
	set(CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS "/MANIFEST /MACHINE:X64 /DEBUG:FULL /LTCG:incremental /NXCOMPAT /DYNAMICBASE /OPT:REF /SUBSYSTEM:CONSOLE /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\"")
	set(CMAKE_EXECUTABLE_CXX_LINK_FLAGS "/MANIFEST /MACHINE:X64 /DEBUG:FULL /LTCG:incremental /NXCOMPAT /DYNAMICBASE /OPT:REF /SUBSYSTEM:CONSOLE /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\"")

	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG")
	set(CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS_DEBUG "${CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS_DEBUG} /DEBUG")

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
else()
	set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
    #	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -Wall -Wextra")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")
	add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ -DUSE_AVX2)
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
