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

project(rand-nsg)
include_directories(${PROJECT_SOURCE_DIR}/include)


#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()


#MKL Config
if (MSVC)
	if (NOT EXISTS "$ENV{ICPP_COMPILER19}" )
		message (FATAL_ERROR "Please install MKL libraries and set env var ICPP_COMPILER19 to the MKL directory. Should be similar to: C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries\\windows\\. ")
	else()
		if ($ENV{ICPP_COMPILER19} MATCHES "\\$"  OR $ENV{ICPP_COMPILER19} MATCHES "/$" )
			set(INTEL_ROOT $ENV{ICPP_COMPILER19})
		else()
			message(STATUS "Appending trailing backslash to Intel tools path")
			set(INTEL_ROOT "$ENV{ICPP_COMPILER19}\\")
		endif()
	endif()
	set(MKL_ROOT ${INTEL_ROOT}/mkl)
	add_compile_options(/arch:AVX2 /Qpar)
	link_libraries("${INTEL_ROOT}/mkl/lib/intel64/mkl_core_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_rt.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_thread_dll.lib" "${INTEL_ROOT}/compiler/lib/intel64/libiomp5md.lib"  "${INTEL_ROOT}/mkl/lib/intel64/mkl_intel_ilp64_dll.lib" "${INTEL_ROOT}/mkl/lib/intel64/mkl_sequential_dll.lib")
	
else()
	set(INTEL_ROOT /opt/intel/compilers_and_libraries/linux)
	set(MKL_ROOT ${INTEL_ROOT}/mkl)
	add_compile_options(-m64 -Wl,--no-as-needed)
	link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
	link_directories(${INTEL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64)
endif()

add_definitions(-DMKL_ILP64)
include_directories(include ${INTEL_ROOT}/include ${MKL_ROOT}/include)


#Main compiler/linker settings 
if(MSVC)
	#language options
	add_compile_options(/permissive- /openmp /Zc:wchar_t /Zc:twoPhase- /Zc:forScope /Zc:inline /WX- /GR- /std:c++14 /Gd /W3 /MP /Zi /sdl- /FC /nologo /diagnostics:classic) 
	#code generation options
	add_compile_options(/Qpar /fp:fast /Zp8 /fp:except- /EHsc /GS- /Gm- /Gy )
	#optimization options
	add_compile_options(/Ot /Oy /Oi /GL)
	#path options
	#add_compile_options(/Fdx64/Release/vc141.pdb /Fox64/Release/)
	add_definitions(-D__NSG_WINDOWS__ -DUSE_AVX2 -DUSE_ACCELERATED_PQ -D_WINDOWS -DNOMINMAX -DUNICODE)
	
	set(CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS "/MANIFEST /MACHINE:X64 /LTCG:incremental /NXCOMPAT /DYNAMICBASE /OPT:REF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\"")
	set(CMAKE_EXECUTABLE_CXX_LINK_FLAGS "/MANIFEST /MACHINE:X64 /LTCG:incremental /NXCOMPAT /DYNAMICBASE /OPT:REF /INCREMENTAL:NO /SUBSYSTEM:CONSOLE /MANIFESTUAC:\"level='asInvoker' uiAccess='false'\"")

	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D_DEBUG")
	set(CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS_DEBUG "${CMAKE_SHARED_LIBRARY_CXX_LINK_FLAGS_DEBUG} /DEBUG")

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/x64/Debug)

	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/x64/Release)
else()
	set(ENV{TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD} 500000000000)
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -DNDEBUG -march=native -mtune=native -ftree-vectorize")
	add_compile_options(-march=native -Wall -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_ACCELERATED_PQ -DUSE_AVX2)
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tests/utils)
