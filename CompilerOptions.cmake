if(MSVC)
	#changing default target to X64
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_MODULE_LINKER_FLAGS_INIT "${CMAKE_MODULE_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_STATIC_LINKER_FLAGS_INIT "${CMAKE_STATIC_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_EXE_LINKER_FLAGS_INIT "${CMAKE_EXE_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_MODULE_LINKER_FLAGS_INIT "${CMAKE_MODULE_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_SHARED_LINKER_FLAGS_INIT "${CMAKE_SHARED_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "/[M|m][A|a][C|c][H|h][I|i][N|n][E|e]:[X|x]86" "/MACHINE:X64" CMAKE_STATIC_LINKER_FLAGS_INIT "${CMAKE_STATIC_LINKER_FLAGS_INIT}")
	string(REGEX REPLACE "Debug" "Release" CMAKE_BUILD_TYPE_INIT "${CMAKE_BUILD_TYPE_INIT}")
endif()


get_cmake_property(_varNames VARIABLES)
list (REMOVE_DUPLICATES _varNames)
list (SORT _varNames)
foreach (_varName ${_varNames})
	message(STATUS "${_varName}=${${_varName}}")
endforeach()

