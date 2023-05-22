#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(MSVC)
    add_subdirectory(dll)
else()
    file(GLOB CPP_SOURCES *.cpp)
    if (RESTAPI)
        file(GLOB RESTAPI_CPP_SOURCES restapi/*.cpp)
        list(APPEND CPP_SOURCES ${RESTAPI_CPP_SOURCES})
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
