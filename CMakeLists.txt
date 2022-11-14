cmake_minimum_required(VERSION 3.12)
project(PyCudaExtension)
 
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/libs)

add_subdirectory(${PROJECT_SOURCE_DIR}/src)

