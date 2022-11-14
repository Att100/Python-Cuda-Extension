find_package(CUDA REQUIRED)
find_package(Python3 3.9 COMPONENTS Interpreter Development REQUIRED)

IF (WIN32)
    set(CUDA_NVCC_FLAGS -arch=compute_60)
ELSEIF (UNIX)
    set(CUDA_NVCC_FLAGS -arch=compute_60 -Xcompiler -std=c++11 -Xcompiler -fPIC) 
ELSE()
    message("Operating System Not Recognized!!")
ENDIF()

file(
    GLOB files 
    ${PROJECT_SOURCE_DIR}/src/*.cpp 
    ${PROJECT_SOURCE_DIR}/src/array/*.cpp
    ${PROJECT_SOURCE_DIR}/src/math/*.cu
    ${PROJECT_SOURCE_DIR}/include/*hpp)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

# for cublas and python(windows)
IF (WIN32)
    string(CONCAT pylib "python" "${Python3_VERSION_MAJOR}" "${Python3_VERSION_MINOR}")
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${pylib})
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
ELSEIF (UNIX)
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
ELSE()
    message("Operating System Not Recognized!!")
ENDIF()

link_libraries(cublas)

cuda_add_library(_pycuda SHARED ${files})

IF (WIN32)
    add_custom_command(
        TARGET _pycuda
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/build/libs/Release/_pycuda.dll ${PROJECT_SOURCE_DIR}/pycuda/libs/_pycuda.pyd
        )
ELSEIF (UNIX)
    add_custom_command(
        TARGET _pycuda
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/build/libs/lib_pycuda.so ${PROJECT_SOURCE_DIR}/pycuda/libs/_pycuda.so)
ELSE()
    message("Operating System Not Recognized!!")
ENDIF()


