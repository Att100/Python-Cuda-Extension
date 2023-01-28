#ifndef PYCUDA_UTILS_H
#define PYCUDA_UTILS_H

#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "cmath"
#include "iostream"
#include "algorithm"
#include "vector"

#ifndef DllExport
#   if defined(_WIN64) || defined(_WIN32)
#       define DllExport _declspec(dllexport)
#   else
#       define DllExport
#   endif
#endif // DllExport

// CUDA Error Check Funtions
#define CHECKMALLOC 0
#define CHECKMEMCPY 1
#define CHECKFREE 2
#define CHECKMEMSET 3
DllExport void check_cuda(cudaError_t code, int type);

// kernel assistant methods
#define MAX_N_THREAD 512
#define MAX_N_BLOCK 65535

// Block and Thread Configuration
// https://github.com/AlexeyAB/darknet/blob/master/src/dark_cuda.c
DllExport dim3 cuda_gridsize(size_t n);
DllExport int get_number_of_blocks(int n_elements, int n_threads);

DllExport void trShapeToStrides(int* shape, int* strides, int _n_dims);

// unroll ndims op kernel wrapper
#define UnrollMaxDims 8
#define UnrollKernelNDimsOpWrapper(ndims, Func, T, ...) \
    do {                                              \
        switch (ndims){                               \
        case 1:                                       \
            Func<T, 1>(__VA_ARGS__);                  \
            break;                                    \
        case 2:                                       \
            Func<T, 2>(__VA_ARGS__);                  \
            break;                                    \
        case 3:                                       \
            Func<T, 3>(__VA_ARGS__);                  \
            break;                                    \
        case 4:                                       \
            Func<T, 4>(__VA_ARGS__);                  \
            break;                                    \
        case 5:                                       \
            Func<T, 5>(__VA_ARGS__);                  \
            break;                                    \
        case 6:                                       \
            Func<T, 6>(__VA_ARGS__);                  \
            break;                                    \
        case 7:                                       \
            Func<T, 7>(__VA_ARGS__);                  \
            break;                                    \
        case 8:                                       \
            Func<T, 8>(__VA_ARGS__);                  \
            break;                                    \
        default:                                      \
            break;                                    \
        }                                             \
    }while(false)                                     \

#endif