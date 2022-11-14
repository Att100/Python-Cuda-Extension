#include "pycuda_utils.h"

// CUDA Error Check Funtions
DllExport void check_cuda(cudaError_t code, int type){
    if (code != cudaSuccess) {
        if (type == CHECKMALLOC){
            printf("cudaFree failed\n");
        }
        else if (type == CHECKMEMCPY){
            printf("cudaMemcopy failed\n");
        }
        else if (type == CHECKFREE){
            printf("cudaFree failed\n");
        }
        else if (type == CHECKMEMSET){
            printf("cudaMemset failed\n");
        }
        else {
            printf("Error Not Recognized\n");
        }
        printf("reason: %s\n", cudaGetErrorString(code));
    }
}

// Kernel Execute Assistant Funtions
DllExport dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / MAX_N_THREAD + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*MAX_N_THREAD) + 1;
    }
    dim3 d;
    d.x = x;
    d.y = y;
    d.z = 1;
    return d;
}

DllExport int get_number_of_blocks(int n_elements, int n_threads){
    int n_blocks = (n_elements / (n_threads + (n_elements % n_threads > 0) ? 1 : 0));
    return (n_blocks > MAX_N_BLOCK) ? MAX_N_BLOCK : n_blocks;
}

DllExport void trShapeToStrides(int* shape, int* strides, int _n_dims){
    strides[_n_dims-1] = 1;
    for(int i=0;i<_n_dims;i++){
        if (i>0){
            strides[_n_dims-i-1] = strides[_n_dims-i] * shape[_n_dims-i];
        }
    }
}