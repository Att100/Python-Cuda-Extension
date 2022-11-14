#include "pycuda_utils.h"
#include "cuda_math.h"


template <typename Dtype> __global__ void cudaTransposeKernel(
    Dtype* src, Dtype* dest, int n_dims, int n, int* axes,
    int* strides_in, int* strides_out,
    int* dims_in){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n){
        int new_lin_i = 0;
        int sub_idx = 0;
        for (int i=0;i<n_dims;i++){
            sub_idx = (int)(idx / strides_in[axes[i]]) % dims_in[axes[i]];
            new_lin_i += strides_out[i] * sub_idx;
        }
        dest[new_lin_i] = src[idx];
    }
}

// template <typename Dtype, int D> __global__ void cudaTransposeUnrollKernel(
//     Dtype* src, Dtype* dest, int n, int* axes,
//     int* strides_in, int* strides_out,
//     int* dims_in){

//     int idx = blockIdx.x*blockDim.x + threadIdx.x;
//     if (idx < n){
//         int new_lin_i = 0;
//         int sub_idx = 0;
// #pragma unroll
//         for (int i=0;i<D;i++){
//             sub_idx = (int)(idx / strides_in[axes[i]]) % dims_in[axes[i]];
//             new_lin_i += strides_out[i] * sub_idx;
//         }
//         dest[new_lin_i] = src[idx];
//     }
// }

// template <typename Dtype, int D> void cudaTransposeUnroll(
//     Dtype* src, Dtype* dest, int n, int* axes,
//     int* strides_in, int* strides_out,
//     int* dims_in){

//     cudaTransposeUnrollKernel<Dtype, D><<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD>>>(
//         src, dest, n, axes,
//         strides_in, strides_out,
//         dims_in);
// }

template <typename Dtype> DllExport void cudaTranspose(
    Dtype* src, Dtype* dest, int n_dims, int n, int* axes,
    int* strides_in, int* strides_out,
    int* dims_in){
    /*
    Transpose operation, axes transformation

    Args:
        Dtype* src: source data
        Dtype* dest: destination
        int n_dims: number of dims of array
        int* strides_in: the array strides on src memory
        int* strides_out: the array strides on dest memory
        int* dims_in: shape of input array
        int n: number of output element
    */
    int* _strides_in, *_strides_out, *_axes, *_dims_in;

    check_cuda(cudaMalloc((void**)&_strides_in, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_strides_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_axes, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_dims_in, sizeof(int) * n_dims), 0); 

    check_cuda(cudaMemcpy(_strides_in, strides_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_strides_out, strides_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);
    check_cuda(cudaMemcpy(_axes, axes, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);
    check_cuda(cudaMemcpy(_dims_in, dims_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);

    // if (n_dims <= UnrollMaxDims){
    //     UnrollKernelNDimsOpWrapper(
    //         n_dims,
    //         cudaTransposeUnroll,
    //         Dtype,
    //         src, dest, n, _axes,
    //         _strides_in, _strides_out,
    //         _dims_in
    //     );
    // } else {
        cudaTransposeKernel<Dtype><<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD>>>(
            src, dest, n_dims, n, _axes,
            _strides_in, _strides_out,
            _dims_in);
    // }

    check_cuda(cudaFree(_strides_in), 2);
    check_cuda(cudaFree(_strides_out), 2);
    check_cuda(cudaFree(_axes), 2);
    check_cuda(cudaFree(_dims_in), 2);
}

template DllExport void cudaTranspose<int>(
    int* src, int* dest, 
    int n_dims, int n, int* axes,
    int* strides_in, int* strides_out,
    int* dims_in);

template DllExport void cudaTranspose<float>(
    float* src, float* dest, 
    int n_dims, int n, int* axes,
    int* strides_in, int* strides_out,
    int* dims_in);
