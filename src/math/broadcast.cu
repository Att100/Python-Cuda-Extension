#include "pycuda_utils.h"
#include "cuda_math.h"

template <typename Dtype> __global__ void cudaBroadcastKernel(
    Dtype* src, Dtype* dest, 
    int n_dims, int* mask,
    int* dims_in, int* strides_in,
    int n_ele_out, int* dims_out, int* strides_out){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n_ele_out){
        int idx_src = 0;
        int idx_dest = 0;
        for (int j=0;j<n_dims;j++){
            int idx_dim = (int)(idx / strides_out[j]) % dims_out[j];
            idx_src += (idx_dim * mask[j]) * strides_in[j];
            idx_dest += idx_dim * strides_out[j];
        }
        dest[idx_dest] = src[idx_src];
        idx += gridDim.x + blockDim.x;
    }
}

// template <typename Dtype, int D> __global__ void cudaBroadcastUnrollKernel(
//     Dtype* src, Dtype* dest, 
//     int* mask,
//     int* dims_in, int* strides_in,
//     int n_ele_out, int* dims_out, int* strides_out){

//     int idx = blockIdx.x*blockDim.x + threadIdx.x;
//     if (idx < n_ele_out){
//         int idx_src = 0;
//         int idx_dest = 0;
// #pragma unroll
//         for (int j=0;j<D;j++){
//             int idx_dim = (int)(idx / strides_out[j]) % dims_out[j];
//             idx_src += (idx_dim * mask[j]) * strides_in[j];
//             idx_dest += idx_dim * strides_out[j];
//         }
//         dest[idx_dest] = src[idx_src];
//         idx += gridDim.x + blockDim.x;
//     }
// }

// template <typename Dtype, int D> void cudaBroadcastUnroll(
//     Dtype* src, Dtype* dest, 
//     int* mask,
//     int* dims_in, int* strides_in,
//     int n_ele_out, int* dims_out, int* strides_out){

//     cudaBroadcastUnrollKernel<Dtype, D><<<get_number_of_blocks(n_ele_out, MAX_N_THREAD), MAX_N_THREAD>>>(
//         src, dest, 
//         mask,
//         dims_in, strides_in,
//         n_ele_out, dims_out, strides_out);
// }

template <typename Dtype> DllExport void cudaBroadcast(
    Dtype* src, Dtype* dest, 
    int n_dims, int* mask,
    int* dims_in, int* strides_in,
    int n_ele_out, int* dims_out, int* strides_out){
    /*
    Broadcast operation, repeat data towards dimensions

    Args:
        Dtype* src: source data
        Dtype* dest: destination
        int n_dims: number of dims of array
        int* mask: mask that label the dimension with size 1
        int* dims_in: shape of src
        int* dims_out: shape of dest
        int* strides_in: the array strides on src memory
        int* strides_out: the array strides on dest memory
        int n_ele_out: number of output element
    */
    int* _dims_out, *_dims_in, *_strides_in, *_strides_out, *_mask;

    check_cuda(cudaMalloc((void**)&_strides_in, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_strides_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_dims_in, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_dims_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_mask, sizeof(int) * n_dims), 0); 

    check_cuda(cudaMemcpy(_strides_in, strides_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_strides_out, strides_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);
    check_cuda(cudaMemcpy(_dims_in, dims_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);
    check_cuda(cudaMemcpy(_dims_out, dims_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);
    check_cuda(cudaMemcpy(_mask, mask, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);

    // if (n_dims <= UnrollMaxDims){
    //     UnrollKernelNDimsOpWrapper(
    //         n_dims,
    //         cudaBroadcastUnroll,
    //         Dtype,
    //         src, dest, 
    //         _mask,
    //         _dims_in, _strides_in,
    //         n_ele_out, _dims_out, _strides_out
    //     );
    // }else{
        cudaBroadcastKernel<Dtype><<<get_number_of_blocks(n_ele_out, MAX_N_THREAD), MAX_N_THREAD>>>(
        src, dest, 
        n_dims, _mask,
        _dims_in, _strides_in,
        n_ele_out, _dims_out, _strides_out);
    // }
    
    
    check_cuda(cudaFree(_strides_in), 2);
    check_cuda(cudaFree(_strides_out), 2);
    check_cuda(cudaFree(_dims_in), 2);
    check_cuda(cudaFree(_dims_out), 2);
    check_cuda(cudaFree(_mask), 2);
}

template DllExport void cudaBroadcast<int>(
    int* src, int* dest, 
    int n_dims, int* mask,
    int* dims_in, int* strides_in,
    int n_ele_out, int* dims_out, int* strides_out);

template DllExport void cudaBroadcast<float>(
    float* src, float* dest, 
    int n_dims, int* mask,
    int* dims_in, int* strides_in,
    int n_ele_out, int* dims_out, int* strides_out);