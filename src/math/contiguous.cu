#include "pycuda_utils.h"
#include "cuda_math.h"

template <typename Dtype> __global__ void copyToContiguousKernel(
    Dtype* src, Dtype* dest, 
    int n_dims, int ptr_offset,
    int n_ele_out, int* strides_out, int* dims_out,
    int* strides_in){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;  // = idx in new array 
    if (idx < n_ele_out){
        int idx_org = ptr_offset;
        for (int j=0;j<n_dims;j++){
            int idx_dim = (int)(idx / strides_out[j]) % dims_out[j];
            idx_org += idx_dim * strides_in[j];
        }
        dest[idx] = src[idx_org];
        idx += gridDim.x + blockDim.x;
    }
}

// template <typename Dtype, int D> __global__ void copyToContiguousUnrollKernel(
//     Dtype* src, Dtype* dest, 
//     int ptr_offset,
//     int n_ele_out, int* strides_out, int* dims_out,
//     int* strides_in){

//     int idx = blockIdx.x*blockDim.x + threadIdx.x;  // = idx in new array 
//     if (idx < n_ele_out){
//         int idx_org = ptr_offset;
// #pragma unroll
//         for (int j=0;j<D;j++){
//             int idx_dim = (int)(idx / strides_out[j]) % dims_out[j];
//             idx_org += idx_dim * strides_in[j];
//         }
//         dest[idx] = src[idx_org];
//         idx += gridDim.x + blockDim.x;
//     }
// }

// template <typename Dtype, int D> void copyToContiguousUnroll(
//         Dtype* src, Dtype* dest, 
//         int ptr_offset,
//         int n_ele_out, int* strides_out, int* dims_out,
//         int* strides_in){

//         copyToContiguousUnrollKernel<Dtype, D><<<get_number_of_blocks(n_ele_out, MAX_N_THREAD), MAX_N_THREAD>>>(
//             src, dest, 
//             ptr_offset,
//             n_ele_out, strides_out, dims_out,
//             strides_in);        
// }

template <typename Dtype> DllExport void copyToContiguous(Dtype* src, Dtype* dest, 
        int n_dims, int ptr_offset,
        int n_ele_out, int* strides_out, int* dims_out,
        int* strides_in){
    /*
    Copy a piece of discontinuous memory to a continuous memory

    Args:
        Dtype* src: source data
        Dtype* dest: destination
        int n_dims: number of dims of array
        int ptr_offset: the offset between the source memory starter and
                        the starter of array need to copy
        int n_ele_out: number of the element of output array
        int* strides_out: the array strides on output of continuous memory
        int* dims_out: the shape of ouput array
        int* strides_in: the strides on source memory (original array)
    */
    int* _strides_out, *_strides_in, *_dims_out;

    check_cuda(cudaMalloc((void**)&_strides_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_strides_in, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_dims_out, sizeof(int) * n_dims), 0); 

    check_cuda(cudaMemcpy(_strides_out, strides_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_strides_in, strides_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_dims_out, dims_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 

    // if (n_dims <= UnrollMaxDims){
    //     UnrollKernelNDimsOpWrapper(
    //         n_dims,
    //         copyToContiguousUnroll,
    //         Dtype,
    //         src, dest, 
    //         ptr_offset,
    //         n_ele_out, _strides_out, _dims_out,
    //         _strides_in
    //     );
    // }else{
        copyToContiguousKernel<Dtype><<<get_number_of_blocks(n_ele_out, MAX_N_THREAD), MAX_N_THREAD>>>(
        src, dest, n_dims, ptr_offset,
        n_ele_out, _strides_out, _dims_out,
        _strides_in);
    // }
    
    check_cuda(cudaFree(_strides_out), 2);
    check_cuda(cudaFree(_strides_in), 2);
    check_cuda(cudaFree(_dims_out), 2);
}

template <typename Dtype> __global__ void copyDiscontinuousKernel(
    Dtype* src, Dtype* dest, int n_ele, int n_dims, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n_ele){
        int idx_src = ptr_offset_in;
        int idx_dest = ptr_offset_out;
        for (int j=0;j<n_dims;j++){
            int idx_dim = (int)(idx / strides_slice[j]) % dims_slice[j];
            idx_src += idx_dim * strides_in[j];
            idx_dest += idx_dim * strides_out[j];
        }
        dest[idx_dest] = src[idx_src];
        idx += gridDim.x + blockDim.x;
    }
}

template <typename Dtype, int D> __global__ void copyDiscontinuousUnrollKernel(
    Dtype* src, Dtype* dest, int n_ele, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n_ele){
        int idx_src = ptr_offset_in;
        int idx_dest = ptr_offset_out;
#pragma unroll
        for (int j=0;j<D;j++){
            int idx_dim = (int)(idx / strides_slice[j]) % dims_slice[j];
            idx_src += idx_dim * strides_in[j];
            idx_dest += idx_dim * strides_out[j];
        }
        dest[idx_dest] = src[idx_src];
        idx += gridDim.x + blockDim.x;
    }
}

template <typename Dtype, int D> void copyDiscontinuousUnroll(
    Dtype* src, Dtype* dest, int n_ele, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out){
    
    copyDiscontinuousUnrollKernel<Dtype, D><<<get_number_of_blocks(n_ele, MAX_N_THREAD), MAX_N_THREAD>>>(
        src, dest, n_ele, 
        strides_slice, dims_slice,
        ptr_offset_in, strides_in,
        ptr_offset_out, strides_out
    );
}

template <typename Dtype> DllExport void copyDiscontinuous(
    Dtype* src, Dtype* dest, int n_ele, int n_dims, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out){
    /*
    Copy a piece of discontinuous memory from source to destination

    Args:
        Dtype* src: source data
        Dtype* dest: destination
        int n_ele: number of element of data to be copied
        int n_dims: number of dims of array
        int* strides_slice: the stride of slice to be copied
        int* dims_slice: the shape of slice to be copied
        int ptr_offset_in: the offset between the source memory starter and
                        the starter of src memory
        int ptr_offset_out: the offset between the source memory starter and
                        the starter of dest memory
        int* strides_in: the array strides on src memory
        int* strides_out: the array strides on dest memory
    */
    int* _strides_out, *_strides_in, *_dims_slice, *_strides_slice;

    check_cuda(cudaMalloc((void**)&_strides_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_strides_in, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_dims_slice, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_strides_slice, sizeof(int) * n_dims), 0); 

    check_cuda(cudaMemcpy(_strides_out, strides_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_strides_in, strides_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_dims_slice, dims_slice, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1);
    check_cuda(cudaMemcpy(_strides_slice, strides_slice, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 

    if (n_dims <= UnrollMaxDims){
        UnrollKernelNDimsOpWrapper(
            n_dims,
            copyDiscontinuousUnroll,
            Dtype,
            src, dest, n_ele, 
            _strides_slice, _dims_slice,
            ptr_offset_in, _strides_in,
            ptr_offset_out, _strides_out
        );
    }else{
        copyDiscontinuousKernel<Dtype><<<get_number_of_blocks(n_ele, MAX_N_THREAD), MAX_N_THREAD>>>(
            src, dest, n_ele, n_dims, 
            _strides_slice, _dims_slice,
            ptr_offset_in, _strides_in,
            ptr_offset_out, _strides_out);
    }
    
    check_cuda(cudaFree(_strides_out), 2);
    check_cuda(cudaFree(_strides_in), 2);
    check_cuda(cudaFree(_dims_slice), 2);
    check_cuda(cudaFree(_strides_slice), 2);
}

// Explicit instantiation !!
template DllExport void copyToContiguous<float>(float* src, float* dest, 
        int n_dims, int ptr_offset,
        int n_ele_out, int* strides_out, int* dims_out,
        int* strides_in);

template DllExport void copyToContiguous<int>(int* src, int* dest, 
        int n_dims, int ptr_offset,
        int n_ele_out, int* strides_out, int* dims_out,
        int* strides_in);

template DllExport void copyDiscontinuous<float>(
    float* src, float* dest, int n_ele, int n_dims, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out);

template DllExport void copyDiscontinuous<int>(
    int* src, int* dest, int n_ele, int n_dims, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out);