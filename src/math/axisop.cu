#include "pycuda_utils.h"
#include "cuda_math.h"

template <typename Dtype> __global__ void axisSumKernel(
    Dtype* src, Dtype* dest, 
    int n_dims, int ptr_offset,
    int n_ele_out, int* strides_out, int* dims_out,
    int* strides_in, int* dims_in, int axis){
    
    // n_dims: dims number of output
    // strides_in: (stride of <axis> should be placed at the bottom of the list)
    //      original definition: [..., strides[axis], ...]
    //      input of this kernel func: [..., ..., strides[axis]] 

    int idx = blockIdx.x*blockDim.x + threadIdx.x;  // = idx in new array 
    if (idx < n_ele_out){
        int idx_org = ptr_offset;
        for (int j=0;j<n_dims;j++){
            int idx_dim = (int)(idx / strides_out[j]) % dims_out[j];
            idx_org += idx_dim * strides_in[j];
        }
        for (int i=0;i<dims_in[axis];i++){
            dest[idx] += src[idx_org + i * strides_in[n_dims]];
        }
        idx += gridDim.x + blockDim.x;
    }
}


template <typename Dtype> DllExport void axisSumKernel(Dtype* src, Dtype* dest, 
        int n_dims, int ptr_offset,
        int n_ele_out, int* strides_out, int* dims_out,
        int* strides_in, int* dims_in, int axis){
    /*
    Sum along an axis of an array

    Args:
        Dtype* src: source data
        Dtype* dest: destination
        int n_dims: number of dims of output array
        int ptr_offset: the offset between the source memory starter and
                        the starter of array need to copy
        int n_ele_out: number of the element of output array
        int* strides_out: the array strides on output of continuous memory
        int* dims_out: the shape of ouput array
        int* strides_in: the strides on source memory (original array)
        int* dims_in: the shape of input array
        int axis: axis to sum along
    */
    int* _strides_out, *_strides_in, *_dims_out, *_dims_in;

    check_cuda(cudaMalloc((void**)&_strides_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_strides_in, sizeof(int) * (n_dims+1)), 0); 
    check_cuda(cudaMalloc((void**)&_dims_out, sizeof(int) * n_dims), 0); 
    check_cuda(cudaMalloc((void**)&_dims_in, sizeof(int) * (n_dims+1)), 0); 

    check_cuda(cudaMemcpy(_strides_out, strides_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_strides_in, strides_in, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 
    check_cuda(cudaMemcpy(_dims_out, dims_out, sizeof(int) * n_dims, cudaMemcpyHostToDevice), 1); 

    axisSumKernel<Dtype><<<get_number_of_blocks(n_ele_out, MAX_N_THREAD), MAX_N_THREAD>>>(
        src, dest, 
        n_dims, ptr_offset,
        n_ele_out, _strides_out, _dims_out,
        _strides_in, _dims_in, axis);
    
    check_cuda(cudaFree(_strides_out), 2);
    check_cuda(cudaFree(_strides_in), 2);
    check_cuda(cudaFree(_dims_out), 2);
    check_cuda(cudaFree(_dims_in), 2);
}


// Explicit instantiation !!
template DllExport void axisSumKernel<float>(float* src, float* dest, 
        int n_dims, int ptr_offset,
        int n_ele_out, int* strides_out, int* dims_out,
        int* strides_in, int* dims_in, int axis);

template DllExport void axisSumKernel<int>(int* src, int* dest, 
        int n_dims, int ptr_offset,
        int n_ele_out, int* strides_out, int* dims_out,
        int* strides_in, int* dims_in, int axis);
