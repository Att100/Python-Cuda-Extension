#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include "cuda_runtime.h"
#include "math.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "pycuda_utils.h"

#define PARALLEL_1D(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

// cuda operators kernels
// __device__ float _add_op(float x, float y);
// __device__ float _sub_op(float x, float y);
// __device__ float _mul_op(float x, float y);
// __device__ float _div_op(float x, float y);
// __device__ float _lt_op(float x, float y);
// __device__ float _le_op(float x, float y);
// __device__ float _gt_op(float x, float y);
// __device__ float _ge_op(float x, float y);
// __device__ float _eq_op(float x, float y);
// __device__ float _ne_op(float x, float y);


// cuda type cast methods
DllExport void castIntToFloat(int* a, float* b, int n);
DllExport void castFloatToInt(float* a, int* b, int n);

// cuda basic element-wise operation methods (support float only)
DllExport void cuda_add(float* x, float* y, float* z, int n);
DllExport void cuda_sub(float* x, float* y, float* z, int n);
DllExport void cuda_mul(float* x, float* y, float* z, int n);
DllExport void cuda_div(float* x, float* y, float* z, int n);
DllExport void cuda_lt(float* x, float* y, float* z, int n);
DllExport void cuda_le(float* x, float* y, float* z, int n);
DllExport void cuda_gt(float* x, float* y, float* z, int n);
DllExport void cuda_ge(float* x, float* y, float* z, int n);
DllExport void cuda_eq(float* x, float* y, float* z, int n);
DllExport void cuda_ne(float* x, float* y, float* z, int n);

// cuda basic element-constant operation methods (support float only)
DllExport void cuda_addc(float* x, float y, float* z, int n);
DllExport void cuda_subc(float* x, float y, float* z, int n);
DllExport void cuda_mulc(float* x, float y, float* z, int n);
DllExport void cuda_divc(float* x, float y, float* z, int n);
DllExport void cuda_ltc(float* x, float y, float* z, int n);
DllExport void cuda_lec(float* x, float y, float* z, int n);
DllExport void cuda_gtc(float* x, float y, float* z, int n);
DllExport void cuda_gec(float* x, float y, float* z, int n);
DllExport void cuda_eqc(float* x, float y, float* z, int n);
DllExport void cuda_nec(float* x, float y, float* z, int n);
DllExport void cuda_modc(float* x, float y, float* z, int n);

// cuda math operations (support float only)
DllExport void cuda_floor(float* x, float* y, int n);
DllExport void cuda_ceil(float* x, float* y, int n);
DllExport void cuda_mod(float* x, float* y, float* z, int n);
DllExport void cuda_round(float* x, float* y, int n);
DllExport void cuda_log(float* x, float* y, int n);
DllExport void cuda_log2(float* x, float* y, int n);
DllExport void cuda_log10(float* x, float* y, int n);
DllExport void cuda_exp(float* x, float* y, int n);
DllExport void cuda_exp2(float* x, float* y, int n);
DllExport void cuda_exp10(float* x, float* y, int n);
DllExport void cuda_pow(float* x, float* y, float p, int n);
DllExport void cuda_sqrt(float* x, float* y, int n);

// assistant operations
template <typename Dtype> DllExport void copyToContiguous(Dtype* src, Dtype* dest, 
    int n_dims, int ptr_offset,
    int n_ele_out, int* strides_out, int* dims_out,
    int* strides_in);
template <typename Dtype> DllExport void copyDiscontinuous(
    Dtype* src, Dtype* dest, int n_ele, int n_dims, 
    int* strides_slice, int* dims_slice,
    int ptr_offset_in, int* strides_in,
    int ptr_offset_out, int* strides_out);
template <typename Dtype> DllExport void cudaBroadcast(
    Dtype* src, Dtype* dest, 
    int n_dims, int* mask,
    int* dims_in, int* strides_in,
    int n_ele_out, int* dims_out, int* strides_out);
template <typename Dtype> DllExport void cudaTranspose(
    Dtype* src, Dtype* dest, 
    int n_dims, int n, int* axes,
    int* strides_in, int* strides_out,
    int* dims_in);

#endif