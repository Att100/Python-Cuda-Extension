#include "pycuda_utils.h"
#include "cuda_math.h"


// element-wise kernel functions on device (device)
__device__ float _add_op(float x, float y){ return x + y; }
__device__ float _sub_op(float x, float y){ return x - y; }
__device__ float _mul_op(float x, float y){ return x * y; }
__device__ float _div_op(float x, float y){ return x / y; }
__device__ float _lt_op(float x, float y){ return x < y; }
__device__ float _le_op(float x, float y){ return x <= y; }
__device__ float _gt_op(float x, float y){ return x > y; }
__device__ float _ge_op(float x, float y){ return x >= y; }
__device__ float _eq_op(float x, float y){ return x == y; }
__device__ float _ne_op(float x, float y){ return x != y; }

// element-constant kernel functions on device (global)
__global__ void _addc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_add_op(x[i],y);}} 
__global__ void _subc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_sub_op(x[i],y);}}
__global__ void _mulc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_mul_op(x[i],y);}}
__global__ void _divc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_div_op(x[i],y);}}
__global__ void _ltc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_lt_op(x[i],y);}}
__global__ void _lec(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_le_op(x[i],y);}}
__global__ void _gtc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_gt_op(x[i],y);}}
__global__ void _gec(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_ge_op(x[i],y);}}
__global__ void _eqc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_eq_op(x[i],y);}}
__global__ void _nec(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i]=_ne_op(x[i],y);}}
__global__ void _modc(float *x, float y, float *z, int n){PARALLEL_1D(i,n){z[i] = fmodf(x[i], y);}}

// element-wise kernel functions on device (global)
__global__ void _add(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_add_op(x[i],y[i]);}} 
__global__ void _sub(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_sub_op(x[i],y[i]);}}
__global__ void _mul(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_mul_op(x[i],y[i]);}}
__global__ void _div(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_div_op(x[i],y[i]);}}
__global__ void _lt(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_lt_op(x[i],y[i]);}}
__global__ void _le(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_le_op(x[i],y[i]);}}
__global__ void _gt(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_gt_op(x[i],y[i]);}}
__global__ void _ge(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_ge_op(x[i],y[i]);}}
__global__ void _eq(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_eq_op(x[i],y[i]);}}
__global__ void _ne(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i]=_ne_op(x[i],y[i]);}}
__global__ void _floor(float *x, float *y, int n){PARALLEL_1D(i,n){y[i]=floorf(x[i]);}}
__global__ void _ceil(float *x, float *y, int n){PARALLEL_1D(i,n){y[i]=ceilf(x[i]);}}
__global__ void _mod(float *x, float *y, float *z, int n){PARALLEL_1D(i,n){z[i] = fmodf(x[i], y[i]);}}
__global__ void _round(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = roundf(x[i]);}}
__global__ void _log(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = logf(x[i]);}}
__global__ void _log2(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = log2f(x[i]);}}
__global__ void _log10(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = log10f(x[i]);}}
__global__ void _exp(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = expf(x[i]);}}
__global__ void _exp2(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = exp2f(x[i]);}}
__global__ void _exp10(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = exp10f(x[i]);}}
__global__ void _pow(float *x, float *y, float p, int n){PARALLEL_1D(i,n){y[i] = powf(x[i], p);}}
__global__ void _sqrt(float *x, float *y, int n){PARALLEL_1D(i,n){y[i] = sqrtf(x[i]);}}

// wrapped element-constant kernel functions on device
DllExport void cuda_addc(float* x, float y, float* z, int n){
    _addc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_subc(float* x, float y, float* z, int n){
    _subc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_mulc(float* x, float y, float* z, int n){
    _mulc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_divc(float* x, float y, float* z, int n){
    _divc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_ltc(float* x, float y, float* z, int n){
    _ltc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_lec(float* x, float y, float* z, int n){
    _lec<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_gtc(float* x, float y, float* z, int n){
    _gec<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_gec(float* x, float y, float* z, int n){
    _gec<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_eqc(float* x, float y, float* z, int n){
    _eqc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_nec(float* x, float y, float* z, int n){
    _nec<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_modc(float* x, float y, float* z, int n){
    _modc<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}

// wrapped element-wise kernel functions on device
DllExport void cuda_add(float* x, float* y, float* z, int n){
    _add<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_sub(float* x, float* y, float* z, int n){
    _sub<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_mul(float* x, float* y, float* z, int n){
    _mul<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_div(float* x, float* y, float* z, int n){
    _div<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_lt(float* x, float* y, float* z, int n){
    _lt<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_le(float* x, float* y, float* z, int n){
    _le<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_gt(float* x, float* y, float* z, int n){
    _ge<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_ge(float* x, float* y, float* z, int n){
    _ge<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_eq(float* x, float* y, float* z, int n){
    _eq<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_ne(float* x, float* y, float* z, int n){
    _ne<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_floor(float* x, float* y, int n){
    _floor<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_ceil(float* x, float* y, int n){
    _ceil<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_mod(float* x, float* y, float* z, int n){
    _mod<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, z, n);
}
DllExport void cuda_round(float* x, float* y, int n){
    _round<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_log(float* x, float* y, int n){
    _log<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_log2(float* x, float* y, int n){
    _log2<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_log10(float* x, float* y, int n){
    _log10<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_exp(float* x, float* y, int n){
    _exp<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_exp2(float* x, float* y, int n){
    _exp2<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_exp10(float* x, float* y, int n){
    _exp10<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}
DllExport void cuda_pow(float* x, float* y, float p, int n){
    _pow<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, p, n);
}
DllExport void cuda_sqrt(float* x, float* y, int n){
    _sqrt<<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>>(x, y, n);
}