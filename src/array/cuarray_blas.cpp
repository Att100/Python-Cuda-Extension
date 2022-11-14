#include "pycuda_math.h"


DllExport CuArray<float> matmulf(
    CuArray<float> arr_a, CuArray<float> arr_b){
    
    int M = arr_a.get_descriptor().shape[0];
    int K = arr_a.get_descriptor().shape[1];
    int N = arr_b.get_descriptor().shape[1];

    int* out_shape = (int*)malloc(2 * sizeof(int));
    out_shape[0] = M; out_shape[1] = N;
    
    CuArray<float> out = CuArray<float>(CuArrayDescriptor(2, out_shape));
    CuArray<float> _arr_a, _arr_b;
    _arr_a = arr_a.is_contiguous() ? arr_a : arr_a.contiguous();
    _arr_b = arr_b.is_contiguous() ? arr_b : arr_b.contiguous();

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha=1.0;
    float beta=0.0;
    cublasSgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        N, M, K, 
        &alpha, 
        _arr_b.get_ptr(), N, 
        _arr_a.get_ptr(), K, 
        &beta, 
        out.get_ptr(), N);
    return out;
}