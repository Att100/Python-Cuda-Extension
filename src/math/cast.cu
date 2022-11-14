#include "pycuda_utils.h"
#include "cuda_math.h"

template <typename DtypeA, typename DtypeB>
__global__ void typeCast(DtypeA* a, DtypeB* b, int n) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) { b[i] = (DtypeB)a[i]; i += gridDim.x + blockDim.x;}
}

DllExport void castIntToFloat(int* a, float* b, int n) {
	typeCast<int, float><<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD>>>(a, b, n);
}

DllExport void castFloatToInt(float* a, int* b, int n) {
	typeCast<float, int><<<get_number_of_blocks(n, MAX_N_THREAD), MAX_N_THREAD >>> (a, b, n);
}