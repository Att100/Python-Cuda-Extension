#ifndef PYCUDA_MATH_H
#define PYCUDA_MATH_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "cuda_runtime.h"
#include "iostream"
#include "vector"
#include "string.h"

#include "pycuda_array.hpp"
#include "pycuda_utils.h"
#include "cuda_math.h"

// type cast methods
DllExport CuArray<int> cast_Float_to_Int(CuArray<float> int_arr);
DllExport CuArray<float> cast_Int_to_Float(CuArray<int> float_arr);

// broadcast methods
template <typename Dtype> 
DllExport CuArray<Dtype> broadcast_to(CuArray<Dtype> arr, py::list shape_in, py::list shape_out, py::list mask);
DllExport CuArray<float> broadcastf_to(CuArray<float> arr, py::list shape_in, py::list shape_out, py::list mask);
DllExport CuArray<int> broadcasti_to(CuArray<int> arr, py::list shape_in, py::list shape_out, py::list mask);

// transpose methods
template <typename Dtype> 
DllExport CuArray<Dtype> transpose(CuArray<Dtype> arr, py::list shape_in, py::list shape_out, py::list axes);
DllExport CuArray<float> transposef(CuArray<float> arr, py::list shape_in, py::list shape_out, py::list axes);
DllExport CuArray<int> transposei(CuArray<int> arr, py::list shape_in, py::list shape_out, py::list axes);

// cublas methods
DllExport CuArray<float> matmulf(CuArray<float> arr_a, CuArray<float> arr_b);


// CuArray basic operation (support float only)
using FtypeUnitary = void(float*, float*, int);
using FtypeBinary = void(float*, float*, float*, int);
using FtypeBinaryC = void(float*, float, float*, int);

DllExport CuArray<float> _cuda_add(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_sub(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_mul(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_div(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_lt(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_le(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_gt(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_ge(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_eq(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_ne(CuArray<float> x, CuArray<float> y);

DllExport CuArray<float> _cuda_addc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_subc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_mulc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_divc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_ltc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_lec(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_gtc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_gec(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_eqc(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_nec(CuArray<float> x, float y);
DllExport CuArray<float> _cuda_modc(CuArray<float> x, float y);

DllExport CuArray<float> _cuda_floor(CuArray<float> x);
DllExport CuArray<float> _cuda_ceil(CuArray<float> x);
DllExport CuArray<float> _cuda_mod(CuArray<float> x, CuArray<float> y);
DllExport CuArray<float> _cuda_round(CuArray<float> x);
DllExport CuArray<float> _cuda_log(CuArray<float> x);
DllExport CuArray<float> _cuda_log2(CuArray<float> x);
DllExport CuArray<float> _cuda_log10(CuArray<float> x);
DllExport CuArray<float> _cuda_exp(CuArray<float> x);
DllExport CuArray<float> _cuda_exp2(CuArray<float> x);
DllExport CuArray<float> _cuda_exp10(CuArray<float> x);
DllExport CuArray<float> _cuda_pow(CuArray<float> x, float p);
DllExport CuArray<float> _cuda_sqrt(CuArray<float> x);
    
#endif // !PYCUDA_KERNEL_H



