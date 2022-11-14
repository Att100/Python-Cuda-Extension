#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include "pycuda_array.hpp"
#include "pycuda_math.h"

PYBIND11_MODULE(_pycuda, m) {
    m.doc() = "cuda array extension for python";
    // CuArray Extension (CuArrayInt)
    py::class_<CuArray<int>>(m, "CuArrayInt", py::dynamic_attr())
        .def(py::init<py::array_t<int>>())
        .def("numpy", &CuArray<int>::numpy, "numpy")
        .def("free", &CuArray<int>::_free, "free")
        .def("size", &CuArray<int>::size, "size")
        .def("copy", &CuArray<int>::copy, "copy")
        .def("cuarray_slices", &CuArray<int>::cuarray_slices, "cuarray_slices")
        .def("view", &CuArray<int>::view, "view")
        .def("contiguous", &CuArray<int>::contiguous, "contiguous")
        .def("set_values", &CuArray<int>::set_values, "set_values")
        .def("is_contiguous", &CuArray<int>::is_contiguous, "is_contiguous");
    // CuArray Extension (CuArrayFloat)
    py::class_<CuArray<float>>(m, "CuArrayFloat", py::dynamic_attr())
        .def(py::init<py::array_t<float>>())
        .def("numpy", &CuArray<float>::numpy, "numpy")
        .def("free", &CuArray<float>::_free, "free")
        .def("size", &CuArray<float>::size, "size")
        .def("copy", &CuArray<float>::copy, "copy")
        .def("cuarray_slices", &CuArray<float>::cuarray_slices, "cuarray_slices")
        .def("view", &CuArray<float>::view, "view")
        .def("contiguous", &CuArray<float>::contiguous, "contiguous")
        .def("set_values", &CuArray<float>::set_values, "set_values")
        .def("is_contiguous", &CuArray<float>::is_contiguous, "is_contiguous");
    // type cast functions
    m.def("cast_int_to_float", &cast_Int_to_Float, "");
    m.def("cast_float_to_int", &cast_Float_to_Int, "");
    // broadcast functions
    m.def("broadcastf_to", &broadcastf_to, "");
    m.def("broadcasti_to", &broadcasti_to, "");
    // transpose fucntions
    m.def("transposef", &transposef, "");
    m.def("transposei", transposei, "");
    // cublas methods
    m.def("matmulf", &matmulf, "");
    // basic element-wise operation functions
    m.def("add", &_cuda_add, "");
    m.def("sub", &_cuda_sub, "");
    m.def("mul", &_cuda_mul, "");
    m.def("div", &_cuda_div, "");
    m.def("lt", &_cuda_lt, "");
    m.def("le", &_cuda_le, "");
    m.def("gt", &_cuda_gt, "");
    m.def("ge", &_cuda_ge, "");
    m.def("eq", &_cuda_eq, "");
    m.def("ne", &_cuda_ne, "");
    // basic element-constant operation functions
    m.def("addc", &_cuda_addc, "");
    m.def("subc", &_cuda_subc, "");
    m.def("mulc", &_cuda_mulc, "");
    m.def("divc", &_cuda_divc, "");
    m.def("ltc", &_cuda_ltc, "");
    m.def("lec", &_cuda_lec, "");
    m.def("gtc", &_cuda_gtc, "");
    m.def("gec", &_cuda_gec, "");
    m.def("eqc", &_cuda_eqc, "");
    m.def("nec", &_cuda_nec, "");
    m.def("_modc", &_cuda_modc, "");
    // math element-wise operation functions
    m.def("_floor", &_cuda_floor, "");
    m.def("_ceil", &_cuda_ceil, "");
    m.def("_mod", &_cuda_mod, "");
    m.def("_round", &_cuda_round, "");
    m.def("_exp", &_cuda_exp, "");
    m.def("_exp2", &_cuda_exp2, "");
    m.def("_exp10", &_cuda_exp10, "");
    m.def("_log", &_cuda_log, "");
    m.def("_log2", &_cuda_log2, "");
    m.def("_log10", &_cuda_log10, "");
    m.def("_pow", &_cuda_pow, "");
    m.def("_sqrt", &_cuda_sqrt, "");
}
