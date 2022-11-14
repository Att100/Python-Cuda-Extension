#ifndef PYCUDA_ARRAY_H
#define PYCUDA_ARRAY_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "cuda_runtime.h"
#include "iostream"
#include "vector"
#include "string.h"

#include "pycuda_utils.h"
#include "cuda_math.h"

namespace py = pybind11;


// Cuda Array Descriptor 
class CuArrayDescriptor{   

public:
DllExport CuArrayDescriptor(int _n_dims, int* _shape){
    n_dims = _n_dims;
    shape = (int*)malloc(sizeof(int) * _n_dims);
    strides = (int*)malloc(sizeof(int) * _n_dims);
    for (int i=0;i<_n_dims;i++){
        n_size *= _shape[i];
        shape[i] = (int)_shape[i];
    }
    trShapeToStrides(shape, strides, n_dims);
}

DllExport CuArrayDescriptor(int _n_dims, pybind11::ssize_t* _shape){
    n_dims = _n_dims;
    shape = (int*)malloc(sizeof(int) * _n_dims);
    strides = (int*)malloc(sizeof(int) * _n_dims);
    for (int i=0;i<_n_dims;i++){
        n_size *= (int)_shape[i];
        shape[i] = (int)_shape[i];
    }
    trShapeToStrides(shape, strides, n_dims);
}

DllExport CuArrayDescriptor(CuArrayDescriptor org_descriptor, int* dims_start, int* dims_stop){
    int* _strides = org_descriptor.strides;
    int _n_dims = org_descriptor.n_dims;
    shape = (int*)malloc(sizeof(int) * _n_dims);
    strides = (int*)malloc(sizeof(int) * _n_dims);
    for (int i=0;i<_n_dims;i++){
        int _dim = dims_stop[i] - dims_start[i];
        ptr_offset += dims_start[i] * _strides[i];
        if (_dim > 0){
            n_size *= _dim;
            shape[n_dims] = _dim;
            strides[n_dims] = _strides[i];
            n_dims += 1;
        }
    }
    shared = true;
    contiguous = false;
}

DllExport CuArrayDescriptor(const CuArrayDescriptor &org_descriptor){
    n_dims = org_descriptor.n_dims;
    n_size = org_descriptor.n_size;
    ptr_offset = org_descriptor.ptr_offset;
    shared = org_descriptor.shared;
    contiguous = org_descriptor.contiguous;
    shape = (int*)malloc(sizeof(int) * n_dims);
    strides = (int*)malloc(sizeof(int) * n_dims);
    memcpy(shape, org_descriptor.shape, sizeof(int) * n_dims);
    memcpy(strides, org_descriptor.strides, sizeof(int) * n_dims);
}

DllExport CuArrayDescriptor(){}

int n_dims = 0;
int n_size = 1;
int ptr_offset = 0;
int* shape;
int* strides;
bool shared = false;
bool contiguous = true; 
};


// Cuda Array
template <typename Dtype> class CuArray {

public:
DllExport CuArray(CuArrayDescriptor _descriptor) {
    descriptor = _descriptor;
    check_cuda(cudaMalloc((void**)&device_ptr, descriptor.n_size * sizeof(Dtype)), 0);
}

DllExport CuArray(CuArrayDescriptor _descriptor, Dtype* &_device_ptr) {
    descriptor = _descriptor;
    device_ptr = _device_ptr;
}

DllExport CuArray(int _n_size, int _dims, std::vector<int> _shape) {
    n_size = _n_size;
    n_dims = _dims;
    shape = _shape;
    check_cuda(cudaMalloc((void**)&device_ptr, n_size * sizeof(Dtype)), 0);
}

DllExport CuArray(py::array_t<Dtype> np_arr) {
    _from_numpy(np_arr);
}

DllExport CuArray(Dtype* &_device_ptr, int _n_size, int _dims, std::vector<int> _shape){
    device_ptr = _device_ptr;
    n_size = _n_size;
    n_dims = _dims;
    shape = _shape;
}

DllExport CuArray(){}

// memory operation methods
DllExport void _from_numpy(py::array_t<Dtype> np_arr) {
    py::buffer_info buf = np_arr.request(true);
    descriptor = CuArrayDescriptor(buf.ndim, buf.shape.data());
    check_cuda(cudaMalloc((void**)&device_ptr, descriptor.n_size * sizeof(Dtype)), 0);
    check_cuda(cudaMemcpy(device_ptr, (Dtype*)buf.ptr, descriptor.n_size * sizeof(Dtype), cudaMemcpyHostToDevice), 1);
}

DllExport py::array_t<Dtype> numpy() {
    if (descriptor.contiguous){
        std::vector<py::ssize_t> shape_v, stride_v;
        Dtype* ptr_cpy = (Dtype*)malloc(descriptor.n_size * sizeof(Dtype));
        for (int j = 0; j < descriptor.n_dims; j++) {
            shape_v.push_back(descriptor.shape[j]);
            stride_v.push_back(descriptor.strides[j] * sizeof(Dtype));
        }
        check_cuda(cudaMemcpy(ptr_cpy, device_ptr, descriptor.n_size * sizeof(Dtype), cudaMemcpyDeviceToHost), 1);
        return py::array_t<Dtype>(shape_v, stride_v, ptr_cpy);
    }
    else{
        CuArray contiguous_arr = contiguous();
        py::array_t<Dtype> np_arr = contiguous_arr.numpy();
        contiguous_arr._free();
        return np_arr;
    }
}

DllExport void _free() {
    if (descriptor.contiguous && !descriptor.shared) {
        check_cuda(cudaFree(device_ptr), 2);
    }
    device_ptr = nullptr;
}

DllExport CuArray<Dtype> copy(){
    if (iscontiguous){
        CuArray arr_cpy = CuArray(CuArrayDescriptor(descriptor));
        check_cuda(cudaMemcpy(arr_cpy.get_ptr(), device_ptr, descriptor.n_size * sizeof(Dtype), cudaMemcpyDeviceToDevice), 1);
        return arr_cpy;
    }
    else{ 
        return contiguous();
    }
}

DllExport CuArray<Dtype> cuarray_slices(std::vector<int> &dims_start, std::vector<int> &dims_stop){
    CuArray<Dtype> arr_cpy = CuArray(CuArrayDescriptor(
        descriptor, dims_start.data(), dims_stop.data()), device_ptr);
    return arr_cpy;
}

DllExport CuArray<Dtype> view(std::vector<int> new_shape){
    // CuArray should be contiguously stored in memory
    CuArrayDescriptor _descriptor = CuArrayDescriptor(new_shape.size(), new_shape.data());
    _descriptor.shared = true;
    CuArray<Dtype> new_arr = CuArray(_descriptor, device_ptr);
    return new_arr;
}

DllExport CuArray<Dtype> contiguous(){
    CuArrayDescriptor _descriptor = CuArrayDescriptor(descriptor.n_dims, descriptor.shape);
    _descriptor.shared = false;
    _descriptor.contiguous = true;
    _descriptor.ptr_offset = 0;
    CuArray arr_cpy = CuArray(_descriptor);
    copyToContiguous(
        device_ptr, arr_cpy.get_ptr(), _descriptor.n_dims, descriptor.ptr_offset,
        _descriptor.n_size, _descriptor.strides, _descriptor.shape,
        descriptor.strides);
    return arr_cpy;
}

DllExport void set_values(CuArray<Dtype> arr){
    int* strides_slice = (int*)malloc(sizeof(int)*n_dims);
    trShapeToStrides(descriptor.shape, strides_slice, descriptor.n_dims);
    copyDiscontinuous(
        arr.get_ptr(), device_ptr, descriptor.n_size, descriptor.n_dims,
        strides_slice, descriptor.shape,
        arr.get_descriptor().ptr_offset, arr.get_descriptor().strides,
        descriptor.ptr_offset, descriptor.strides);
    free(strides_slice);
}

DllExport py::list size() {
    py::list list = py::list();
    for (int i = 0; i < descriptor.n_dims; i++) {
        list.append(descriptor.shape[i]);
    }
    return list;
}

// properties getter methods
DllExport CuArrayDescriptor get_descriptor(){ return descriptor; }
DllExport Dtype* get_ptr() { return device_ptr; }
DllExport int get_n_dims() { return descriptor.n_dims; }
DllExport int get_n_size() { return descriptor.n_size; }
DllExport int get_offset() { return descriptor.ptr_offset; }
DllExport bool is_contiguous(){ return descriptor.contiguous; }
DllExport bool is_shared(){ return descriptor.shared; }
DllExport int* get_shape() { return descriptor.shape; }
DllExport int* get_strides() { return descriptor.strides; }

// properties setter methods
DllExport void set_shared_status(bool status){ shared = status; }
DllExport void set_ptr_offset(int offset){ ptr_offset = offset; }
DllExport void set_strides(std::vector<int> _strides){ strides = _strides;}
DllExport void set_contiguous_status(bool status){ iscontiguous = status; }

private:
    // cuda array descriptor
    CuArrayDescriptor descriptor;
    int n_dims = 0;  // n dims
    int n_size = 1;
    int ptr_offset = 0;
    bool shared = false;
    bool isfree = false;
    bool iscontiguous = true;
    std::vector<int> shape;
    std::vector<int> strides;
    // device pointer to cuda array
    Dtype* device_ptr = nullptr;
};


#endif // !CUARRAY_H
