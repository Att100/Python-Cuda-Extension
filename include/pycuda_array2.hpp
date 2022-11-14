#ifndef PYCUDA_ARRAY2_H
#define PYCUDA_ARRAY2_H

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "cuda_runtime.h"
#include "iostream"
#include "vector"
#include "string.h"

#include "pycuda_utils.h"
#include "cuda_math.h"

namespace py = pybind11;

// CuArray utils
void shapeToStrides(int* shape, int* strides, int _n_dims){
    strides[_n_dims-1] = 1;
    for(int i=0;i<_n_dims;i++){
        if (i>0){
            strides[_n_dims-i-1] = strides[_n_dims-i] * shape[_n_dims-i];
        }
    }
}

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
    shapeToStrides(shape, strides, n_dims);
}

DllExport CuArrayDescriptor(CuArrayDescriptor org_descriptor, int* dims_start, int* dims_stop){
    int* _strides = org_descriptor.get_strides();
    int _n_dims = org_descriptor.get_n_dims();
    shape = (int*)malloc(sizeof(int) * _n_dims);
    strides = (int*)malloc(sizeof(int) * _n_dims);
    for (int i=0;i<n_dims;i++){
        int _dim = dims_stop[i] - dims_start[i];
        ptr_offset += dims_start[i] * _strides[i];
        n_size *= _dim;
        if (_dim != 1){
            shape[n_dims] = _dim;
            strides[n_dims] = strides[i];
            n_dims += 1;
        }
    }
    shared = true;
    iscontiguous = false;
}

DllExport CuArrayDescriptor(CuArrayDescriptor &org_descriptor){
    n_dims = org_descriptor.get_n_dims();
    n_size = org_descriptor.get_n_size();
    ptr_offset = org_descriptor.get_ptr_offset();
    shared = org_descriptor.get_shared_status();
    iscontiguous = org_descriptor.get_contiguous_status();
    shape = (int*)malloc(sizeof(int) * n_dims);
    strides = (int*)malloc(sizeof(int) * n_dims);
    memcpy(shape, org_descriptor.get_shape(), sizeof(int) * n_dims);
    memcpy(strides, org_descriptor.get_strides(), sizeof(int) * n_dims);
}

DllExport CuArrayDescriptor(){}

DllExport int get_n_dims(){ return n_dims; }
DllExport int get_n_size(){ return n_size;}
DllExport int get_ptr_offset(){ return ptr_offset; }
DllExport int* get_shape(){ return shape; }
DllExport int* get_strides(){ return strides; }
DllExport bool get_shared_status(){ return shared; }
DllExport bool get_contiguous_status(){ return iscontiguous; }

private:
    int n_dims = 0;
    int n_size = 1;
    int ptr_offset = 0;
    int* shape;
    int* strides;
    bool shared = false;
    bool iscontiguous = true; 
};

// Cuda Array
template <typename Dtype> class CuArray {
public:
DllExport CuArray(CuArrayDescriptor _descriptor) {
    descriptor = _descriptor;
    check_cuda(cudaMalloc((void**)&device_ptr, descriptor.get_n_size() * sizeof(Dtype)), 0);
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
    Dtype* ptr = (Dtype*)buf.ptr;
    n_size = 1;
    n_dims = buf.ndim;
    shape.clear();
    int* _strides = (int*)malloc(sizeof(int) * n_dims);
    int stride = 1;
    _strides[n_dims-1] = 1;
    for (int i = 0; i < n_dims; i++) {
        n_size *= buf.shape[i];
        shape.push_back(buf.shape[i]);
        if (i > 0){
            stride *= buf.shape[n_dims-i];
            _strides[n_dims-i-1] = stride;
        }
    }
    for (int i=0;i<n_dims;i++){
        strides.push_back(_strides[i]);
    }
    check_cuda(cudaMalloc((void**)&device_ptr, n_size * sizeof(Dtype)), 0);
    check_cuda(cudaMemcpy(device_ptr, ptr, n_size * sizeof(Dtype), cudaMemcpyHostToDevice), 1);
    free(_strides);
}
DllExport py::array_t<Dtype> numpy() {
    if (iscontiguous){
        int accum = 1;
        std::vector<py::ssize_t> shape_v, stride_v;
        for (int j = 0; j < n_dims; j++) {
            accum *= shape[j];
            int mult = n_size / accum;
            shape_v.push_back(shape[j]);
            stride_v.push_back(mult * sizeof(Dtype));
        }
        Dtype* ptr_cpy = (Dtype*)malloc(n_size * sizeof(Dtype));
        check_cuda(cudaMemcpy(ptr_cpy, device_ptr, n_size * sizeof(Dtype), cudaMemcpyDeviceToHost), 1);
        py::array_t<Dtype> np_arr = py::array_t<Dtype>(shape_v, stride_v, ptr_cpy); 
        return np_arr;
    }
    else{
        CuArray contiguous_arr = contiguous();
        py::array_t<Dtype> np_arr = contiguous_arr.numpy();
        contiguous_arr._free();
        return np_arr;
    }
}
DllExport void _free() {
    if (!isfree && iscontiguous && !shared) {
        cudaError_t code = cudaFree(device_ptr);
        if (code != cudaSuccess) {
            printf("cudaFree failed\n");
            printf("reason: %s\n", cudaGetErrorString(code));
        }
        else{ isfree = true;}
    }
    std::vector<int> vector_arr;
    std::vector<int> vector_arr2;
    vector_arr.swap(shape);
    vector_arr2.swap(strides);
    device_ptr = nullptr;
}
DllExport CuArray<Dtype> copy(){
    if (iscontiguous){
        CuArray arr_cpy = CuArray(n_size, n_dims, shape);
        check_cuda(cudaMemcpy(arr_cpy.get_ptr(), device_ptr, n_size * sizeof(Dtype), cudaMemcpyDeviceToDevice), 1);
        return arr_cpy;
    }
    else{ 
        return contiguous();
    }
}
DllExport CuArray<Dtype> cuarray_slices(
    py::list dims_start, py::list dims_end){
    int _n_size = 1;  // output n_size 
    int _ptr_offset = 0;  // output ptr offset
    std::vector<int> _shape;  // output shape
    std::vector<int> _strides;
    int _n_dims = 0;
    for (int i=0;i<n_dims;i++){
        int _dim = dims_end[i].cast<int>() - dims_start[i].cast<int>();
        _ptr_offset += dims_start[i].cast<int>() * strides[i];
        _n_size *= _dim;
        if (_dim != 1){
            _shape.push_back(_dim);
            _strides.push_back(strides[i]);
            _n_dims += 1;
        }
    }
    // set new array with the reference of current data
    CuArray arr_cpy = CuArray(device_ptr, _n_size, _n_dims, _shape);
    arr_cpy.set_ptr_offset(_ptr_offset);
    arr_cpy.set_contiguous_status(false);
    arr_cpy.set_strides(_strides);
    arr_cpy.set_shared_status(true);
    return arr_cpy;
}
DllExport CuArray<Dtype> view(py::list new_shape){
    // CuArray should be contiguously stored in memory
    std::vector<int> _shape;
    std::vector<int> _strides;
    int _n_dims = new_shape.size();
    int stride = 1;
    int* _stride_new = (int*)malloc(sizeof(int)*_n_dims); 
    _stride_new[_n_dims-1] = 1;
    for (int i=0;i<_n_dims;i++){
        _shape.push_back(new_shape[i].cast<int>());
        if (i>0){
            stride *= new_shape[_n_dims-i].cast<int>();
            _stride_new[_n_dims-i-1] = stride;
        }
    }
    for (int i=0;i<_n_dims;i++){
        _strides.push_back(_stride_new[i]);
    }
    free(_stride_new);
    CuArray<Dtype> new_arr = CuArray(device_ptr, n_size, _n_dims, _shape);
    new_arr.set_ptr_offset(0);
    new_arr.set_strides(_strides);
    new_arr.set_shared_status(true);
    return new_arr;
}
DllExport CuArray<Dtype> contiguous(){
    std::vector<int> _shape;
    for (int i=0;i<n_dims;i++){
        _shape.push_back(shape[i]);
    }
    int* strides_out = (int*)malloc(sizeof(int) * n_dims);
    std::vector<int> _strides;
    strides_out[n_dims-1] = 1;
    int stride = 1;
    for (int i=0;i<n_dims;i++){
        if (i > 0){
            stride *= shape[n_dims-i];
            strides_out[n_dims-i-1] = stride;
        }
    }
    for (int i=0;i<n_dims;i++){
        _strides.push_back(strides_out[i]);
    }
    CuArray arr_cpy = CuArray(n_size, n_dims, _shape);
    arr_cpy.set_ptr_offset(0);
    arr_cpy.set_strides(_strides);
    arr_cpy.set_contiguous_status(true);
    copyToContiguous(
        device_ptr, arr_cpy.get_ptr(), n_dims, ptr_offset,
        n_size, strides_out, shape.data(),
        strides.data());
    free(strides_out);
    return arr_cpy;
}
DllExport void set_values(CuArray<Dtype> arr){
    int* strides_slice = (int*)malloc(sizeof(int)*n_dims);
    strides_slice[n_dims-1] = 1;
    int stride = 1;
    for (int i=0;i<n_dims;i++){
        if (i > 0){
            stride *= shape[n_dims-i];
            strides_slice[n_dims-i-1] = stride;
        }
    }
    copyDiscontinuous(
        arr.get_ptr(), device_ptr, n_size, n_dims,
        strides_slice, shape.data(),
        arr.get_offset(), arr.get_strides().data(),
        ptr_offset, strides.data());
    free(strides_slice);
}

DllExport py::list size() {
    py::list list = py::list();
    for (int i = 0; i < n_dims; i++) {
        list.append(shape[i]);
    }
    return list;
}

// properties getter methods
DllExport Dtype* get_ptr() { return device_ptr; }
DllExport int get_n_dims() { return n_dims; }
DllExport int get_n_size() { return n_size; }
DllExport int get_offset() { return ptr_offset; }
DllExport bool is_contiguous(){ return iscontiguous; }
DllExport bool is_shared(){ return shared; }
DllExport std::vector<int> get_shape() { return shape; }
DllExport std::vector<int> get_strides() { return strides; }

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



