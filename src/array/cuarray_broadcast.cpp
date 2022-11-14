#include "pycuda_math.h"
#include "cuda_math.h"

template <typename Dtype> DllExport CuArray<Dtype> broadcast_to(
	CuArray<Dtype> arr, py::list shape_in, py::list shape_out, py::list mask){

	int _n_dims = shape_out.size();
	int _n_size = 1;
	int stride_in = 1;
	int stride_out = 1;
	std::vector<int> _shape;
	std::vector<int> strides_out;
	int* _mask = (int*)malloc(sizeof(int) * _n_dims);
	int* dims_in = (int*)malloc(sizeof(int) * _n_dims);
	int* dims_out;
	int* _strides_in = (int*)malloc(sizeof(int) * _n_dims);
	int* _strides_out = (int*)malloc(sizeof(int) * _n_dims);
	for (int i=0;i<_n_dims;i++){
		_n_size *= shape_out[i].cast<int>();
		_shape.push_back(shape_out[i].cast<int>());
		dims_in[i] = shape_in[i].cast<int>();
		_mask[i] = mask[i].cast<int>();
		_strides_out[_n_dims-i-1] = stride_out;
		_strides_in[_n_dims-i-1] = stride_in;
		stride_in *= shape_in[_n_dims-i-1].cast<int>();
		stride_out *= shape_out[_n_dims-i-1].cast<int>();
	}
	for (int i=0;i<_n_dims;i++){
		strides_out.push_back(_strides_out[i]);
	}
	dims_out = _shape.data();
	CuArray<Dtype> out = CuArray<Dtype>(CuArrayDescriptor(_n_dims, dims_out));
	out.set_strides(strides_out);
	cudaBroadcast(
		arr.get_ptr(), out.get_ptr(),
		_n_dims, _mask, 
		dims_in, _strides_in, 
		_n_size, dims_out, _strides_out);
	free(_mask);
	free(dims_in);
	free(_strides_in);
	free(_strides_out);
	return out;
}

template DllExport CuArray<int> broadcast_to<int>(
	CuArray<int> arr, py::list shape_in, py::list shape_out, py::list mask);

template DllExport CuArray<float> broadcast_to<float>(
	CuArray<float> arr, py::list shape_in, py::list shape_out, py::list mask);

DllExport CuArray<float> broadcastf_to(CuArray<float> arr, py::list shape_in, py::list shape_out, py::list mask){
	return broadcast_to(arr, shape_in, shape_out, mask);
}

DllExport CuArray<int> broadcasti_to(CuArray<int> arr, py::list shape_in, py::list shape_out, py::list mask){
	return broadcast_to(arr, shape_in, shape_out, mask);
}