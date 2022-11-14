#include "pycuda_math.h"
#include "cuda_math.h"

template <typename Dtype> DllExport CuArray<Dtype> transpose(
	CuArray<Dtype> arr, py::list shape_in, py::list shape_out, py::list axes){

	int _n_dims = shape_out.size();
	int _n_size = 1;
	int stride_in = 1;
	int stride_out = 1;
	std::vector<int> _shape;
	std::vector<int> strides_out;
	int* _strides_in = (int*)malloc(sizeof(int) * _n_dims);
	int* _strides_out = (int*)malloc(sizeof(int) * _n_dims);
	int* _dims_in = (int*)malloc(sizeof(int) * _n_dims);
    int* _axes = (int*)malloc(sizeof(int) * _n_dims);
	for (int i=0;i<_n_dims;i++){
		_n_size *= shape_out[i].cast<int>();
		_shape.push_back(shape_out[i].cast<int>());
		_strides_out[_n_dims-1-i] = stride_out;
		_strides_in[_n_dims-1-i] = stride_in;
		stride_in *= shape_in[_n_dims-1-i].cast<int>();
		stride_out *= shape_out[_n_dims-1-i].cast<int>();
        _axes[i] = axes[i].cast<int>();
		_dims_in[i] = shape_in[i].cast<int>();
	}
	for (int i=0;i<_n_dims;i++){
		strides_out.push_back(_strides_out[i]);
	}
	int* dims_out = _shape.data();
	CuArray<Dtype> out = CuArray<Dtype>(CuArrayDescriptor(_n_dims, dims_out));
	out.set_strides(strides_out);
	cudaTranspose(
		arr.get_ptr(), out.get_ptr(),
		_n_dims, _n_size, _axes,
		_strides_in, _strides_out,
		_dims_in);
	free(_strides_in);
	free(_strides_out);
    free(_axes);
	return out;
}

template DllExport CuArray<int> transpose<int>(
	CuArray<int> arr, py::list shape_in, py::list shape_out, py::list axes);

template DllExport CuArray<float> transpose<float>(
	CuArray<float> arr, py::list shape_in, py::list shape_out, py::list axes);

DllExport CuArray<float> transposef(CuArray<float> arr, py::list shape_in, py::list shape_out, py::list axes){
	return transpose(arr, shape_in, shape_out, axes);
}

DllExport CuArray<int> transposei(CuArray<int> arr, py::list shape_in, py::list shape_out, py::list axes){
	return transpose(arr, shape_in, shape_out, axes);
}

