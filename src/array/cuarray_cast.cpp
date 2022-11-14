#include "pycuda_math.h"

DllExport CuArray<float> cast_Int_to_Float(CuArray<int> int_arr) {
	CuArray<float> float_arr = CuArray<float>(CuArrayDescriptor(
        int_arr.get_descriptor().n_dims, int_arr.get_descriptor().shape));
	CuArray<int> _int_arr = int_arr;
	if (!int_arr.is_contiguous()){
		_int_arr = int_arr.contiguous();
	}
	castIntToFloat(_int_arr.get_ptr(), float_arr.get_ptr(), int_arr.get_descriptor().n_size);
	if (!int_arr.is_contiguous()){
		_int_arr._free();
	}
	return float_arr;
}

DllExport CuArray<int> cast_Float_to_Int(CuArray<float> float_arr) {
	CuArray<int> int_arr = CuArray<int>(CuArrayDescriptor(
        float_arr.get_descriptor().n_dims, float_arr.get_descriptor().shape));
	CuArray<float> _float_arr = float_arr;
	if (!float_arr.is_contiguous()){
		_float_arr = float_arr.contiguous();
	}
	castFloatToInt(_float_arr.get_ptr(), int_arr.get_ptr(), _float_arr.get_descriptor().n_size);
	if (!float_arr.is_contiguous()){
		_float_arr._free();
	}
	return int_arr;
}