from numpy.lib.function_base import iterable
import numpy as np
from typing import Iterable

from ..libs import _pycuda


class CuArray(object):
    """
    # pycuda.CuArray

        wrapper of cuarray c extension, support the operation of cuda which contains
        the copy of data from host to device and support the interaction with numpy 
        ndarray.

    ## Args

        np_arr: np.ndarray, numpy input array
        dtype: str, data type (support int (int32) and float (float32))

    ## Examples

        1. cuarray = CuArray(np.ones((10, 10)), dtype="float32")
        2. nparray = cuarray.numpy()  # from CuArray (GPU) to numpy array (CPU)
    """
    def __init__(self, np_arr: np.ndarray, dtype: str = "float32") -> None:
        self._cuda_arr = None
        self._dtype = dtype
        if np_arr is not None:
            if dtype == "float" or dtype == "float32":
                if np_arr.dtype != np.float32:
                    np_arr = np_arr.astype(np.float32)
                self._cuda_arr = _pycuda.CuArrayFloat(np_arr)
            elif dtype == "int" or dtype == "int32":
                if np_arr.dtype != np.int32:
                    np_arr = np_arr.astype(np.int32)
                self._cuda_arr = _pycuda.CuArrayInt(np_arr)
            else:
                raise Exception("Only Support int (int32) and float (float32) data type!")

    def __del__(self):
        self._cuda_arr.free()
        self._cuda_arr = None

    def __str__(self):
        return self.numpy().__str__()

    def __setitem__(self, index, value_arr):
        _value_arr = value_arr
        if (type(self._cuda_arr) != type(value_arr._cuda_arr)):
            _value_arr = value_arr.astype(self._dtype)
        dims_start, dims_stop = _get_slice_range(index, self.size())
        _cuda_arr = self._cuda_arr.cuarray_slices(dims_start, dims_stop)
        _cuda_arr.set_values(_value_arr._cuda_arr)
    
    def __getitem__(self, index):
        dims_start, dims_stop = _get_slice_range(index, self.size())
        # call slices method
        cuarray_new = CuArray(None, self._dtype)
        cuarray_new._cuda_arr = self._cuda_arr.cuarray_slices(dims_start, dims_stop)
        return cuarray_new
    
    def __add__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.add)
        return _cu_array_binary_op(self, x, _pycuda.addc)

    def __sub__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.sub)
        return _cu_array_binary_op(self, x, _pycuda.subc)

    def __mul__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.mul)
        return _cu_array_binary_op(self, x, _pycuda.mulc)

    def __truediv__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.div)
        return _cu_array_binary_op(self, x, _pycuda.divc)

    def __floordiv__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            mid = _cu_array_binary_op(self, x, _pycuda.div)
        else:
            mid = _cu_array_binary_op(self, x, _pycuda.divc)
        return _cu_array_unitary_op(mid, _pycuda._floor)

    def __mod__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda._mod)
        return _cu_array_binary_op(self, x, _pycuda._modc)

    def __pow__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            raise Exception("CuArray OP, CuArray Instance is not acceptable for power")
        return _cu_array_binary_op(self, x, _pycuda._pow)

    def __neg__(self):
        # Only Support Float Output
        return _cu_array_binary_op(self, -1, _pycuda.mulc)

    def __lt__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.lt)
        return _cu_array_binary_op(self, x, _pycuda.ltc)

    def __le__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.le)
        return _cu_array_binary_op(self, x, _pycuda.lec)

    def __gt__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.gt)
        return _cu_array_binary_op(self, x, _pycuda.gtc)

    def __ge__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.ge)
        return _cu_array_binary_op(self, x, _pycuda.gec)

    def __eq__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.eq)
        return _cu_array_binary_op(self, x, _pycuda.eqc)

    def __ne__(self, x):
        # Only Support Float Output
        if isinstance(x, CuArray):
            return _cu_array_binary_op(self, x, _pycuda.ne)
        return _cu_array_binary_op(self, x, _pycuda.nec)

    def view(self, *args):
        assert self.is_contiguous(), "CuArray View, View Operation Is Only Supported For contiguous \
            CuArray Object"
        org_shape = self.size()
        if iterable(args[0]):
            new_shape = list(args[0])
        else:
            new_shape = list(args)
        org_size = 1
        new_size = 1
        for i in org_shape: org_size *= i
        for i in new_shape: 
            if i > 0 and type(i)==int:
                new_size *= i  
            else: 
                raise Exception("CuArray View, Only Positive Integer Is Accepted!")
        assert org_size == new_size, "CuArray View, Input Shape Is Not Compitable With Output Shape!"
        cuarray_cpy = CuArray(None, self._dtype)
        cuarray_cpy._cuda_arr = self._cuda_arr.view(new_shape)
        return cuarray_cpy

    def reshape(self, *args):
        """
        combine the view operation with contiguous operation
        """
        if self.is_contiguous():
            return self.view(args)
        else:
            return self.contiguous().view(args)

    def is_contiguous(self) -> bool:
        """
        whether data is contiguous on cuda memory
        """
        return self._cuda_arr.is_contiguous()

    def contiguous(self):
        """
        return a contiguous copy of data on cuda memory
        """
        cuarray_cpy = CuArray(None, self._dtype)
        cuarray_cpy._cuda_arr = self._cuda_arr.contiguous()
        return cuarray_cpy

    def numpy(self) -> np.ndarray:
        """
        return the numpy ndarray (copy from gpu to cpu)
        """
        return self._cuda_arr.numpy()

    def copy(self):
        """
        make a copy of original data
        """
        cuarray_cpy = CuArray(None, self._dtype)
        cuarray_cpy._cuda_arr = self._cuda_arr.copy()
        return cuarray_cpy

    def size(self) -> list:
        """
        shape of CuArray
        """
        return self._cuda_arr.size()

    def astype(self, dtype: str = "float32"):
        """
        type cast to int32 or float32
        """
        if dtype == self._dtype:
            return self
        if dtype == "float" or dtype == "float32":
            if self._dtype == "int" or self._dtype == "int32":
                _cuda_arr = _pycuda.cast_int_to_float(self._cuda_arr)
        elif dtype == "int" or dtype == "int32":
            if self._dtype == "float" or self._dtype == "float32":
                _cuda_arr = _pycuda.cast_float_to_int(self._cuda_arr)
        else:
            raise Exception("CuArray Error: TypeCast, Only Support int (int32) And float (float32) Data Type!")
        cuarray_cpy = CuArray(None, dtype)
        cuarray_cpy._cuda_arr = _cuda_arr
        return cuarray_cpy

def _cu_array_binary_op(x: CuArray, y, func) -> CuArray:
    # Only Support Float Output
    cuarray_cpy = CuArray(None)
    x_is_float = x._dtype == 'float32' or x._dtype == "float"
    x_in = x if x_is_float else x.copy().astype('float')
    if isinstance(y, CuArray):
        y_is_float = y._dtype == 'float32' or y._dtype == "float"
        y_in = y if y_is_float else y.copy().astype('float')
        cuarray_cpy._cuda_arr = func(x_in._cuda_arr, y_in._cuda_arr)
    else:
        cuarray_cpy._cuda_arr = func(x_in._cuda_arr, y)
    return cuarray_cpy

def _cu_array_unitary_op(x: CuArray, func, param=None) -> CuArray:
    cuarray_cpy = CuArray(None)
    x_is_float = x._dtype == 'float32' or x._dtype == "float"
    x_in = x if x_is_float else x.copy().astype('float')
    if param is not None:
        cuarray_cpy._cuda_arr = func(x_in._cuda_arr, param)
    else:
        cuarray_cpy._cuda_arr = func(x_in._cuda_arr)
    return cuarray_cpy

def _get_slice_range(index, shape: list):
    n_dims = len(shape)
    index = list(index) if iterable(index) else [index]
    i = 0  # index on input slices (n_slices <= n_dims)
    dim_i = 0  # index on dims
    n_slices = 0
    n_ellipsis = 0
    dims_start = []
    dims_stop = []
    # check correctness of input slices 
    for item in index:
        if item == Ellipsis:
            n_ellipsis += 1
        elif isinstance(item, slice) or isinstance(item, int):
            n_slices += 1
        else:
            raise Exception("CuArray Slices, {} Is Not Supported For Slices!".format(type(item)))
    if n_ellipsis > 1:
        raise Exception("CuArray Slices, Accept at most one Ellipsis!")
    elif n_ellipsis < 1:
        if len(index) != n_dims:
            raise Exception(
                "CuArray Slices, Have Dims {} CuArray, But Got Dims {} Slices!".format(
                    n_dims, len(index)))
    # generate dims_start, dims_end
    while i < len(index):
        if index[i] == Ellipsis:
            for _i in range(n_dims-n_slices):
                dims_start.append(0)
                dims_stop.append(shape[dim_i])
                dim_i += 1
            i += 1
        elif isinstance(index[i], slice):
            start = index[i].start if index[i].start is not None else 0
            stop = index[i].stop if index[i].stop is not None else shape[dim_i]
            if start >= stop:
                raise Exception(
                    "CuArray Slices, Got Start Index >= Stop Index In Dims {}".format(dim_i))
            if start < 0:
                raise Exception("CuArray Slices, Got Start Index < 0 In Dims {}".format(dim_i))
            if stop > shape[dim_i]:
                raise Exception(
                    "CuArray Slices, Got Stop Index >= {} In Dims {}".format(shape[dim_i], dim_i))
            dims_start.append(start)
            dims_stop.append(stop)
            i += 1
            dim_i += 1
        elif isinstance(index[i], int):
            dims_start.append(index[i])
            # dims_stop.append(index[i]+1)
            dims_stop.append(-1)  # if dims_stop == -1, squeeze this dimension
            i += 1
            dim_i += 1
        else:
            raise Exception("CuArray Slices, {} Is Not Supported For Slices!".format(type(index[i])))
    return dims_start, dims_stop
        

