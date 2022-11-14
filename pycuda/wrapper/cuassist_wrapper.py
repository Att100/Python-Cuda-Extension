from typing import Iterable
from ..libs import _pycuda
from .cuarray_wrapper import CuArray

__all__ = ["broadcast_to", "transpose"]

def broadcast_to(arr: CuArray, shape: Iterable):
    assert isinstance(shape, Iterable), "CuArray Error, shape should be an instance of Iterable"
    shape = list(shape)
    in_shape = [1 for i in range(len(shape)-len(arr.size()))] + arr.size()
    mask = [0 if dim==1 else 1 for dim in in_shape]
    new_arr = CuArray(None, arr._dtype)
    if arr._dtype == "float32" or arr._dtype == "float":
        new_arr._cuda_arr = _pycuda.broadcastf_to(arr._cuda_arr, in_shape, shape, mask)
    elif arr._dtype == "int32" or arr._dtype == "int":
        new_arr._cuda_arr = _pycuda.broadcasti_to(arr._cuda_arr, in_shape, shape, mask)
    return new_arr

def transpose(arr: CuArray, axes: Iterable):
    in_shape = arr.size()
    assert isinstance(axes , Iterable), "CuArray Error, axes should be an instance of Iterable"
    assert len(in_shape)==len(axes), "CuArray Error, length of axes should be equal to number of dims of arr"
    flag = True
    for ax in axes:
        if ax >= len(in_shape) or ax < 0:
            raise Exception("CuArray Error, all item of axes should in range of [0, n_dim)")
    out_shape = [in_shape[i] for i in axes]
    new_arr = CuArray(None, arr._dtype)
    if arr._dtype == "float32" or arr._dtype == "float":
        new_arr._cuda_arr = _pycuda.transposef(arr._cuda_arr, in_shape, out_shape, list(axes))
    elif arr._dtype == "int32" or arr._dtype == "int":
        new_arr._cuda_arr = _pycuda.transposei(arr._cuda_arr, in_shape, out_shape, list(axes))
    return new_arr