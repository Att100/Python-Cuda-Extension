from ..libs import _pycuda
from .cuarray_wrapper import CuArray

__all__ = ['floor', 'ceil', 'round', 'exp', 'exp2', 'exp10', 'log', 'log2', 'log10', 
    'pow', 'sqrt', 'matmul']

def type_check(x: CuArray) -> CuArray:
    if x._dtype == "float" or x._dtype == "float32":
        return x
    else:
        return x.copy().astype('float32')

def floor(x: CuArray):
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._floor(checked_x._cuda_arr)
    return y

def ceil(x: CuArray):
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._ceil(checked_x._cuda_arr)
    return y

def round(x: CuArray):
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._round(checked_x._cuda_arr)
    return y

def exp(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._exp(checked_x._cuda_arr)
    return y

def exp2(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._exp2(checked_x._cuda_arr)
    return y

def exp10(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._exp10(checked_x._cuda_arr)
    return y

def log(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._log(checked_x._cuda_arr)
    return y

def log2(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._log2(checked_x._cuda_arr)
    return y

def log10(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._log10(checked_x._cuda_arr)
    return y

def pow(x: CuArray, p: float): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._pow(checked_x._cuda_arr, p)
    if float(p) == float(int(p)) and p > 0 and 'int' in x._dtype:
        return y.astype('int')
    return y

def sqrt(x: CuArray): 
    y = CuArray(None)
    checked_x = type_check(x)
    y._cuda_arr = _pycuda._sqrt(checked_x._cuda_arr)
    return y

def matmul(x: CuArray, y: CuArray):
    assert isinstance(x, CuArray) and isinstance(y, CuArray), "CuArray Error, x and y should be both CuArray"
    assert len(x.size())==2 and len(y.size())==2, "CuArray Error, x and y should be both in len(shape)==2"
    assert x.size()[1] == y.size()[0], "CuArray Error, the second dim of x should equal to the first dimension of y"
    x_is_float = x._dtype == 'float32' or x._dtype == "float"
    y_is_float = y._dtype == 'float32' or y._dtype == "float"
    back_to_int = (not x_is_float) and (not y_is_float)
    x, y = type_check(x), type_check(y)
    cuarray_cpy = CuArray(None)
    cuarray_cpy._cuda_arr = _pycuda.matmulf(x._cuda_arr, y._cuda_arr)
    if back_to_int:
        return cuarray_cpy.astype('int')
    return cuarray_cpy

