#include "pycuda_math.h"

CuArray<float> FnBinary(CuArray<float> x, CuArray<float> y, FtypeBinary fn){
    CuArray<float> _x, _y;
    CuArray<float> z = CuArray<float>(CuArrayDescriptor(
        x.get_descriptor().n_dims, x.get_descriptor().shape));
    _x = x.is_contiguous() ? x : x.contiguous();
    _y = y.is_contiguous() ? y : y.contiguous();
    fn(_x.get_ptr(), _y.get_ptr(), z.get_ptr(), x.get_descriptor().n_size);
    if (!x.is_contiguous()){
        _x._free();
    }
    if (!y.is_contiguous()){
        _y._free();
    }
    return z;
}

CuArray<float> FnBinaryC(CuArray<float> x, float y, FtypeBinaryC fn){
    CuArray<float> _x;
    CuArray<float> z = CuArray<float>(CuArrayDescriptor(
        x.get_descriptor().n_dims, x.get_descriptor().shape));
    _x = x.is_contiguous() ? x : x.contiguous();
    fn(_x.get_ptr(), y, z.get_ptr(), x.get_descriptor().n_size);
    if (!x.is_contiguous()){
        _x._free();
    }
    return z;
}

CuArray<float> FnUnitary(CuArray<float> x, FtypeUnitary fn){
    CuArray<float> y;
    if (x.is_contiguous()){
        y = CuArray<float>(CuArrayDescriptor(
            x.get_descriptor().n_dims, x.get_descriptor().shape));
        fn(x.get_ptr(), y.get_ptr(), x.get_descriptor().n_size);
    }
    else {
        y = x.contiguous();
        fn(y.get_ptr(), y.get_ptr(), x.get_descriptor().n_size);
    }
    return y;
}

DllExport CuArray<float> _cuda_add(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_add);}
DllExport CuArray<float> _cuda_sub(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_sub);}
DllExport CuArray<float> _cuda_mul(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_mul);}
DllExport CuArray<float> _cuda_div(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_div);}
DllExport CuArray<float> _cuda_lt(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_lt);}
DllExport CuArray<float> _cuda_le(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_le);}
DllExport CuArray<float> _cuda_gt(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_gt);}
DllExport CuArray<float> _cuda_ge(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_ge);}
DllExport CuArray<float> _cuda_eq(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_eq);}
DllExport CuArray<float> _cuda_ne(CuArray<float> x, CuArray<float> y){return FnBinary(x,y,cuda_ne);}

DllExport CuArray<float> _cuda_addc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_addc);}
DllExport CuArray<float> _cuda_subc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_subc);}
DllExport CuArray<float> _cuda_mulc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_mulc);}
DllExport CuArray<float> _cuda_divc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_divc);}
DllExport CuArray<float> _cuda_ltc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_ltc);}
DllExport CuArray<float> _cuda_lec(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_lec);}
DllExport CuArray<float> _cuda_gtc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_gtc);}
DllExport CuArray<float> _cuda_gec(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_gec);}
DllExport CuArray<float> _cuda_eqc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_eqc);}
DllExport CuArray<float> _cuda_nec(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_nec);}
DllExport CuArray<float> _cuda_modc(CuArray<float> x, float y){return FnBinaryC(x,y,cuda_modc);}

DllExport CuArray<float> _cuda_floor(CuArray<float> x){return FnUnitary(x,cuda_floor);}
DllExport CuArray<float> _cuda_ceil(CuArray<float> x){return FnUnitary(x,cuda_ceil);}
DllExport CuArray<float> _cuda_mod(CuArray<float> x, CuArray<float> y){return FnBinary(x, y, cuda_mod);}
DllExport CuArray<float> _cuda_round(CuArray<float> x){return FnUnitary(x,cuda_round);}
DllExport CuArray<float> _cuda_log(CuArray<float> x){return FnUnitary(x,cuda_log);}
DllExport CuArray<float> _cuda_log2(CuArray<float> x){return FnUnitary(x,cuda_log2);}
DllExport CuArray<float> _cuda_log10(CuArray<float> x){return FnUnitary(x,cuda_log10);}
DllExport CuArray<float> _cuda_exp(CuArray<float> x){return FnUnitary(x,cuda_exp);}
DllExport CuArray<float> _cuda_exp2(CuArray<float> x){return FnUnitary(x,cuda_exp2);}
DllExport CuArray<float> _cuda_exp10(CuArray<float> x){return FnUnitary(x,cuda_exp10);}
DllExport CuArray<float> _cuda_pow(CuArray<float> x, float p){
    CuArray<float> y;
    if (x.is_contiguous()){
        y = CuArray<float>(CuArrayDescriptor(
            x.get_descriptor().n_dims, x.get_descriptor().shape));
        cuda_pow(x.get_ptr(), y.get_ptr(), p, x.get_descriptor().n_size);
    }
    else {
        y = x.contiguous();
        cuda_pow(y.get_ptr(), y.get_ptr(), p, x.get_descriptor().n_size);
    }
    return y;
}
DllExport CuArray<float> _cuda_sqrt(CuArray<float> x){return FnUnitary(x,cuda_sqrt);}