import numpy as np
import time
from pycuda import CuArray
import pycuda


if __name__ == "__main__":
    # Speed Test of Matmul
    arr1 = np.random.random((3, 4))
    arr1_cu = CuArray(np_arr=arr1, dtype="float")\

    print(pycuda.broadcast_to(arr1_cu.view(3, 1, 4), [2, 3, 2, 4]))
    print(np.broadcast_to(arr1.reshape(3, 1, 4), [2, 3, 2, 4]))





    


    
            







