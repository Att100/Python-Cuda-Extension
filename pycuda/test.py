import numpy as np
import time

from pycuda import CuArray
import pycuda

__all__ = ["TestCase", "MatmulTest", "TransposeTest", "BasicTest"]

# Base Test Case
class TestCase:
    test_name = "base test case"
    def __init__(self) -> None:
        self.time_record = 0
        self.is_passed = False
    
    def start(self, enable_warmup=True, *args, **kwargs):
        data = self.prepare(*args, **kwargs)
        if enable_warmup:
            self.warmup()
        st_time = time.time()
        out = self.run(data)
        end_time = time.time()
        self.time_record = end_time - st_time
        self.is_passed = self.check(out, data)

    def show(self):
        print(
            "Test: {} => [status: {}, time: {:.5f} sec]".format(
                self.test_name, 
                "Passed" if self.is_passed else "Error",
                self.time_record))

    def clear(self):
        self.time_record = 0
        self.is_passed = False

    def prepare(self, *args, **kwargs):
        return {'data': None}

    def run(self, data):
        pass

    def check(self, out, data):
        return False

    def warmup(self):
        n = 2048
        arr1 = np.random.random((n, n))
        arr2 = np.random.random((n, n))
        arr1_cu = CuArray(np_arr=arr1, dtype="float")
        arr2_cu = CuArray(np_arr=arr2, dtype="float")
        arr3_cu = pycuda.matmul(arr1_cu, arr2_cu)


# Test Case (constructor/numpy)
class BasicTest(TestCase):
    test_name = "constructor/numpy"
    def __init__(self) -> None:
        super().__init__()

    def prepare(self):
        n = 1024
        arr1 = np.random.random((n, n)).astype('float32')
        return {"in1": arr1, "target": arr1}

    def run(self, data):
        arr1 = data["in1"]
        arr1_cu = CuArray(arr1)
        return arr1_cu.numpy()

    def check(self, out, data):
        return (data["target"]==out).all()


# Test Case (matmul)
class MatmulTest(TestCase):
    test_name = "matmul"
    def __init__(self) -> None:
        super().__init__()

    def prepare(self):
        n = 1024
        arr1 = np.random.random((n, n)).astype('float32')
        arr2 = np.random.random((n, n)).astype('float32')
        arr3 = arr1.dot(arr2)

        arr1_cu = CuArray(np_arr=arr1, dtype="float")
        arr2_cu = CuArray(np_arr=arr2, dtype="float")
        return {"in1": arr1_cu, "in2": arr2_cu, "target": arr3}

    def run(self, data):
        arr1_cu, arr2_cu = data["in1"], data["in2"]
        arr3_cu = pycuda.matmul(arr1_cu, arr2_cu)
        return arr3_cu

    def check(self, out, data):
        return np.mean((data["target"].astype("float32")-out.numpy())**2) < 1e-8


# Test Case (transpose)
class TransposeTest(TestCase):
    test_name = "transpose"
    def __init__(self) -> None:
        super().__init__()

    def prepare(self):
        axes = [1, 0, 4, 2, 3]
        arr1 = np.random.random([10, 4, 5, 6, 8]).astype('float32')

        arr1_cu = CuArray(np_arr=arr1, dtype="float")
        return {"in1": arr1_cu, "in2": axes, "target": arr1.transpose(axes)}

    def run(self, data):
        arr1_cu, axes = data["in1"], data["in2"]
        arr2_cu = pycuda.transpose(arr1_cu, axes)
        return arr2_cu

    def check(self, out, data):
        return (out.numpy()==data["target"].astype('float32')).all()