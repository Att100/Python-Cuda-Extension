# Python-Cuda-Extension

## 1. Introduction

This is a python-cuda extension which aim to provide tensor acceleration. This extension now support a number of basic operations and more functions will be added to this project in the future.

Please remember that THIS PROJECT IS COMPLETELY DIFFERENT FROM THE NVIDIA PYCUDA LIBRARY.

## 2. Build Python-Cuda Extension

### 2.1 Environment

Please make sure that your cuda, cmake, python (>=3.9), and numpy have been installed correctly before running the scripts below.

### 2.2 Build on Windows

```
git clone https://github.com/Att100/Python-Cuda-Extension.git
cd Python-Cuda-Extension
git clone https://github.com/pybind/pybind11.git
./scripts/build_pycuda_win.cmd
```

### 2.3 Build on Linux

```
git clone https://github.com/Att100/Python-Cuda-Extension.git
cd Python-Cuda-Extension
git clone https://github.com/pybind/pybind11.git
sh ./scripts/build_pycuda_win.sh
```

### 2.4 Run Tests

```
python run_tests.py

# output -->
Test: constructor/numpy => [status: Passed, time: 0.00441 sec]
Test: matmul => [status: Passed, time: 0.00103 sec]
Test: transpose => [status: Passed, time: 0.00064 sec]
All tests passed!
```

## 3. Usage

Please reference the documents in [doc](./doc/index.md) for guidence.

## References

- [pybind11](https://github.com/pybind/pybind11)