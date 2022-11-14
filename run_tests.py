from pycuda.test import *

tests = {
    "basic": BasicTest(),
    "matmul": MatmulTest(),
    "transpose": TransposeTest()
}

def run_all_tests():
    n = len(list(tests.keys()))
    n_passed = 0
    for i, (name, test) in enumerate(tests.items()):
        test.start(i==0)
        test.show()
        if test.is_passed:
            n_passed += 1
    if n_passed == n:
        print("All tests passed!")
    else:
        print("{} tests passed, {} tests failed!".format(n_passed, n-n_passed))

if __name__ == "__main__":
    run_all_tests()
