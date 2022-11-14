# Python Cuda Extension API Documentation

* [1. pycuda](#pycuda)
  - [1.1 basic mathmatics functions](#basic-mathmatics-functions)
    + [pycuda.floor](#pycuda.floor)
    + [pycuda.ceil](#pycuda.ceil)
    + [pycuda.round](#pycuda.round)
    + [pycuda.exp](#pycuda.exp)
    + [pycuda.exp2](#pycuda.exp2)
    + [pycuda.exp10](#pycuda.exp10)
    + [pycuda.log](#pycuda.log)
    + [pycuda.log2](#pycuda.log2)
    + [pycuda.log10](#pycuda.log10)
    + [pycuda.pow](#pycuda.pow)
    + [pycuda.sqrt](#pycuda.sqrt)
  - [1.2 assistant functions](#assistant-functions)
    + [pycuda.broadcast_to](#pycuda.broadcast_to)
* [2. pycuda.CuArray](#pycuda.CuArray)

<h2 id="pycuda"> 1. pycuda </h2>

<h3 id="basic-mathmatics-functions"> 1.1 basic mathmatics functions </h3>

* <h4 id="pycuda.floor"><font color=#FF4500>pycuda.floor</font></h4>

    Calculate the largest integer less than or equal to x.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(3, 8, size=(2, 3)))
        outp = pycuda.floor(inp)

        inp:
            [[4.9694386 3.692315  4.128368 ]
             [6.405143  7.715407  7.168354 ]]

        outp: 
            [[4. 3. 4.]
             [6. 7. 7.]]
        ```

* <h4 id="pycuda.ceil"><font color=#FF4500>pycuda.ceil</font></h4>

    Compute the smallest integer value not less than x.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(3, 8, size=(2, 3)))
        outp = pycuda.ceil(inp)

        inp:
            [[6.4372826 6.7513876 7.2809124]
             [3.0020728 7.907349  4.2184167]]

        outp: 
            [[7. 7. 8.]
             [4. 8. 5.]]
        ```

* <h4 id="pycuda.round"><font color=#FF4500>pycuda.round</font></h4>

    Round to nearest integer value in floating-point.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(3, 8, size=(2, 3)))
        outp = pycuda.round(inp)

        inp:
            [[4.282896  5.0667205 4.5214834]
             [7.662672  5.0775805 7.065243 ]]

        outp: 
            [[4. 5. 5.]
             [8. 5. 7.]]
        ```

* <h4 id="pycuda.exp"><font color=#FF4500>pycuda.exp</font></h4>

    Calculate the e base exponential of the input argument x.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(3, 8, size=(2, 3)))
        outp = pycuda.exp(inp)

        inp:
            [[7.771955  6.6278324 3.009553 ]
             [4.307004  3.1554427 6.1338243]]

        outp: 
            [[2373.1062    755.84204    20.278334]
             [  74.217804   23.463425  461.19656 ]]
        ```

* <h4 id="pycuda.exp2"><font color=#FF4500>pycuda.exp2</font></h4>

    Calculate the 2 base exponential of the input argument x.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(3, 8, size=(2, 3)))
        outp = pycuda.exp2(inp)

        inp:
            [[3.5498948 4.3330393 6.924489 ]
             [6.839512  6.074056  6.1402364]]

        outp: 
            [[ 11.711832  20.154629 121.47276 ]
             [114.52446   67.37102   70.533485]]
        ```

* <h4 id="pycuda.exp10"><font color=#FF4500>pycuda.exp10</font></h4>

    Calculate the 10 base exponential of the input argument x.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 3, size=(2, 3)))
        outp = pycuda.exp10(inp)

        inp:
            [[1.8646079 2.808665  1.413951 ]
             [2.3203433 1.7365348 2.7144063]]

        outp: 
            [[ 73.21633  643.6726    25.93887 ]
             [209.0948    54.517365 518.09125 ]]
        ```

* <h4 id="pycuda.log"><font color=#FF4500>pycuda.log</font></h4>

    Calculate the natural logarithm of the input argument.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 3, size=(2, 3)))
        outp = pycuda.log(inp)

        inp:
            [[2.5424805 2.8960075 2.0681388]
             [2.5941432 2.4453204 1.5023806]]

        outp: 
            [[0.93314016 1.063333   0.7266491 ]
             [0.95325625 0.8941761  0.40705094]]
        ```

* <h4 id="pycuda.log2"><font color=#FF4500>pycuda.log2</font></h4>

    Calculate the base 2 logarithm of the input argument.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 3, size=(2, 3)))
        outp = pycuda.log2(inp)

        inp:
            [[2.690697  2.826743  1.9656531]
             [2.1776628 1.5496598 1.8306476]]

        outp: 
            [[1.42798   1.4991407 0.9750087]
             [1.1227806 0.6319516 0.8723541]]
        ```

* <h4 id="pycuda.log10"><font color=#FF4500>pycuda.log10</font></h4>

    Calculate the base 10 logarithm of the input argument.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 3, size=(2, 3)))
        outp = pycuda.log10(inp)

        inp:
            [[1.500737  1.4493077 2.2137134]
             [1.408106  2.9620194 2.1534207]]

        outp: 
            [[0.17630458 0.16116059 0.34512138]
             [0.14863534 0.4715879  0.3331289 ]]
        ```

* <h4 id="pycuda.pow"><font color=#FF4500>pycuda.pow</font></h4>

    Calculate the value of first argument to the power of second argument.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |
        | p | int / float | - | power |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 3, size=(2, 3)))
        outp = pycuda.pow(inp, 2)

        inp:
            [[1.3033447 2.9068935 1.9962732]
             [1.5107776 1.8359412 1.3731227]]

        outp: 
            [[1.6987076 8.450029  3.9851067]
             [2.282449  3.37068   1.885466 ]]
        ```
    
* <h4 id="pycuda.sqrt"><font color=#FF4500>pycuda.sqrt</font></h4>

    Calculate the square root of the input argument.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 | output array |

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 3, size=(2, 3)))
        outp = pycuda.sqrt(inp)

        inp:
            [[2.9292395 2.5749023 1.7448848]
             [2.8064773 1.9148637 1.1952676]]

        outp: 
            [[1.7115021 1.6046503 1.3209409]
             [1.6752543 1.383786  1.0932829]]
        ```

<h3 id="assistant-functions"> 1.2 assistant functions </h3>

* <h4 id="pycuda.broadcast_to"><font color=#FF4500>pycuda.broadcast_to</font></h4>

    Broadcast the input array to specified shape.

    + ##### Args

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | x | CuArray | float32 / int32 | input array |
        | shape | list | - | output shape |

    + ##### Return

        | name | type | dtype | description |
        | -- | -- | -- | -- |
        | y | CuArray | float32 / int32 | output array |

    + ##### Broadcast Principles

        * xxx
        * xxx
        * xxx

    + ##### Example

        ```python
        import pycuda
        import numpy as np
        from pycuda import CuArray
        inp = CuArray(np.random.uniform(1, 6, size=(3, 1)))
        outp = pycuda.broadcast_to(inp, [3, 2])

        inp:
            [[3.3116004]
             [3.9600136]
             [5.822823 ]]

        outp: 
            [[3.3116004 3.3116004]
             [3.9600136 3.9600136]
             [5.822823  5.822823 ]]
        ```

<h2 id="pycuda.CuArray"> 2. pycuda.CuArray </h2>



