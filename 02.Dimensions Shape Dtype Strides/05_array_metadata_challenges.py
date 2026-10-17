"""Challenges: array metadata."""

import numpy as np

def inspect(arr):
    return {
        'shape': arr.shape,
        'ndim': arr.ndim,
        'size': arr.size,
        'dtype': str(arr.dtype),
        'itemsize': arr.itemsize,
        'nbytes': arr.nbytes,
        'strides': arr.strides,
        'C_CONTIGUOUS': arr.flags['C_CONTIGUOUS'],
    }

if __name__ == "__main__":
    test_arrays = [
        np.array(5),
        np.arange(10),
        np.zeros((3, 4)),
        np.ones((2, 3, 4), dtype=np.float32),
        np.array([[1, 2], [3, 4]], order='F'),
    ]
    for arr in test_arrays:
        print(f"Array:\n{arr}")
        for k, v in inspect(arr).items():
            print(f"  {k}: {v}")
        print()