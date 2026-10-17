"""size, itemsize, and nbytes."""

import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array:\n{arr}")
print(f"size (total elements): {arr.size}")
print(f"itemsize (bytes per element): {arr.itemsize}")
print(f"nbytes (total bytes): {arr.nbytes}")
print(f"dtype: {arr.dtype}")

# Different dtypes
for dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
    a = np.array([1, 2, 3], dtype=dtype)
    print(f"\n{dtype.__name__}: itemsize={a.itemsize}, nbytes={a.nbytes}")

# Larger array
big = np.zeros((100, 100), dtype=np.float32)
print(f"\n100x100 float32: size={big.size}, nbytes={big.nbytes} = {big.nbytes/1024:.1f} KB")