"""Strides and memory layout (C order vs F order)."""

import numpy as np

# C-order (row-major) is default
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(f"C-order array:\n{arr_c}")
print(f"  strides: {arr_c.strides}")
print(f"  flags C_CONTIGUOUS: {arr_c.flags['C_CONTIGUOUS']}")
print(f"  flags F_CONTIGUOUS: {arr_c.flags['F_CONTIGUOUS']}")

# F-order (column-major)
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f"\nF-order array:\n{arr_f}")
print(f"  strides: {arr_f.strides}")
print(f"  flags C_CONTIGUOUS: {arr_f.flags['C_CONTIGUOUS']}")
print(f"  flags F_CONTIGUOUS: {arr_f.flags['F_CONTIGUOUS']}")

# Strides explain element jump per dimension
a = np.arange(12).reshape(3, 4)
print(f"\nArray:\n{a}")
print(f"  shape: {a.shape}")
print(f"  strides: {a.strides}")
print(f"  Row stride: {a.strides[0]} bytes (skip to next row)")
print(f"  Col stride: {a.strides[1]} bytes (skip to next col)")
print(f"  itemsize: {a.itemsize}")

# Verify with flat memory
print(f"\nFlattened: {a.ravel()}")
print(f"A[0,0]={a[0,0]}, A[0,1]={a[0,1]} (col stride = {a.strides[1]} = 1 element)")
print(f"A[1,0]={a[1,0]} (row stride = {a.strides[0]} = {a.shape[1]} elements)")

# Transpose changes strides
at = a.T
print(f"\nTransposed:\n{at}")
print(f"  shape: {at.shape}, strides: {at.strides}")