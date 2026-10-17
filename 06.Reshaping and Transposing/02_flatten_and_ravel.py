"""Flattening arrays with ravel and flatten."""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original:\n{a}")

# ravel: returns view when possible (fast)
r = a.ravel()
print(f"\nravel(): {r}")
r[0] = 999
print(f"After modifying ravel view, original changed:\n{a}")

# flatten: always returns copy (safe)
b = np.array([[1, 2, 3], [4, 5, 6]])
f = b.flatten()
print(f"\nflatten(): {f}")
f[0] = 999
print(f"After modifying flatten copy, original unchanged:\n{b}")

# order parameter
c = np.array([[1, 2], [3, 4]])
print(f"\nC-order (row): {c.ravel()}")
print(f"F-order (col): {c.ravel(order='F')}")

# Memory comparison
big = np.zeros((1000, 1000))
import time

start = time.time()
for _ in range(1000):
    big.ravel()
print(f"\n1000x ravel: {time.time() - start:.4f}s")

start = time.time()
for _ in range(1000):
    big.flatten()
print(f"1000x flatten: {time.time() - start:.4f}s")

# 3D ravel
d = np.arange(24).reshape(2, 3, 4)
print(f"\n3D ravel shape: {d.ravel().shape}")
print(f"3D ravel: {d.ravel()}")

# ravel on slice may copy
sl = a[::2]
r_sl = sl.ravel()
print(f"\nSlice ravel: {r_sl}")