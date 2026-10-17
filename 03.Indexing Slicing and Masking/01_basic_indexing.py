"""Basic indexing in 1D and 2D arrays."""

import numpy as np

# 1D indexing
a = np.array([10, 20, 30, 40, 50])
print(f"Array: {a}")
print(f"a[0] = {a[0]}")
print(f"a[2] = {a[2]}")
print(f"a[-1] = {a[-1]}")
print(f"a[-2] = {a[-2]}")

# Modify
a[1] = 99
print(f"After a[1] = 99: {a}")

# 2D indexing
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{b}")
print(f"b[0, 0] = {b[0, 0]}")
print(f"b[1, 2] = {b[1, 2]}")
print(f"b[2, -1] = {b[2, -1]}")
print(f"b[1][1] = {b[1][1]}  (chained, less efficient)")

# Whole row / column
print(f"\nb[0] (first row): {b[0]}")
print(f"b[:, 1] (second col): {b[:, 1]}")
print(f"b[-1, :] (last row): {b[-1, :]}")

# 3D indexing
c = np.arange(24).reshape(2, 3, 4)
print(f"\n3D array shape: {c.shape}")
print(f"c[0] (first 2D slice):\n{c[0]}")
print(f"c[0, 1, 2] = {c[0, 1, 2]}")
print(f"c[1, -1, :] = {c[1, -1, :]}")