"""Slicing multi-dimensional arrays."""

import numpy as np

a = np.arange(1, 13).reshape(3, 4)
print(f"Array:\n{a}")

print(f"\na[0:2, 1:3]:\n{a[0:2, 1:3]}")
print(f"a[::2, ::2]:\n{a[::2, ::2]}")
print(f"a[:, 1]:\n{a[:, 1]}")
print(f"a[1, :]:\n{a[1, :]}")
print(f"a[:, -1]:\n{a[:, -1]}")
print(f"a[::-1, :]:\n{a[::-1, :]}")
print(f"a[::-1, ::-1]:\n{a[::-1, ::-1]}")

# 3D slicing
b = np.arange(24).reshape(2, 3, 4)
print(f"\n3D array shape {b.shape}:\n{b}")
print(f"\nb[0]:\n{b[0]}")
print(f"b[:, 0]:\n{b[:, 0]}")
print(f"b[:, :, 1]:\n{b[:, :, 1]}")
print(f"b[1, 1:, :]:\n{b[1, 1:, :]}")

# Modifying via slicing
c = np.zeros((3, 3))
c[1, :] = 5
c[:, 2] = 7
print(f"\nModified:\n{c}")