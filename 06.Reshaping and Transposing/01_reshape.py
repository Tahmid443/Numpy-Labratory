"""Reshaping arrays."""

import numpy as np

# 1D to 2D
a = np.arange(12)
print(f"Original: {a}")
b = a.reshape(3, 4)
print(f"reshape(3, 4):\n{b}")

# 2D to 3D
c = a.reshape(2, 3, 2)
print(f"\nreshape(2, 3, 2):\n{c}")

# Use -1 to infer dimension
d = a.reshape(4, -1)
print(f"\nreshape(4, -1) infers: {d.shape}")
print(d)

e = a.reshape(-1, 6)
print(f"\nreshape(-1, 6): {e.shape}")

# Flatten to 1D
print(f"\nreshape(-1): {b.reshape(-1)}")
print(f"flatten(): {b.flatten()}")

# reshape returns a view when possible
f = np.arange(6)
g = f.reshape(2, 3)
g[0, 0] = 99
print(f"\nAfter modifying reshaped: {f}")

# If view not possible, returns copy
h = np.arange(10)
h2 = h[::2]  # non-contiguous
try:
    h2.reshape(5, 1)
    print("reshape on non-contiguous works (may copy)")
except Exception as e:
    print(f"Error: {e}")

# order='F' for column-major reshape
print(f"\nreshape((3,4), order='F'):\n{a.reshape(3, 4, order='F')}")

# Practice: reshape with -1
arr = np.arange(24)
print(f"\nreshape(2, -1): {arr.reshape(2, -1).shape}")
print(f"reshape(-1, 2): {arr.reshape(-1, 2).shape}")
print(f"reshape(2, 3, -1): {arr.reshape(2, 3, -1).shape}")