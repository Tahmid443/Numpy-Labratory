"""Transposing arrays."""

import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array (2x3):\n{a}")
print(f"shape: {a.shape}")

# Transpose
print(f"\na.T (3x2):\n{a.T}")
print(f"a.T.shape: {a.T.shape}")

# np.transpose (same as .T for 2D)
print(f"\nnp.transpose(a):\n{np.transpose(a)}")

# Transpose returns a view
b = a.T
b[0, 0] = 99
print(f"\nAfter modifying transpose view:\n{a}")

# 3D transpose
c = np.arange(24).reshape(2, 3, 4)
print(f"\n3D shape: {c.shape}")
print(f"c.T shape: {c.T.shape}")
print(f"c.T:\n{c.T}")

# np.transpose with axes permutation
d = c.transpose(2, 0, 1)
print(f"\ntranspose(2, 0, 1) shape: {d.shape}")

# swapaxes
e = c.swapaxes(0, 2)
print(f"\nswapaxes(0, 2) shape: {e.shape}")

# moveaxis
f = np.moveaxis(c, 0, -1)
print(f"\nmoveaxis(0, -1) shape: {f.shape}")

# Practical: transpose for matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\nA @ B:\n{A @ B}")
print(f"A @ B.T:\n{A @ B.T}")

# Symmetric check
S = np.array([[1, 2], [2, 1]])
print(f"\nS is symmetric: {np.array_equal(S, S.T)}")