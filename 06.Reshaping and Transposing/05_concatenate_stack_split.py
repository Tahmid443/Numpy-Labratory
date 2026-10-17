"""Concatenate, stack, and split arrays."""

import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Concatenate (1D)
print(f"np.concatenate([a, b]): {np.concatenate([a, b])}")

# 2D concatenation
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
print(f"\nm1:\n{m1}")
print(f"m2:\n{m2}")
print(f"\nconcatenate axis=0 (vertical):\n{np.concatenate([m1, m2], axis=0)}")
print(f"concatenate axis=1 (horizontal):\n{np.concatenate([m1, m2], axis=1)}")

# vstack, hstack, dstack
print(f"\nnp.vstack([m1, m2]):\n{np.vstack([m1, m2])}")
print(f"\nnp.hstack([m1, m2]):\n{np.hstack([m1, m2])}")

# stack: adds new axis
print(f"\nnp.stack([a, b]):\n{np.stack([a, b])}")
print(f"np.stack([a, b], axis=0).shape: {np.stack([a, b], axis=0).shape}")
print(f"np.stack([a, b], axis=1).shape: {np.stack([a, b], axis=1).shape}")

print(f"\nnp.vstack([a, b]):\n{np.vstack([a, b])}")
print(f"np.hstack([a, b]):\n{np.hstack([a, b])}")

# Split
c = np.arange(12)
print(f"\nc: {c}")
print(f"np.split(c, 3): {np.split(c, 3)}")
print(f"np.split(c, [2, 5, 8]): {np.split(c, [2, 5, 8])}")

# hsplit, vsplit
m = np.arange(12).reshape(3, 4)
print(f"\nm:\n{m}")
print(f"np.hsplit(m, 2):\n{np.hsplit(m, 2)}")
print(f"np.vsplit(m, 3):\n{np.vsplit(m, 3)}")

# array_split for uneven
c = np.arange(10)
print(f"\nnp.split(c, 3) raises error for 10/3")
print(f"np.array_split(c, 3): {np.array_split(c, 3)}")

# append (creates copy, slow)
print(f"\nnp.append(a, 4): {np.append(a, 4)}")
print(f"np.append(a, [4, 5]): {np.append(a, [4, 5])}")