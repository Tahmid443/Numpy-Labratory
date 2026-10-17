"""Broadcasting between 1D and 2D arrays."""

import numpy as np

# 2D + 1D (broadcast row-wise)
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([10, 20, 30])
print(f"Matrix:\n{m}")
print(f"Vector: {v}")
print(f"Matrix + Vector:\n{m + v}")
print("Each column j gets added v[j]")

# 2D + column vector
col = np.array([[100], [200], [300]])
print(f"\nColumn vector shape: {col.shape}")
print(f"Matrix + Column:\n{m + col}")
print("Each row i gets added col[i]")

# Outer product via broadcasting
a = np.array([1, 2, 3])
b = np.array([10, 20])
outer = a[:, np.newaxis] * b
print(f"\nOuter product of {a} and {b}:\n{outer}")
print(f"Shape: {outer.shape}")

# Row-wise normalization
data = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
row_means = data.mean(axis=1, keepdims=True)
print(f"\nRow means shape: {row_means.shape}")
normalized = data - row_means
print(f"Normalized:\n{normalized}")

# Column-wise normalization
col_means = data.mean(axis=0)
print(f"\nCol means: {col_means}")
col_norm = data - col_means
print(f"Column normalized:\n{col_norm}")