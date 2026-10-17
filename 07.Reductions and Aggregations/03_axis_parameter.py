"""Understanding the axis parameter."""

import numpy as np

# 1D: axis=0
a = np.array([1, 2, 3, 4])
print(f"1D: {a}")
print(f"  sum(axis=0) = {a.sum(axis=0)}")

# 2D: axis=0 (columns), axis=1 (rows)
m = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D:\n{m}")
print(f"  sum(axis=0) = {m.sum(axis=0)}  # operate DOWN columns")
print(f"  sum(axis=1) = {m.sum(axis=1)}  # operate ACROSS rows")
print(f"  sum() = {m.sum()}  # all elements")

# 3D: axes 0, 1, 2
b = np.arange(24).reshape(2, 3, 4)
print(f"\n3D shape: {b.shape}")
print(f"  sum(axis=0).shape: {b.sum(axis=0).shape}")
print(f"  sum(axis=1).shape: {b.sum(axis=1).shape}")
print(f"  sum(axis=2).shape: {b.sum(axis=2).shape}")
print(f"  sum(axis=(0,1)).shape: {b.sum(axis=(0, 1)).shape}")

# axis as tuple
print(f"\nmean(axis=(0, 2)).shape: {b.mean(axis=(0, 2)).shape}")

# Negative axis
print(f"\nsum(axis=-1) same as axis=2: {b.sum(axis=-1).shape}")

# keepdims
print(f"\nkeepdims=True preserves dimensions:")
print(f"  sum(axis=0) shape: {m.sum(axis=0).shape}")
print(f"  sum(axis=0, keepdims=True) shape: {m.sum(axis=0, keepdims=True).shape}")

# Practical: normalize by column means
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col_means = data.mean(axis=0)
normalized = data - col_means
print(f"\nData:\n{data}")
print(f"Column means: {col_means}")
print(f"Normalized:\n{normalized}")