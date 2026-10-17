"""Sum, min, max and their variants."""

import numpy as np

a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Array: {a}")

# Basic reductions
print(f"sum: {a.sum()}")
print(f"min: {a.min()}")
print(f"max: {a.max()}")
print(f"prod: {a.prod()}")

# np functions
print(f"\nnp.sum: {np.sum(a)}")
print(f"np.min: {np.min(a)}")
print(f"np.max: {np.max(a)}")
print(f"np.amin: {np.amin(a)}")
print(f"np.amax: {np.amax(a)}")
print(f"np.ptp (peak-to-peak): {np.ptp(a)}")

# 2D reductions
m = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix:\n{m}")
print(f"m.sum() = {m.sum()}")
print(f"m.sum(axis=0) = {m.sum(axis=0)}")
print(f"m.sum(axis=1) = {m.sum(axis=1)}")
print(f"m.min(): {m.min()}")
print(f"m.max(axis=0): {m.max(axis=0)}")

# Keep dims
print(f"\nm.sum(axis=1, keepdims=True):\n{m.sum(axis=1, keepdims=True)}")

# initial parameter
print(f"\na.sum(initial=100): {a.sum(initial=100)}")

# where parameter (conditional)
print(f"a.sum(where=a > 3): {a.sum(where=a > 3)}")

# Multiple arrays
b = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(f"\nnp.maximum(a, b): {np.maximum(a, b)}")
print(f"np.fmax(a, b): {np.fmax(a, b)}")