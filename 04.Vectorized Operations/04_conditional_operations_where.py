"""Conditional operations with np.where."""

import numpy as np

a = np.array([1, 5, 3, 8, 2, 9, 4])
print(f"Array: {a}")

# np.where as conditional selector
result = np.where(a > 4, a, 0)
print(f"np.where(a > 4, a, 0): {result}")

# np.where with scalar values
print(f"np.where(a > 4, 1, 0): {np.where(a > 4, 1, 0)}")
print(f"np.where(a > 4, 'big', 'small'): {np.where(a > 4, 'big', 'small')}")

# np.where as index finder (1 arg)
print(f"np.where(a > 4): {np.where(a > 4)}")

# 2D example
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{b}")
print(f"np.where(b > 5, b, 0):\n{np.where(b > 5, b, 0)}")
print(f"np.where(b % 2 == 0, b, -1):\n{np.where(b % 2 == 0, b, -1)}")

# Multiple conditions with nested where
c = a.copy()
result = np.where(c < 3, 'low',
         np.where(c < 6, 'medium', 'high'))
print(f"\nNested where: {result}")

# np.where with broadcasting
row = np.array([1, 2, 3])
col = np.array([[10], [20], [30]])
result = np.where(col > 15, row, 0)
print(f"\nBroadcast where:\n{result}")

# Performance benefit over loops
def python_where(arr, threshold):
    return [x if x > threshold else 0 for x in arr]

import time
big = np.random.default_rng(42).integers(0, 100, 1_000_000)

start = time.time()
_ = python_where(big, 50)
t1 = time.time() - start

start = time.time()
_ = np.where(big > 50, big, 0)
t2 = time.time() - start

print(f"\nPython loop: {t1:.3f}s")
print(f"np.where:    {t2:.3f}s")
print(f"Speedup:     {t1/t2:.0f}x")