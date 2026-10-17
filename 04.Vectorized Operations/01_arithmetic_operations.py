"""Arithmetic operations on arrays."""

import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a = {a}")
print(f"b = {b}")

# Element-wise operations
print(f"\na + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a // 2 = {a // 2}")
print(f"a % 3 = {a % 3}")
print(f"a ** 2 = {a ** 2}")
print(f"2 ** a = {2 ** a}")
print(f"-a = {-a}")

# In-place operations
c = a.copy()
c += 10
print(f"\nAfter c += 10: {c}")

d = a.copy()
d *= 2
print(f"After d *= 2: {d}")

# With scalars
print(f"\na + 100 = {a + 100}")
print(f"a * 10 = {a * 10}")
print(f"a / 2 = {a / 2}")

# 2D operations
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[10, 20], [30, 40]])
print(f"\nm1 + m2:\n{m1 + m2}")
print(f"m1 * m2 (element-wise):\n{m1 * m2}")

# Integer division vs float
print(f"\n7 / 2 = {7 / 2} (float)")
print(f"7 // 2 = {7 // 2} (int)")