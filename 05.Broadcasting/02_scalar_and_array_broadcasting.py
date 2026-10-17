"""Scalar broadcasting."""

import numpy as np

# Scalar broadcast to any shape
a = np.array([1, 2, 3])
print(f"a + 5: {a + 5}")
print(f"a * 3: {a * 3}")
print(f"a ** 2: {a ** 2}")
print(f"10 / a: {10 / a}")

m = np.arange(6).reshape(2, 3)
print(f"\nm:\n{m}")
print(f"m + 100:\n{m + 100}")
print(f"m * 2:\n{m * 2}")
print(f"m > 2:\n{m > 2}")

# 3D
b = np.arange(24).reshape(2, 3, 4)
print(f"\n3D + scalar shape: {(b + 1).shape}")

# Scalar operations preserve dtype for int+int
print(f"\nint array + int scalar: {(a + 5).dtype}")
print(f"int array + float scalar: {(a + 5.0).dtype}")