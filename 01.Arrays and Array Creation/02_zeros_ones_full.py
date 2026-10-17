"""Creating arrays filled with zeros, ones, or a specific value."""

import numpy as np

# Zeros
z1 = np.zeros(5)
z2 = np.zeros((2, 3))
z3 = np.zeros((2, 3, 2))
print(f"zeros(5): {z1}")
print(f"zeros((2,3)):\n{z2}")
print(f"zeros((2,3,2)).shape: {z3.shape}")

# Ones
o1 = np.ones(4)
o2 = np.ones((3, 2), dtype=int)
print(f"\nones(4): {o1}")
print(f"ones((3,2), dtype=int):\n{o2}")

# Full
f1 = np.full(5, 7)
f2 = np.full((2, 2), 3.14)
f3 = np.full((2, 2), "x", dtype=str)
print(f"\nfull(5, 7): {f1}")
print(f"full((2,2), 3.14):\n{f2}")
print(f"full with str:\n{f3}")

# Empty (uninitialized)
e = np.empty(3)
print(f"\nempty(3): {e}  (values are arbitrary)")

# Zeros_like, ones_like, full_like
template = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(f"\nzeros_like: {np.zeros_like(template)}")
print(f"ones_like:\n{np.ones_like(template)}")
print(f"full_like:\n{np.full_like(template, 9)}")