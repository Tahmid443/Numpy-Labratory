"""Boolean indexing (masking)."""

import numpy as np

a = np.array([1, 5, 2, 8, 3, 9, 4])
print(f"Array: {a}")

# Create mask
mask = a > 4
print(f"Mask (a > 4): {mask}")

# Apply mask
print(f"a[mask]: {a[mask]}")
print(f"a[a > 4]: {a[a > 4]}")

# Combine conditions
print(f"a[(a > 2) & (a < 8)]: {a[(a > 2) & (a < 8)]}")
print(f"a[(a < 3) | (a > 8)]: {a[(a < 3) | (a > 8)]}")
print(f"a[~(a > 4)]: {a[~(a > 4)]}")

# 2D boolean indexing
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{b}")
print(f"b[b > 5]: {b[b > 5]}")
print(f"b[b % 2 == 0]: {b[b % 2 == 0]}")

# Boolean assignment
c = a.copy()
c[c > 4] = 0
print(f"\nSet values > 4 to 0: {c}")

# np.where
print(f"np.where(a > 4, 100, -1): {np.where(a > 4, 100, -1)}")

# Counting
print(f"Count > 4: {(a > 4).sum()}")
print(f"Any > 8: {(a > 8).any()}")
print(f"All > 0: {(a > 0).all()}")

# Index positions
print(f"Indices where a > 4: {np.where(a > 4)[0]}")