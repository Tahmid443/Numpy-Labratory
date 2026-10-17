"""Matrix dimension compatibility rules."""

import numpy as np

# Rule: (m, n) @ (n, p) -> (m, p)
A = np.ones((2, 3))
B = np.ones((3, 4))
print(f"A{ A.shape } @ B{ B.shape } -> {(A @ B).shape}")

# Incompatible
try:
    A = np.ones((2, 3))
    C = np.ones((4, 5))
    A @ C
except ValueError as e:
    print(f"\nIncompatible: {A.shape} @ {C.shape}")
    print(f"Error: {e}")

# 1D handling: (n,) is treated as (1, n) on left, (n, 1) on right
v = np.array([1, 2, 3])
M = np.ones((3, 4))
print(f"\nv.shape={v.shape}, M.shape={M.shape}")
print(f"v @ M shape: {(v @ M).shape}  # (4,)")
print(f"M.T @ v shape: {(M.T @ v).shape}  # (4,)")

# Scalar multiplication vs matrix multiplication
A = np.ones((2, 2))
print(f"\nA * 2 (scalar): {A * 2}")
print(f"A @ 2 (invalid): trying...")
try:
    print(A @ 2)
except Exception as e:
    print(f"  Error: {e}")

# Stack of matrices
A = np.ones((3, 2, 2))
B = np.ones((2, 2))
print(f"\nA{ A.shape } @ B{ B.shape } -> {(A @ B).shape}")

# Vector dot product
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"\nv1 @ v2: {v1 @ v2}")

# Matrix-vector
M = np.array([[1, 2, 3], [4, 5, 6]])
print(f"M{M.shape} @ v{v.shape}: {M @ v}")
print(f"M @ v (matrix-vector): {(M @ v).shape}")