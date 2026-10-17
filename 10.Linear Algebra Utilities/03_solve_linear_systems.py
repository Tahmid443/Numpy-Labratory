"""Solving linear systems."""

import numpy as np

# Solve Ax = b
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 7])

print(f"A:\n{A}")
print(f"b: {b}")

x = np.linalg.solve(A, b)
print(f"\nSolution x: {x}")
print(f"Verify A @ x: {A @ x} (should be {b})")

# Larger system
A = np.array([
    [3, 2, -1],
    [2, -2, 4],
    [-1, 0.5, -1]
])
b = np.array([1, -2, 0])

print(f"\nLarger system:")
print(f"A:\n{A}")
print(f"b: {b}")

x = np.linalg.solve(A, b)
print(f"x: {x}")
print(f"Verify: {A @ x}")

# Multiple right-hand sides
B = np.array([[1, 2], [3, 4], [5, 6]])
A = np.array([[3, 1], [1, 2], [1, 1]])
print(f"\nMultiple RHS: A{A.shape} @ X = B{B.shape}")
try:
    X = np.linalg.solve(A, B)  # only works if A is square
except Exception as e:
    print(f"Error (A not square): {e}")

# Use lstsq for overdetermined
X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
print(f"\nlstsq X:\n{X}")
print(f"residuals: {residuals}")
print(f"rank: {rank}")

# Verify
print(f"A @ X:\n{A @ X}")

# Sparse-like: tri-diagonal via solve_banded (simplified)
from scipy.linalg import solve_banded  # optional