"""Determinant and inverse."""

import numpy as np

A = np.array([[1, 2], [3, 4]])
print(f"A:\n{A}")

# Determinant
det = np.linalg.det(A)
print(f"det(A): {det}")

# Inverse
try:
    A_inv = np.linalg.inv(A)
    print(f"\nA_inv:\n{A_inv}")
    print(f"A @ A_inv (should be identity):\n{A @ A_inv}")
except np.linalg.LinAlgError as e:
    print(f"Inverse error: {e}")

# Singular matrix
B = np.array([[1, 2], [2, 4]])
print(f"\nB:\n{B}")
print(f"det(B): {np.linalg.det(B)} (near zero)")

try:
    np.linalg.inv(B)
except np.linalg.LinAlgError:
    print("B is singular, cannot invert")

# Pseudo-inverse
B_pinv = np.linalg.pinv(B)
print(f"\npinv(B):\n{B_pinv}")

# 3x3
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
print(f"\nC:\n{C}")
print(f"det(C): {np.linalg.det(C):.4f}")
print(f"inv(C):\n{np.linalg.inv(C)}")
print(f"C @ inv(C):\n{C @ np.linalg.inv(C)}")

# Trace
print(f"\ntrace(C): {np.trace(C)}")