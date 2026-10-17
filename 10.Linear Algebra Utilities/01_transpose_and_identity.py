"""Transpose and identity in linear algebra context."""

import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A:\n{A}")
print(f"A.T:\n{A.T}")

# Properties
print(f"\n(A.T).T == A: {np.array_equal((A.T).T, A)}")
print(f"shape of A: {A.shape}")
print(f"shape of A.T: {A.T.shape}")

# Identity
I = np.eye(3)
print(f"\nI:\n{I}")
print(f"A @ I (should be A):\n{A @ I[:, :2]}")

# Verify A @ A.T is symmetric
B = A @ A.T
print(f"\nA @ A.T:\n{B}")
print(f"Is symmetric: {np.allclose(B, B.T)}")

# Trace
print(f"\ntrace(A @ A.T): {np.trace(B)}")
print(f"A.T @ A:\n{A.T @ A}")

# Complex conjugate transpose
C = np.array([[1+1j, 2], [3, 4-1j]])
print(f"\nC:\n{C}")
print(f"C.T:\n{C.T}")
print(f"C.conj().T (Hermitian):\n{C.conj().T}")