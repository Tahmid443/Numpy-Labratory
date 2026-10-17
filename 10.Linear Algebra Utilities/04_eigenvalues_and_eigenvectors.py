"""Eigenvalues and eigenvectors."""

import numpy as np

A = np.array([[4, 2], [1, 3]])
print(f"A:\n{A}")

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: A @ v = lambda * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"\nEigenvalue {lam:.4f}:")
    print(f"  A @ v: {A @ v}")
    print(f"  lambda * v: {lam * v}")
    print(f"  Match: {np.allclose(A @ v, lam * v)}")

# Symmetric matrix: real eigenvalues
S = np.array([[4, 1], [1, 4]])
vals, vecs = np.linalg.eigh(S)
print(f"\nSymmetric matrix:")
print(f"eigvalsh: {vals}")
print(f"eigvecs:\n{vecs}")

# Verify orthonormal
print(f"\nV^T @ V (should be I):\n{vecs.T @ vecs}")

# Reconstruct from eigendecomposition
D = np.diag(vals)
print(f"\nReconstruction V @ D @ V^T:\n{vecs @ D @ vecs.T}")
print(f"Original S:\n{S}")

# Matrix powers using eigenvalues
print(f"\nS^2 via eig:\n{vecs @ np.diag(vals**2) @ vecs.T}")
print(f"S @ S:\n{S @ S}")

# Condition number
print(f"\ncondition number: {np.linalg.cond(A):.2f}")

# For non-symmetric complex eigenvalues
C = np.array([[0, -1], [1, 0]])
vals, vecs = np.linalg.eig(C)
print(f"\nRotation matrix eigenvalues: {vals}")