"""SVD, matrix rank, and related decompositions."""

import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A:\n{A}")
print(f"A.shape: {A.shape}")

# Rank
rank = np.linalg.matrix_rank(A)
print(f"\nRank: {rank}")

# SVD
U, s, Vt = np.linalg.svd(A)
print(f"\nU shape: {U.shape}")
print(f"s (singular values): {s}")
print(f"Vt shape: {Vt.shape}")

# Reconstruct
print(f"\nReconstruction U @ diag(s) @ Vt:\n{U @ np.diag(s) @ Vt}")

# With full matrices
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"\nfull_matrices=False:")
print(f"U.shape: {U.shape}")
print(f"s.shape: {s.shape}")
print(f"Vt.shape: {Vt.shape}")

# Low-rank approximation
A_big = np.random.default_rng(42).random((10, 10))
U, s, Vt = np.linalg.svd(A_big, full_matrices=False)
k = 3
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
print(f"\nRank-{k} approximation error: {np.linalg.norm(A_big - A_approx):.4f}")
print(f"Original norm: {np.linalg.norm(A_big):.4f}")

# Qr decomposition
Q, R = np.linalg.qr(A)
print(f"\nQR:")
print(f"Q shape: {Q.shape}\nQ:\n{Q}")
print(f"R shape: {R.shape}\nR:\n{R}")
print(f"Q @ R:\n{Q @ R}")

# Cholesky (positive definite only)
S = np.array([[4, 2], [2, 3]])
L = np.linalg.cholesky(S)
print(f"\nCholesky:\n{L}")
print(f"L @ L.T:\n{L @ L.T}")
print(f"Original S:\n{S}")

# condition number
print(f"\ncondition number of A: {np.linalg.cond(A):.4f}")