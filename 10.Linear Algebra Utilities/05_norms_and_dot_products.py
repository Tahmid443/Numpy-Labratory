"""Norms and dot products."""

import numpy as np

v = np.array([3, 4])
print(f"v: {v}")

# Vector norms
print(f"\nL1 norm: {np.linalg.norm(v, 1)}  # |3| + |4| = 7")
print(f"L2 norm (Euclidean): {np.linalg.norm(v, 2)}  # sqrt(9+16) = 5")
print(f"Infinity norm: {np.linalg.norm(v, np.inf)}  # max(|3|, |4|) = 4")
print(f"Default (L2): {np.linalg.norm(v)}")

# 3D vector
v3 = np.array([1, 2, 3])
print(f"\nv3: {v3}")
print(f"L2 norm: {np.linalg.norm(v3):.4f}")
print(f"Normalized: {v3 / np.linalg.norm(v3)}")

# Matrix norms
A = np.array([[1, 2], [3, 4]])
print(f"\nA:\n{A}")
print(f"Frobenius norm: {np.linalg.norm(A, 'fro'):.4f}")
print(f"  = sqrt(sum(A^2)): {np.sqrt((A**2).sum()):.4f}")
print(f"Spectral norm: {np.linalg.norm(A, 2):.4f}  # largest singular value")
print(f"Nuclear norm: {np.linalg.norm(A, 'nuc'):.4f}")

# Row-wise and column-wise norms
M = np.array([[3, 4], [5, 12], [8, 15]])
print(f"\nM:\n{M}")
print(f"Row norms: {np.linalg.norm(M, axis=1)}")
print(f"Column norms: {np.linalg.norm(M, axis=0)}")

# Dot products
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\na: {a}, b: {b}")
print(f"dot: {np.dot(a, b)}")
print(f"inner: {np.inner(a, b)}")
print(f"vdot: {np.vdot(a, b)}")

# Angle between vectors
cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.arccos(np.clip(cos_angle, -1, 1))
print(f"\ncosine similarity: {cos_angle:.4f}")
print(f"angle: {np.degrees(angle):.2f} degrees")

# Cosine similarity matrix
vectors = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
])
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normed = vectors / norms
cos_sim = normed @ normed.T
print(f"\nCosine similarity matrix:\n{cos_sim}")