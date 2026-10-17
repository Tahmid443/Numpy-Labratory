"""Batched matrix multiplication."""

import numpy as np

# Batch of matrices: (B, M, K) @ (B, K, N) -> (B, M, N)
B = 4
batch_A = np.random.default_rng(42).random((B, 2, 3))
batch_B = np.random.default_rng(42).random((B, 3, 5))

result = batch_A @ batch_B
print(f"batch_A shape: {batch_A.shape}")
print(f"batch_B shape: {batch_B.shape}")
print(f"Result shape: {result.shape}")

# Compare with loop
loop_result = np.array([batch_A[i] @ batch_B[i] for i in range(B)])
print(f"Allclose with loop: {np.allclose(result, loop_result)}")

# Broadcast same matrix across batch
A = np.random.default_rng(42).random((3, 2, 3))
single = np.random.default_rng(43).random((3, 5))

print(f"\nA shape: {A.shape}, single shape: {single.shape}")
print(f"A @ single shape: {(A @ single).shape}")

# Broadcast batch of matrices with single matrix
batch = np.random.default_rng(42).random((4, 3, 3))
single = np.random.default_rng(43).random((3, 3))
result = batch @ single
print(f"\nbatch{batch.shape} @ single{single.shape} -> {result.shape}")

# Broadcasting in both directions
A = np.random.default_rng(42).random((2, 1, 3, 4))
B = np.random.default_rng(43).random((1, 5, 4, 6))
print(f"\nA.shape: {A.shape}")
print(f"B.shape: {B.shape}")
result = A @ B
print(f"A @ B shape: {result.shape}")

# Practical: apply same transformation to many points
points = np.random.default_rng(42).random((100, 3))  # 100 3D points
transform = np.eye(3) + 0.1 * np.random.default_rng(43).random((3, 3))
transformed = points @ transform.T
print(f"\nPoints{points.shape} @ transform{transform.T.shape} -> {transformed.shape}")

# Batch of dot products with einsum
a = np.random.default_rng(42).random((4, 3))
b = np.random.default_rng(43).random((4, 3))
dots = np.einsum('ij,ij->i', a, b)
print(f"\nBatched dot products: {dots}")
print(f"Manual: {np.array([a[i] @ b[i] for i in range(4)])}")