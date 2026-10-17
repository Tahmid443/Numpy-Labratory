"""np.dot vs np.matmul vs @ operator."""

import numpy as np

# For 2D arrays, all three are equivalent
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A @ B:\n{A @ B}")
print(f"np.dot(A, B):\n{np.dot(A, B)}")
print(f"np.matmul(A, B):\n{np.matmul(A, B)}")

# For 1D arrays: dot = inner product, matmul = inner product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\n1D:")
print(f"  np.dot(a, b): {np.dot(a, b)}")
print(f"  np.matmul(a, b): {np.matmul(a, b)}")
print(f"  a @ b: {a @ b}")

# For ND arrays: dot uses last axis of first & second-to-last of second
# matmul uses last two dims and broadcasts the rest
X = np.arange(24).reshape(2, 3, 4)
Y = np.arange(20).reshape(2, 4, 5)

print(f"\n3D arrays:")
print(f"  np.dot(X, Y).shape: {np.dot(X, Y).shape}")
print(f"  np.matmul(X, Y).shape: {np.matmul(X, Y).shape}")

# Vector @ matrix
v = np.array([1, 2])
m = np.array([[3, 4], [5, 6]])
print(f"\nv @ m: {v @ m}")
print(f"m @ v: {m @ v}")

# Inner, outer, vdot
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"\nnp.inner: {np.inner(a, b)}")
print(f"np.outer:\n{np.outer(a, b)}")
print(f"np.vdot (flat dot): {np.vdot(a, b)}")

# tensordot
A = np.arange(12).reshape(3, 4)
B = np.arange(20).reshape(4, 5)
print(f"\nnp.tensordot(A, B, axes=1).shape: {np.tensordot(A, B, axes=1).shape}")

# einsum: powerful einsum
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\nnp.einsum('ij,jk->ik', A, B):\n{np.einsum('ij,jk->ik', A, B)}")
print(f"Same as A @ B: {np.allclose(np.einsum('ij,jk->ik', A, B), A @ B)}")