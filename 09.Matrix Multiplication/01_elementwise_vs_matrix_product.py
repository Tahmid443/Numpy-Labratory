"""Element-wise vs matrix product."""

import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A:\n{A}")
print(f"B:\n{B}")

# Element-wise (Hadamard) product
print(f"\nElement-wise A * B:\n{A * B}")

# Matrix product
print(f"\nMatrix product A @ B:\n{A @ B}")
print(f"np.dot(A, B):\n{np.dot(A, B)}")
print(f"np.matmul(A, B):\n{np.matmul(A, B)}")

# Verify manually for 2x2
# A @ B:
# [1*5+2*7, 1*6+2*8] = [19, 22]
# [3*5+4*7, 3*6+4*8] = [43, 50]
print(f"\nManual verification:")
print(f"A[0] dot B[:,0] = {A[0] @ B[:, 0]} (should be 19)")
print(f"A[0] dot B[:,1] = {A[0] @ B[:, 1]} (should be 22)")

# Non-square
C = np.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
D = np.array([[1, 2], [3, 4], [5, 6]])  # (3,2)
print(f"\nC (2,3) @ D (3,2):\n{C @ D}  # result (2,2)")

# Element-wise impossible for non-matching shapes
try:
    C * D
except ValueError as e:
    print(f"\nElement-wise error: {e}")