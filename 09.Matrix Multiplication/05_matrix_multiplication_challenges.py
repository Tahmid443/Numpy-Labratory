"""Challenges: matrix multiplication."""

import numpy as np

def challenge_1():
    """Compute A^2 for a matrix A."""
    A = np.array([[1, 2], [3, 4]])
    return A @ A

def challenge_2():
    """Compute matrix times its transpose (A @ A.T)."""
    A = np.random.default_rng(42).random((3, 4))
    return A @ A.T

def challenge_3():
    """Compute inner product of each row with itself (row norms squared)."""
    A = np.random.default_rng(42).random((5, 3))
    return np.einsum('ij,ij->i', A, A)

def challenge_4():
    """Compute bilinear form x^T A x."""
    x = np.array([1, 2, 3])
    A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    return x @ A @ x

def challenge_5():
    """Batch of matrix multiplications."""
    A = np.random.default_rng(42).random((5, 3, 3))
    B = np.random.default_rng(43).random((5, 3, 2))
    return A @ B

def challenge_6():
    """Compute covariance matrix from data."""
    rng = np.random.default_rng(42)
    data = rng.random((100, 3))
    centered = data - data.mean(axis=0)
    return centered.T @ centered / (len(data) - 1)

def challenge_7():
    """Compute weighted sum of columns."""
    data = np.array([[1, 2], [3, 4], [5, 6]])
    weights = np.array([0.3, 0.7])
    return data @ weights

def challenge_8():
    """Verify associativity of matrix multiplication."""
    rng = np.random.default_rng(42)
    A, B, C = rng.random((2, 2)), rng.random((2, 2)), rng.random((2, 2))
    lhs = (A @ B) @ C
    rhs = A @ (B @ C)
    return np.allclose(lhs, rhs)

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}:\n{globals()[f'challenge_{i}']()}\n")