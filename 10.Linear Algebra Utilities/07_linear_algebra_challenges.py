"""Challenges: linear algebra."""

import numpy as np

def challenge_1():
    """Solve system: x + 2y = 5, 3x + 4y = 11."""
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 11])
    return np.linalg.solve(A, b)

def challenge_2():
    """Compute det of 3x3 matrix."""
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    return np.linalg.det(A)

def challenge_3():
    """Compute eigenvalues of a symmetric matrix."""
    S = np.array([[2, 1], [1, 2]])
    return np.linalg.eigvalsh(S)

def challenge_4():
    """Normalize all rows of a matrix to unit length."""
    M = np.array([[3, 4], [5, 12], [8, 15]])
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / norms

def challenge_5():
    """Find rank of a rank-deficient matrix."""
    A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    return np.linalg.matrix_rank(A)

def challenge_6():
    """Compute inverse and verify."""
    A = np.array([[2, 1], [1, 1]])
    A_inv = np.linalg.inv(A)
    return A_inv

def challenge_7():
    """Project vector onto another vector."""
    a = np.array([1, 2, 3])
    b = np.array([1, 0, 1])
    return (a @ b / (b @ b)) * b

def challenge_8():
    """Compute condition number of Hilbert matrix."""
    n = 5
    H = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
    return np.linalg.cond(H)

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}:\n{globals()[f'challenge_{i}']()}\n")