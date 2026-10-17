"""Challenges: broadcasting."""

import numpy as np

def challenge_1():
    """Compute (3,3) + (3,) without loops."""
    m = np.arange(9).reshape(3, 3)
    v = np.array([1, 2, 3])
    return m + v

def challenge_2():
    """Compute pairwise Euclidean distances between 2 sets of points."""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8], [9, 10]])
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.sqrt((diff**2).sum(axis=-1))

def challenge_3():
    """Compute multiplication table 1-9."""
    return np.arange(1, 10)[:, np.newaxis] * np.arange(1, 10)

def challenge_4():
    """Normalize each column of a matrix (mean 0, std 1)."""
    m = np.random.default_rng(42).random((4, 3))
    return (m - m.mean(axis=0)) / m.std(axis=0)

def challenge_5():
    """Compute cosine similarity between all pairs of vectors."""
    vectors = np.random.default_rng(42).random((5, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / norms
    return normed @ normed.T

def challenge_6():
    """Create checkerboard 8x8 using broadcasting."""
    even = np.arange(8)[:, np.newaxis]
    odd = np.arange(8)[np.newaxis, :]
    return (even + odd) % 2

def challenge_7():
    """Compute outer product sum: sum_i sum_j (a_i * b_j)."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    outer = a[:, np.newaxis] * b
    return outer.sum()

def challenge_8():
    """Create a 5x5 matrix where M[i,j] = i*j if i>=j else 0."""
    i = np.arange(5)[:, np.newaxis]
    j = np.arange(5)
    mask = i >= j
    return np.where(mask, i * j, 0)

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}:\n{globals()[f'challenge_{i}']()}\n")