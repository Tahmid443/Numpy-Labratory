"""Challenges: indexing."""

import numpy as np

def challenge_1():
    """Get all even numbers from 0-19."""
    a = np.arange(20)
    return a[a % 2 == 0]

def challenge_2():
    """Get elements of a 3x3 matrix where value > 4."""
    m = np.arange(1, 10).reshape(3, 3)
    return m[m > 4]

def challenge_3():
    """Get first and last column of a 4x4 array."""
    m = np.arange(16).reshape(4, 4)
    return m[:, [0, -1]]

def challenge_4():
    """Reverse each row."""
    m = np.arange(12).reshape(3, 4)
    return m[:, ::-1]

def challenge_5():
    """Get middle 3x3 submatrix of a 5x5 array."""
    m = np.arange(25).reshape(5, 5)
    return m[1:4, 1:4]

def challenge_6():
    """Replace all negative values with 0."""
    a = np.array([-2, 3, -5, 7, -1, 8])
    result = a.copy()
    result[result < 0] = 0
    return result

def challenge_7():
    """Get 3 random indices from array without replacement."""
    rng = np.random.default_rng(42)
    a = np.arange(100)
    return a[rng.choice(100, 3, replace=False)]

def challenge_8():
    """Get anti-diagonal of 4x4 matrix."""
    m = np.arange(16).reshape(4, 4)
    return m[::-1, :][np.eye(4, dtype=bool)]

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}: {globals()[f'challenge_{i}']()}")