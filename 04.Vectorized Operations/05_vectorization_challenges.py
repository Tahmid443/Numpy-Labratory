"""Challenges: vectorized operations."""

import numpy as np

def challenge_1():
    """Compute a^2 + b^2 for arrays."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    return a**2 + b**2

def challenge_2():
    """Normalize array to range [0, 1]."""
    a = np.array([10, 20, 30, 40, 50], dtype=float)
    return (a - a.min()) / (a.max() - a.min())

def challenge_3():
    """Apply ReLU to array: max(0, x)."""
    a = np.array([-3, -1, 0, 1, 3])
    return np.maximum(0, a)

def challenge_4():
    """Compute sigmoid: 1/(1+exp(-x))."""
    x = np.array([-2, -1, 0, 1, 2], dtype=float)
    return 1 / (1 + np.exp(-x))

def challenge_5():
    """Count elements > mean."""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return int((a > a.mean()).sum())

def challenge_6():
    """Compute Manhattan distance between two points."""
    p1 = np.array([1, 2])
    p2 = np.array([4, 6])
    return np.abs(p1 - p2).sum()

def challenge_7():
    """One-hot encode labels."""
    labels = np.array([0, 2, 1, 3, 0])
    n_classes = 4
    return np.eye(n_classes)[labels]

def challenge_8():
    """Clamp values to [lower, upper]."""
    a = np.array([-5, 0, 5, 10, 15])
    return np.clip(a, 0, 10)

if __name__ == "__main__":
    for i in range(1, 9):
        result = globals()[f'challenge_{i}']()
        print(f"Challenge {i}:\n{result}\n")