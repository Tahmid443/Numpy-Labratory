"""Challenges: array creation."""

import numpy as np

def challenge_1():
    """Create an array [10, 20, 30, 40, 50]."""
    return np.array([10, 20, 30, 40, 50])

def challenge_2():
    """Create 4x4 zeros matrix."""
    return np.zeros((4, 4))

def challenge_3():
    """Create array of even numbers from 0 to 20."""
    return np.arange(0, 21, 2)

def challenge_4():
    """Create 5 evenly spaced values from 5 to 25."""
    return np.linspace(5, 25, 5)

def challenge_5():
    """Create identity matrix of size 6."""
    return np.identity(6)

def challenge_6():
    """Create 3x3 matrix with all elements = 7."""
    return np.full((3, 3), 7)

def challenge_7():
    """Create a 2D array [[1,2,3],[4,5,6]] using arange and reshape."""
    return np.arange(1, 7).reshape(2, 3)

def challenge_8():
    """Create a 5x5 checkerboard of 0s and 1s."""
    return (np.indices((5, 5)).sum(axis=0) % 2)

if __name__ == "__main__":
    for i in range(1, 9):
        result = globals()[f"challenge_{i}"]()
        print(f"Challenge {i}:\n{result}\n")