"""Challenges: shape transformation."""

import numpy as np

def challenge_1():
    """Convert 1D array of 12 elements to 3x4."""
    return np.arange(12).reshape(3, 4)

def challenge_2():
    """Flatten 2D array in column-major order."""
    m = np.arange(12).reshape(3, 4)
    return m.ravel(order='F')

def challenge_3():
    """Reshape (2,3,4) to (6,4)."""
    a = np.arange(24).reshape(2, 3, 4)
    return a.reshape(6, 4)

def challenge_4():
    """Stack two 1D arrays as columns."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    return np.column_stack([a, b])

def challenge_5():
    """Split 1D array into chunks of 4."""
    a = np.arange(20)
    return np.split(a, 5)

def challenge_6():
    """Remove all length-1 dimensions from (1, 3, 1, 4)."""
    a = np.zeros((1, 3, 1, 4))
    return np.squeeze(a)

def challenge_7():
    """Batch of images: (N, H, W, C) -> (N, C, H, W)."""
    images = np.zeros((10, 32, 32, 3))
    return images.transpose(0, 3, 1, 2)

def challenge_8():
    """Tile a 1D array 3 times."""
    a = np.array([1, 2, 3])
    return np.tile(a, 3)

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}:\n{globals()[f'challenge_{i}']()}\n")