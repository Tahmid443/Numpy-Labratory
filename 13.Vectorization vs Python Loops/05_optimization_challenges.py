"""Challenges: optimization."""

import numpy as np

def challenge_1():
    """Convert a Python function to vectorized form."""
    # Original Python: sum of squares of evens
    data = list(range(1000))
    py_result = sum(x**2 for x in data if x % 2 == 0)
    
    # Vectorized
    arr = np.arange(1000)
    np_result = (arr[arr % 2 == 0] ** 2).sum()
    
    return py_result, np_result, py_result == np_result

def challenge_2():
    """Count how many elements are above mean."""
    a = np.random.default_rng(42).random(10000)
    # Vectorized
    return int((a > a.mean()).sum())

def challenge_3():
    """Apply piecewise function."""
    x = np.linspace(-5, 5, 100)
    return np.where(x < 0, x**2, x**3)

def challenge_4():
    """Compute pairwise distance matrix."""
    points = np.random.default_rng(42).random((50, 3))
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

def challenge_5():
    """Moving average using convolution."""
    data = np.random.default_rng(42).random(1000)
    window = 10
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')

def challenge_6():
    """Count occurrences of each value."""
    data = np.random.default_rng(42).integers(0, 10, 10000)
    return np.bincount(data)

def challenge_7():
    """Sigmoid using stable formulation."""
    x = np.linspace(-100, 100, 200)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def challenge_8():
    """Groupby-like operation."""
    keys = np.array([1, 2, 1, 2, 1, 2])
    values = np.array([10, 20, 30, 40, 50, 60])
    unique_keys = np.unique(keys)
    return {k: values[keys == k].sum() for k in unique_keys}

if __name__ == "__main__":
    for i in range(1, 9):
        result = globals()[f'challenge_{i}']()
        print(f"Challenge {i}: {result if not hasattr(result, 'shape') else result.shape}")