"""Mixed problem set covering all topics."""

import numpy as np

def problem_1():
    """Create array of 20 zeros, set every 3rd to 5."""
    a = np.zeros(20)
    a[::3] = 5
    return a

def problem_2():
    """Normalize to sum to 1."""
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    return a / a.sum()

def problem_3():
    """Matrix multiplication chain: A@B@C."""
    A = np.eye(3)
    B = np.arange(9).reshape(3, 3)
    C = np.ones((3, 3))
    return A @ B @ C

def problem_4():
    """Remove NaN and compute stats."""
    a = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    clean = a[~np.isnan(a)]
    return clean, clean.mean(), clean.std()

def problem_5():
    """One-hot encode."""
    labels = np.array([0, 2, 1, 0, 2])
    n_classes = 3
    return np.eye(n_classes)[labels]

def problem_6():
    """Compute cumulative product of non-zero elements."""
    a = np.array([1, 2, 0, 3, 4, 0, 5])
    nz = a[a != 0]
    return np.cumprod(nz)

def problem_7():
    """Find top 3 rows by sum from matrix."""
    rng = np.random.default_rng(42)
    m = rng.integers(0, 100, (10, 4))
    row_sums = m.sum(axis=1)
    top_idx = np.argsort(-row_sums)[:3]
    return m[top_idx]

def problem_8():
    """Moving average with window 3."""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    window = 3
    kernel = np.ones(window) / window
    return np.convolve(a, kernel, mode='valid')

def problem_9():
    """Pairwise Euclidean distances."""
    pts = np.array([[1, 2], [4, 6], [7, 8]])
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt((diff**2).sum(axis=-1))

def problem_10():
    """One-hot decode."""
    one_hot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    return one_hot.argmax(axis=1)

def problem_11():
    """Create diagonal matrix from vector."""
    v = np.array([1, 2, 3, 4])
    return np.diag(v)

def problem_12():
    """Extract lower triangle."""
    m = np.arange(1, 17).reshape(4, 4)
    return np.tril(m)

def problem_13():
    """Compute z-scores per column."""
    rng = np.random.default_rng(42)
    m = rng.random((5, 3))
    return (m - m.mean(axis=0)) / m.std(axis=0)

def problem_14():
    """Smooth an array using rolling max."""
    a = np.array([1, 3, 2, 5, 4, 7, 6])
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(a, 3).max(axis=1)

def problem_15():
    """Find indices where array changes sign."""
    a = np.array([1, -2, 3, -4, -5, 6, 7, -8])
    return np.where(np.diff(np.sign(a)) != 0)[0]

if __name__ == "__main__":
    print("=" * 60)
    print("MIXED PROBLEM SET")
    print("=" * 60)
    for i in range(1, 16):
        print(f"\nProblem {i}:")
        result = globals()[f'problem_{i}']()
        print(result)