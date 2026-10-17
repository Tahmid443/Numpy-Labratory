"""Challenges: statistics."""

import numpy as np

def challenge_1():
    """Compute mean, median, std of random array."""
    a = np.random.default_rng(42).integers(1, 100, 50)
    return a.mean(), np.median(a), a.std()

def challenge_2():
    """Find the 5 largest values in an array."""
    a = np.random.default_rng(42).integers(1, 1000, 100)
    return np.sort(a)[-5:]

def challenge_3():
    """Compute row-wise means of a 4x5 matrix."""
    m = np.random.default_rng(42).random((4, 5))
    return m.mean(axis=1)

def challenge_4():
    """Count elements in each quartile."""
    a = np.random.default_rng(42).random(1000)
    q = np.percentile(a, [25, 50, 75])
    counts = [
        int(((a >= 0) & (a < q[0])).sum()),
        int(((a >= q[0]) & (a < q[1])).sum()),
        int(((a >= q[1]) & (a < q[2])).sum()),
        int((a >= q[2]).sum()),
    ]
    return counts

def challenge_5():
    """Z-score normalization."""
    a = np.array([1, 2, 3, 4, 5], dtype=float)
    return (a - a.mean()) / a.std()

def challenge_6():
    """Compute correlation matrix of 3 variables."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 100))
    return np.corrcoef(data)

def challenge_7():
    """Find outlier using IQR method."""
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return a[(a < lower) | (a > upper)]

def challenge_8():
    """Weighted average."""
    values = np.array([10, 20, 30])
    weights = np.array([1, 2, 3])
    return np.average(values, weights=weights)

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}:\n{globals()[f'challenge_{i}']()}\n")