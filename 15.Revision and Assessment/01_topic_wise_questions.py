"""Topic-wise revision questions."""

import numpy as np

def setup():
    np.random.seed(42)
    return np.random.default_rng(42)

def q_array_creation():
    """Q: Create a 3x3 matrix with 1 on diagonal and 0 elsewhere."""
    print("Q1: Create 3x3 identity matrix")
    result = np.eye(3)
    print(f"A: {result.tolist()}")

def q_indexing():
    """Q: Extract every other element from [0..9] starting at 1."""
    a = np.arange(10)
    print(f"Q2: Every other from index 1: {a[1::2]}")

def q_broadcasting():
    """Q: Add a (3,) vector to each row of a (4, 3) matrix."""
    M = np.arange(12).reshape(4, 3)
    v = np.array([1, 2, 3])
    print(f"Q3: M + v:\n{M + v}")

def q_reductions():
    """Q: Compute per-column mean of a 4x5 random matrix."""
    M = np.random.default_rng(42).random((4, 5))
    print(f"Q4: Per-column means: {M.mean(axis=0)}")

def q_reshape():
    """Q: Reshape 1D array of 12 to 3D (2,2,3)."""
    a = np.arange(12)
    print(f"Q5: Reshape (2,2,3):\n{a.reshape(2, 2, 3)}")

def q_boolean():
    """Q: Get all values between 3 and 7 (exclusive) from array."""
    a = np.array([1, 2, 3, 5, 6, 8, 4, 7, 9])
    print(f"Q6: (3 < a < 7): {a[(a > 3) & (a < 7)]}")

def q_where():
    """Q: Replace all values > 5 with 0."""
    a = np.array([1, 5, 8, 3, 9, 2])
    print(f"Q7: Replace >5 with 0: {np.where(a > 5, 0, a)}")

def q_linalg():
    """Q: Solve 2x+y=5, x+3y=7."""
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 7])
    x = np.linalg.solve(A, b)
    print(f"Q8: Solution: x={x[0]}, y={x[1]}")

def q_axis():
    """Q: Sum along axis=1 of (3,4) matrix."""
    M = np.arange(12).reshape(3, 4)
    print(f"Q9: Row sums: {M.sum(axis=1)}")

def q_argmax():
    """Q: Find index of maximum in each row of matrix."""
    M = np.array([[1, 5, 3], [4, 2, 8]])
    print(f"Q10: Row argmax: {M.argmax(axis=1)}")

def q_transpose():
    """Q: Swap the two axes of a (3,4) array."""
    M = np.arange(12).reshape(3, 4)
    print(f"Q11: Transposed shape: {M.T.shape}")

def q_random():
    """Q: Generate 100 standard normal samples with seed 42."""
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(100)
    print(f"Q12: Mean={samples.mean():.4f}, std={samples.std():.4f}")

def q_cumsum():
    """Q: Running sum of [1,2,3,4,5]."""
    a = np.array([1, 2, 3, 4, 5])
    print(f"Q13: cumsum: {np.cumsum(a)}")

def q_nan():
    """Q: Compute mean ignoring NaN."""
    a = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    print(f"Q14: nanmean: {np.nanmean(a)}")

def q_stack():
    """Q: Stack three arrays of shape (3,) into (3,3)."""
    a, b, c = np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])
    print(f"Q15: stacked:\n{np.stack([a, b, c])}")

if __name__ == "__main__":
    questions = [
        q_array_creation, q_indexing, q_broadcasting, q_reductions,
        q_reshape, q_boolean, q_where, q_linalg, q_axis, q_argmax,
        q_transpose, q_random, q_cumsum, q_nan, q_stack,
    ]
    for q in questions:
        q()
        print()