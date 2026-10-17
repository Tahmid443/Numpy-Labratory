"""Challenges: numerical precision."""

import numpy as np

def challenge_1():
    """Check if 0.1 + 0.2 equals 0.3 exactly."""
    return 0.1 + 0.2 == 0.3, np.isclose(0.1 + 0.2, 0.3)

def challenge_2():
    """Compare float32 vs float64 precision for pi."""
    import math
    pi64 = np.float64(math.pi)
    pi32 = np.float32(math.pi)
    return float(abs(pi64 - pi32))

def challenge_3():
    """Sum 100,000 tiny numbers in float32 and float64."""
    n = 100_000
    val = 1e-8
    sum32 = np.full(n, val, dtype=np.float32).sum()
    sum64 = np.full(n, val, dtype=np.float64).sum()
    return sum32, sum64, val * n

def challenge_4():
    """Safely compute log(sum(exp(x))) using logsumexp trick."""
    x = np.array([1000, 1001, 1002])
    max_x = x.max()
    result = max_x + np.log(np.sum(np.exp(x - max_x)))
    return result

def challenge_5():
    """Numerically stable sigmoid."""
    x = np.array([-1000, -10, 0, 10, 1000])
    result = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    return result

def challenge_6():
    """Check underflow behavior."""
    return np.array([1e-300]) / np.array([1e10])

def challenge_7():
    """Find safe tolerance for comparing two arrays."""
    a = np.random.default_rng(42).random(10)
    b = a + np.random.default_rng(43).random(10) * 1e-10
    return np.allclose(a, b, rtol=1e-9)

def challenge_8():
    """Verify associativity failure in float."""
    a, b, c = 1e16, -1e16, 1.0
    lhs = (a + b) + c
    rhs = a + (b + c)
    return lhs, rhs, lhs == rhs

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}: {globals()[f'challenge_{i}']()}")