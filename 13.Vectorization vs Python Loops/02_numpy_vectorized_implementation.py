"""Vectorized NumPy implementation."""

import numpy as np
import time

if __name__ == "__main__":
    n = 100_000
    a = np.arange(n)
    b = np.arange(n)

    start = time.time()
    result = a + b
    print(f"NumPy add ({n} elements): {time.time() - start:.6f}s")

    start = time.time()
    result = a ** 2
    print(f"NumPy square: {time.time() - start:.6f}s")

    start = time.time()
    result = a.sum()
    print(f"NumPy sum: {time.time() - start:.6f}s")

    start = time.time()
    result = a @ b
    print(f"NumPy dot: {time.time() - start:.6f}s")

    # Matrix multiply
    n = 100
    A = np.arange(n * n).reshape(n, n)
    B = np.arange(n * n).reshape(n, n)
    start = time.time()
    result = A @ B
    print(f"NumPy matrix multiply ({n}x{n}): {time.time() - start:.6f}s")

    # Batched operations
    n = 1000
    matrix = np.random.default_rng(42).random((n, n))
    start = time.time()
    result = np.sqrt(np.exp(matrix))
    print(f"NumPy sqrt(exp) on {n}x{n}: {time.time() - start:.4f}s")