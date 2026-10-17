"""Benchmarking with timeit."""

import numpy as np
import timeit

# Setup
n = 10_000
py_list = list(range(n))
np_arr = np.arange(n)

# Add operation
def py_add():
    return [x + 1 for x in py_list]

def np_add():
    return np_arr + 1

t_py = timeit.timeit(py_add, number=1000)
t_np = timeit.timeit(np_add, number=1000)
print(f"Add 10K elements (1000 iterations):")
print(f"  Python: {t_py:.4f}s")
print(f"  NumPy:  {t_np:.4f}s")
print(f"  Speedup: {t_py/t_np:.1f}x")

# Sum
t_py = timeit.timeit(lambda: sum(py_list), number=1000)
t_np = timeit.timeit(lambda: np_arr.sum(), number=1000)
print(f"\nSum:")
print(f"  Python: {t_py:.4f}s")
print(f"  NumPy:  {t_np:.4f}s")
print(f"  Speedup: {t_py/t_np:.1f}x")

# Multiply
t_py = timeit.timeit(lambda: [x * 2 for x in py_list], number=1000)
t_np = timeit.timeit(lambda: np_arr * 2, number=1000)
print(f"\nMultiply:")
print(f"  Python: {t_py:.4f}s")
print(f"  NumPy:  {t_np:.4f}s")
print(f"  Speedup: {t_py/t_np:.1f}x")

# Matrix operations
n = 50
A = np.random.default_rng(42).random((n, n))
B = np.random.default_rng(43).random((n, n))

t_np = timeit.timeit(lambda: A @ B, number=100)
print(f"\nMatrix multiply 50x50 (100 iters): {t_np:.4f}s")

# Using timeit with setup
t = timeit.timeit(
    'a + b',
    setup='import numpy as np; a = np.arange(10000); b = np.arange(10000)',
    number=1000
)
print(f"\nTimeit with setup: {t:.4f}s")

# Compare different sizes
print("\nScaling with size:")
for size in [100, 1000, 10_000, 100_000]:
    setup = f'import numpy as np; a = np.arange({size})'
    t_np = timeit.timeit('a + 1', setup=setup, number=1000)
    print(f"  size={size}: {t_np:.4f}s")