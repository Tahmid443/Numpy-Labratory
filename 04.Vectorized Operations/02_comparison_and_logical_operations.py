"""Comparison and logical operations."""

import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 2, 4, 1, 3])

# Comparison -> boolean arrays
print(f"a = {a}")
print(f"b = {b}")
print(f"\na > 3: {a > 3}")
print(f"a < 3: {a < 3}")
print(f"a == 3: {a == 3}")
print(f"a != 3: {a != 3}")
print(f"a >= 3: {a >= 3}")
print(f"a <= 3: {a <= 3}")

# Element-wise comparison between arrays
print(f"\na > b: {a > b}")
print(f"a == b: {a == b}")
print(f"a != b: {a != b}")

# Logical operations
print(f"\n(a > 1) & (a < 5): {(a > 1) & (a < 5)}")
print(f"(a < 2) | (a > 4): {(a < 2) | (a > 4)}")
print(f"~(a > 3): {~(a > 3)}")

# np.all and np.any
print(f"\n(a > 0).all(): {(a > 0).all()}")
print(f"(a > 4).any(): {(a > 4).any()}")
print(f"(a > 10).all(): {(a > 10).all()}")

# np.logical_and, logical_or, logical_not, logical_xor
print(f"\nnp.logical_and(a > 1, a < 5): {np.logical_and(a > 1, a < 5)}")
print(f"np.logical_or(a < 2, a > 4): {np.logical_or(a < 2, a > 4)}")
print(f"np.logical_xor(a > 2, b > 2): {np.logical_xor(a > 2, b > 2)}")

# np.isclose
print(f"\nnp.isclose([1.0, 2.0], [1.0000001, 2.0]): {np.isclose([1.0, 2.0], [1.0000001, 2.0])}")