"""Conditional selection with where and select."""

import numpy as np

a = np.array([1, 5, 3, 8, 2, 9, 4])
print(f"Array: {a}")

# np.where (vectorized if-else)
result = np.where(a > 5, 100, 0)
print(f"np.where(a > 5, 100, 0): {result}")

# Nested where
result = np.where(a > 5, 100, np.where(a > 2, 50, 0))
print(f"Nested where: {result}")

# np.select for multiple conditions
conditions = [a < 3, a < 6, a < 9]
choices = ['low', 'medium', 'high']
result = np.select(conditions, choices, default='very high')
print(f"np.select: {result}")

# np.choose
result = np.choose(np.array([0, 1, 0, 1]), ['a', 'b'])
print(f"np.choose: {result}")

# np.clip
result = np.clip(a, 3, 7)
print(f"np.clip(a, 3, 7): {result}")

# np.isin
b = np.array([1, 5, 3, 8, 2, 9, 4])
mask = np.isin(b, [3, 4, 5])
print(f"np.isin(b, [3, 4, 5]): {mask}")
print(f"Filter: {b[mask]}")

# np.piecewise
x = np.linspace(-2, 2, 9)
y = np.piecewise(x, [x < 0, x >= 0], [lambda x: x**2, lambda x: x**3])
print(f"np.piecewise:\nx={x}\ny={y}")