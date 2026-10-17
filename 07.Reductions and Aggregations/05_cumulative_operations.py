"""Cumulative operations."""

import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(f"Array: {a}")
print(f"cumsum: {np.cumsum(a)}")
print(f"cumprod: {np.cumprod(a)}")

# cumsum with axis
m = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix:\n{m}")
print(f"cumsum() (flat): {m.cumsum()}")
print(f"cumsum(axis=0):\n{np.cumsum(m, axis=0)}")
print(f"cumsum(axis=1):\n{np.cumsum(m, axis=1)}")

# Accumulate methods
print(f"\nnp.add.accumulate(a): {np.add.accumulate(a)}")
print(f"np.multiply.accumulate(a): {np.multiply.accumulate(a)}")

# Cumulative max/min
print(f"\nnp.maximum.accumulate(a): {np.maximum.accumulate(a)}")
print(f"np.minimum.accumulate(a): {np.minimum.accumulate(a)}")

b = np.array([3, 1, 4, 1, 5, 9, 2])
print(f"\nArray: {b}")
print(f"running max: {np.maximum.accumulate(b)}")

# diff: differences between consecutive elements
print(f"\nnp.diff(b): {np.diff(b)}")
print(f"np.diff(b, n=2): {np.diff(b, n=2)}")
print(f"np.diff(m, axis=0):\n{np.diff(m, axis=0)}")

# ediff1d
print(f"np.ediff1d(b): {np.ediff1d(b)}")

# Cumulative sum example: interest
principal = 1000
rate = 0.05
years = 5
values = principal * (1 + rate) ** np.arange(years + 1)
print(f"\nCompound interest: {values}")
print(f"Cumulative gains: {np.diff(values)}")

# reset at zero: useful for position tracking
steps = np.array([1, 1, -1, 1, -1, -1, 1, 1, 1])
positions = np.cumsum(steps)
print(f"\nSteps: {steps}")
print(f"Positions: {positions}")