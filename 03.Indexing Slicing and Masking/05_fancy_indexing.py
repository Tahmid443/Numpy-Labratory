"""Fancy indexing: indexing with arrays of indices."""

import numpy as np

a = np.array([10, 20, 30, 40, 50])
print(f"Array: {a}")

# Select elements in order
print(f"a[[0, 2, 4]]: {a[[0, 2, 4]]}")
print(f"a[[4, 0, 2]]: {a[[4, 0, 2]]}")
print(f"a[[-1, 0]]: {a[[-1, 0]]}")

# Repeated indices
print(f"a[[0, 0, 1, 1, 1]]: {a[[0, 0, 1, 1, 1]]}")

# 2D fancy indexing
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D array:\n{b}")
print(f"b[[0, 2]] (rows 0 and 2):\n{b[[0, 2]]}")
print(f"b[[0, 2], [1, 0]] (pairs): {b[[0, 2], [1, 0]]}")

# Extract specific rows and columns
rows = [0, 2]
cols = [0, 2]
print(f"b[np.ix_({rows}, {cols})]:\n{b[np.ix_(rows, cols)]}")

# Fancy assignment
c = a.copy()
c[[1, 3]] = 0
print(f"\nAfter c[[1, 3]] = 0: {c}")

d = a.copy()
d[[0, 1, 4]] = [999, 888, 777]
print(f"After c[[0,1,4]] = [999, 888, 777]: {d}")

# Important: fancy indexing creates a copy
e = a[[0, 1, 2]]
e[0] = 999
print(f"\nFancy index copy (original unchanged): {a}")