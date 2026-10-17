"""Finding indices of min and max values."""

import numpy as np

a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Array: {a}")
print(f"argmax: {a.argmax()} (value={a[a.argmax()]})")
print(f"argmin: {a.argmin()} (value={a[a.argmin()]})")

# First occurrence by default
b = np.array([5, 1, 3, 5, 1, 5])
print(f"\nArray: {b}")
print(f"argmax: {b.argmax()}")

# Last occurrence: reverse
print(f"last argmax: {len(b) - 1 - b[::-1].argmax()}")

# 2D argmax (returns flat index by default)
m = np.array([[1, 5, 3], [4, 2, 8]])
print(f"\n2D:\n{m}")
print(f"m.argmax() = {m.argmax()} (flat index)")

# Use unravel_index to get 2D coords
flat_idx = m.argmax()
print(f"unravel_index: {np.unravel_index(flat_idx, m.shape)}")

# argmax along axis
print(f"\nm.argmax(axis=0): {m.argmax(axis=0)}  # index of max in each column")
print(f"m.argmax(axis=1): {m.argmax(axis=1)}  # index of max in each row")

# With NaN
c = np.array([1.0, np.nan, 3.0, 2.0])
print(f"\nWith NaN: {c}")
try:
    print(f"c.argmax(): {c.argmax()} (NaN propagates)")
except Exception as e:
    print(f"Error: {e}")

# nanargmax
print(f"np.nanargmax: {np.nanargmax(c)}")

# np.argsort for sorting indices
d = np.array([3, 1, 4, 1, 5])
print(f"\nd: {d}")
print(f"np.argsort(d): {np.argsort(d)}")
print(f"d[argsort]: {d[np.argsort(d)]}")
print(f"Descending: {d[np.argsort(-d)]}")

# np.argpartition for top-k
e = np.array([10, 5, 20, 15, 30, 25, 8])
k = 3
top_k_idx = np.argpartition(e, -k)[-k:]
print(f"\ne: {e}")
print(f"Top {k} indices: {top_k_idx}")
print(f"Top {k} values: {e[top_k_idx]}")