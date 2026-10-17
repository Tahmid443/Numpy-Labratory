"""Creating identity, eye, and diagonal arrays."""

import numpy as np

# Identity matrix
i1 = np.identity(3)
i2 = np.identity(4, dtype=int)
print(f"identity(3):\n{i1}")
print(f"identity(4, dtype=int):\n{i2}")

# eye: with optional offset
e1 = np.eye(3)
e2 = np.eye(3, k=1)
e3 = np.eye(3, k=-1)
e4 = np.eye(3, 4)
print(f"\neye(3):\n{e1}")
print(f"eye(3, k=1) (superdiagonal):\n{e2}")
print(f"eye(3, k=-1) (subdiagonal):\n{e3}")
print(f"eye(3, 4) (non-square):\n{e4}")

# diag: extract or construct
v = np.array([1, 2, 3, 4])
d1 = np.diag(v)
print(f"\ndiag(v) constructs matrix:\n{d1}")

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d2 = np.diag(m)
print(f"diag(m) extracts diagonal: {d2}")

d3 = np.diag(v, k=1)
print(f"diag(v, k=1):\n{d3}")

# diagflat
df = np.diagflat([[1, 2], [3, 4]])
print(f"\ndiagflat: {df}")