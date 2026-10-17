"""Creating arrays from Python sequences."""

import numpy as np

# From list
a = np.array([1, 2, 3, 4, 5])
print(f"From list: {a}")

# From tuple
b = np.array((1, 2, 3))
print(f"From tuple: {b}")

# From list of lists (2D)
c = np.array([[1, 2, 3], [4, 5, 6]])
print(f"From nested list:\n{c}")
print(f"  Shape: {c.shape}, ndim: {c.ndim}")

# From list of lists of lists (3D)
d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"From 3-level list:\n{d}")
print(f"  Shape: {d.shape}, ndim: {d.ndim}")

# Specify dtype
e = np.array([1, 2, 3], dtype=np.float32)
print(f"Float32 array: {e}, dtype={e.dtype}")

# Mixed types are upcast
f = np.array([1, 2.5, 3])
print(f"Mixed list: {f}, dtype={f.dtype}")

# From range
g = np.array(range(10))
print(f"From range: {g}")

# From string
h = np.array("hello")
print(f"From string: {h!r}, dtype={h.dtype}")