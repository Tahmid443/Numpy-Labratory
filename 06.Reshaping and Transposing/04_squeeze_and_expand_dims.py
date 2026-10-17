"""squeeze, expand_dims, and atleast_* functions."""

import numpy as np

# expand_dims
a = np.array([1, 2, 3])
print(f"a.shape: {a.shape}")
print(f"expand_dims(a, 0).shape: {np.expand_dims(a, 0).shape}")
print(f"expand_dims(a, 1).shape: {np.expand_dims(a, 1).shape}")
print(f"expand_dims(a, -1).shape: {np.expand_dims(a, -1).shape}")

# squeeze: remove length-1 dimensions
b = np.array([[[1, 2, 3]]])
print(f"\nb.shape: {b.shape}")
print(f"squeeze(b).shape: {np.squeeze(b).shape}")
print(f"squeeze(b): {np.squeeze(b)}")

c = np.zeros((1, 3, 1, 4, 1))
print(f"\nc.shape: {c.shape}")
print(f"squeeze(c).shape: {np.squeeze(c).shape}")

# squeeze with specific axis
d = np.zeros((1, 3, 1, 4, 1))
print(f"squeeze(d, axis=0).shape: {np.squeeze(d, axis=0).shape}")
print(f"squeeze(d, axis=2).shape: {np.squeeze(d, axis=2).shape}")

# Can't squeeze non-1 dimension
try:
    np.squeeze(np.zeros((2, 3)), axis=0)
except ValueError as e:
    print(f"\nError squeezing size-2 dim: {e}")

# atleast_1d, atleast_2d, atleast_3d
scalar = np.array(5)
print(f"\nscalar: {scalar}, ndim={scalar.ndim}")
print(f"atleast_1d: {np.atleast_1d(scalar)}, ndim={np.atleast_1d(scalar).ndim}")
print(f"atleast_2d: {np.atleast_2d(scalar)}, ndim={np.atleast_2d(scalar).ndim}")
print(f"atleast_3d: {np.atleast_3d(scalar)}, ndim={np.atleast_3d(scalar).ndim}")

v = np.array([1, 2, 3])
print(f"\natleast_2d({v}):\n{np.atleast_2d(v)}")
print(f"atleast_3d({v}):\n{np.atleast_3d(v)}")

# Multiple arrays
result = np.atleast_2d(np.array([1, 2]), np.array([3, 4]))
print(f"\nMultiple: {len(result)} arrays")
for i, r in enumerate(result):
    print(f"  {i}: shape={r.shape}")