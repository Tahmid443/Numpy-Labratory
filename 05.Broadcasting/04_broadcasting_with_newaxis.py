"""Using newaxis (np.newaxis) to control broadcasting."""

import numpy as np

a = np.array([1, 2, 3])
print(f"a.shape: {a.shape}")

# Add new axis
b = a[:, np.newaxis]
print(f"a[:, np.newaxis].shape: {b.shape}")
print(f"a[:, np.newaxis]:\n{b}")

c = a[np.newaxis, :]
print(f"a[np.newaxis, :].shape: {c.shape}")
print(f"a[np.newaxis, :]:\n{c}")

# Expand_dims equivalent
print(f"\nnp.expand_dims(a, axis=0).shape: {np.expand_dims(a, 0).shape}")
print(f"np.expand_dims(a, axis=1).shape: {np.expand_dims(a, 1).shape}")

# Practical: outer product
x = np.array([1, 2, 3])
y = np.array([10, 20])
outer = x[:, np.newaxis] * y[np.newaxis, :]
print(f"\nOuter product shape: {outer.shape}")
print(outer)

# Pairwise differences
a = np.array([1, 5, 3])
b = np.array([2, 4])
diff = a[:, np.newaxis] - b
print(f"\nPairwise differences (a - b):\n{diff}")

# Pairwise distances
points = np.array([[1, 2], [4, 6], [7, 8]])
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=-1))
print(f"\nPairwise distances:\n{distances}")

# Broadcasting with 3 arrays
a = np.arange(3)[:, np.newaxis, np.newaxis]
b = np.arange(4)[np.newaxis, :, np.newaxis]
c = np.arange(5)[np.newaxis, np.newaxis, :]
result = a + b + c
print(f"\na.shape={a.shape}, b.shape={b.shape}, c.shape={c.shape}")
print(f"result.shape: {result.shape}")