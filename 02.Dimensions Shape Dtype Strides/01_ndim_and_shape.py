"""ndim and shape attributes."""

import numpy as np

a0 = np.array(42)
a1 = np.array([1, 2, 3])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for name, arr in [("0D", a0), ("1D", a1), ("2D", a2), ("3D", a3)]:
    print(f"{name}: ndim={arr.ndim}, shape={arr.shape}, size={arr.size}")

# Shape of scalar is ()
print(f"\nScalar shape: {a0.shape} (empty tuple)")
print(f"1D shape: {a1.shape}")

# Shape is a tuple
print(f"Shape type: {type(a2.shape)}")
print(f"Rows: {a2.shape[0]}, Cols: {a2.shape[1]}")

# 0-D array
scalar = np.array(5)
print(f"\n0-D array: {scalar}, ndim={scalar.ndim}, shape={scalar.shape}")
print(f"Access value: {scalar.item()}")