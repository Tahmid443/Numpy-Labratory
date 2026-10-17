"""Broadcasting rules explained."""

import numpy as np

"""
Broadcasting rules:
1. If arrays have different ndim, pad the smaller shape with 1s on the left
2. Dimensions are compatible if they are equal OR one of them is 1
3. The smaller array is "broadcast" to match the larger shape
"""

# Rule 1: same shape - element-wise
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Same shape: {a} + {b} = {a + b}")

# Rule 2: scalar broadcasting
print(f"\nScalar: {a} + 10 = {a + 10}")

# Rule 3: (3, 3) + (3,) broadcasts (3,) to (3, 3)
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([10, 20, 30])
print(f"\nMatrix + Vector:\n{m}\n+ {v}\n=\n{m + v}")

# Rule 4: (3, 1) + (1, 4) -> (3, 4)
col = np.array([[1], [2], [3]])
row = np.array([[10, 20, 30, 40]])
print(f"\n(3,1) + (1,4) -> (3,4):")
print(f"col shape: {col.shape}, row shape: {row.shape}")
print(f"Result shape: {(col + row).shape}")
print(f"Result:\n{col + row}")

# Rule 5: Incompatible shapes
try:
    a = np.ones((3, 4))
    b = np.ones((3, 3))
    a + b
except ValueError as e:
    print(f"\nIncompatible:\n{e}")

# Demonstrate padding
a = np.ones((2, 3, 4))
b = np.ones(4)  # treated as (1, 1, 4)
print(f"\n(2,3,4) + (4,) -> {(a+b).shape}")

b = np.ones((3, 4))  # treated as (1, 3, 4)
print(f"(2,3,4) + (3,4) -> {(a+b).shape}")

b = np.ones((2, 1, 4))
print(f"(2,3,4) + (2,1,4) -> {(a+b).shape}")