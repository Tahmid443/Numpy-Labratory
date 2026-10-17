"""Common broadcasting errors and how to fix them."""

import numpy as np

# Error 1: incompatible dimensions
try:
    a = np.ones((2, 3))
    b = np.ones((2, 4))
    a + b
except ValueError as e:
    print(f"Error 1 (incompatible inner dims):\n  {e}")
    print("  Fix: use matching shapes or reshape")

# Error 2: (3,) and (4,)
try:
    np.ones(3) + np.ones(4)
except ValueError as e:
    print(f"\nError 2 (different lengths):\n  {e}")
    print("  Fix: np.ones(3)[:, np.newaxis] + np.ones(4) for outer")

# Error 3: wrong axis for outer operation
try:
    a = np.arange(3)
    b = np.arange(3)
    # a * b works (element-wise) but not outer product
    print(f"\na * b (element-wise): {a * b}")
    # For outer product use:
    print(f"Outer product: {a[:, np.newaxis] * b}")
except Exception as e:
    print(f"Error: {e}")

# Error 4: forget to add axis for row operations
m = np.arange(6).reshape(2, 3)
row = np.array([10, 20, 30])
# This works - row broadcast to each row
print(f"\nm + row:\n{m + row}")

# But for column operations
try:
    col_wrong = np.array([10, 20])
    print(f"\nm + col_wrong (wrong): {m + col_wrong}")
except ValueError as e:
    print(f"\nError 4: {e}")
    print("Fix: use column reshape:")
    print(m + np.array([10, 20])[:, np.newaxis])

# Error 5: np.where with incompatible shapes
try:
    np.where(np.ones(3) > 0, np.ones(4), 0)
except Exception as e:
    print(f"\nError 5: {e}")