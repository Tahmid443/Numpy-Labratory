"""arange and linspace: creating sequences."""

import numpy as np

# arange: like Python's range but returns array
a1 = np.arange(10)
a2 = np.arange(2, 10)
a3 = np.arange(0, 1, 0.1)
a4 = np.arange(10, 0, -2)
print(f"arange(10): {a1}")
print(f"arange(2, 10): {a2}")
print(f"arange(0, 1, 0.1): {a3}")
print(f"arange(10, 0, -2): {a4}")

# linspace: evenly spaced values including endpoints
l1 = np.linspace(0, 1, 5)
l2 = np.linspace(0, 10, 11)
l3 = np.linspace(-1, 1, 9)
print(f"\nlinspace(0, 1, 5): {l1}")
print(f"linspace(0, 10, 11): {l2}")
print(f"linspace(-1, 1, 9): {l3}")

# linspace without endpoint
l4 = np.linspace(0, 1, 5, endpoint=False)
print(f"linspace(0, 1, 5, endpoint=False): {l4}")

# logspace
log1 = np.logspace(0, 3, 4)
log2 = np.logspace(1, 3, 3, base=2)
print(f"\nlogspace(0, 3, 4): {log1}")
print(f"logspace(1, 3, 3, base=2): {log2}")

# Key difference
print("\narange: step-based, endpoint may be excluded")
print("linspace: count-based, endpoint included by default")