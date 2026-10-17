"""Detecting and controlling memory sharing."""

import numpy as np

a = np.array([1, 2, 3, 4, 5])

# np.shares_memory checks overlap
b = a[:3]
c = a.copy()
print(f"a: {a}")
print(f"b = a[:3]")
print(f"c = a.copy()")
print(f"shares_memory(a, b): {np.shares_memory(a, b)}")
print(f"shares_memory(a, c): {np.shares_memory(a, c)}")

# np.may_share_memory (more permissive)
print(f"may_share_memory(a, b): {np.may_share_memory(a, b)}")

# Base object
print(f"\nb.base is a: {b.base is a}")
print(f"c.base: {c.base}")

# Force copy on non-overlapping slice
d = a[::2]
print(f"\nd = a[::2] shares: {np.shares_memory(a, d)}")

# Memory-mapped files
import tempfile
import os

with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
    filename = f.name

try:
    arr = np.arange(1000)
    np.save(filename, arr)
    loaded = np.load(filename, mmap_mode='r')
    print(f"\nMemory-mapped load: type={type(loaded)}")
    print(f"First 5: {loaded[:5]}")
    loaded2 = np.load(filename)  # regular load
    print(f"Regular load type: {type(loaded2)}")
finally:
    os.remove(filename)

# Verify memory sharing in operations
a = np.arange(10)
b = a.reshape(2, 5)
print(f"\nReshape shares: {np.shares_memory(a, b)}")

c = a.astype(float)
print(f"astype shares: {np.shares_memory(a, c)}")

d = a.view(np.float32)  # reinterpret
print(f"view shares: {np.shares_memory(a, d)}")

# .base chain
a = np.arange(24)
b = a.reshape(2, 3, 4)
c = b[0]
d = c[:, 1]
print(f"\nBase chain: d.base is c: {d.base is c}")
print(f"c.base is b: {c.base is b}")
print(f"b.base is a: {b.base is a}")
print(f"a.base is None: {a.base is None}")