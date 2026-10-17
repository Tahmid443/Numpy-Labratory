"""Slicing 1D arrays."""

import numpy as np

a = np.arange(10)
print(f"Array: {a}")

print(f"a[2:5] = {a[2:5]}")        # [2,5)
print(f"a[:3] = {a[:3]}")          # [0,3)
print(f"a[7:] = {a[7:]}")          # [7,end)
print(f"a[::2] = {a[::2]}")        # every other
print(f"a[1::3] = {a[1::3]}")      # start=1, step=3
print(f"a[::-1] = {a[::-1]}")      # reversed
print(f"a[-3:] = {a[-3:]}")        # last 3
print(f"a[:-3] = {a[:-3]}")        # all but last 3

# Slicing returns a view (important!)
view = a[2:5]
view[0] = 999
print(f"\nAfter modifying view: {a}")

# Copy to avoid this
b = np.arange(10)
copy_slice = b[2:5].copy()
copy_slice[0] = 999
print(f"Copy approach: {b} (unchanged)")

# Slice assignment
c = np.arange(10)
c[2:5] = 0
print(f"\nSlice assignment c[2:5] = 0: {c}")

d = np.arange(10)
d[::2] = -1
print(f"d[::2] = -1: {d}")