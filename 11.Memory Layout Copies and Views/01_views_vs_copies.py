"""Views vs copies."""

import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(f"Original: {a}")

# Slicing returns a VIEW
b = a[1:4]
b[0] = 99
print(f"\nSlice view: original changed: {a}")

# Basic indexing returns view (for scalars, it's a copy)
a = np.arange(10)
scalar = a[2]
scalar = 100
print(f"\nScalar is a copy: original unchanged: {a}")

# Fancy indexing returns a COPY
a = np.arange(10)
c = a[[1, 3, 5]]
c[0] = 99
print(f"\nFancy index is copy: original unchanged: {a}")

# Boolean indexing returns a COPY
a = np.arange(10)
d = a[a > 5]
d[0] = 99
print(f"\nBoolean index is copy: original unchanged: {a}")

# Methods returning view vs copy
a = np.arange(10)
print(f"\nravel(): {'view' if a.ravel().base is not None else 'copy'}")
print(f"flatten(): {'view' if a.flatten().base is not None else 'copy'}")

# Checking .base
a = np.arange(12)
b = a[:5]
c = a.copy()
print(f"\nb.base is a: {b.base is a}")
print(f"c.base is a: {c.base is a}")

# Transpose returns view
d = a.reshape(3, 4).T
print(f"transpose base exists: {d.base is not None}")

# Reshape returns view if possible
e = a.reshape(3, 4)
print(f"reshape base exists: {e.base is a}")

# Non-contiguous reshape returns copy
f = a[::2]
g = f.reshape(2, 3) if f.size == 6 else None
print(f"contiguous reshape base: {g.base if g is not None else 'N/A'}")