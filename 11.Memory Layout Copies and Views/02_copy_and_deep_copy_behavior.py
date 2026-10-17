"""Copy and deepcopy behavior."""

import numpy as np
import copy

a = np.array([[1, 2], [3, 4]])
print(f"Original:\n{a}")

# Standard assignment: both reference same data
b = a
b[0, 0] = 99
print(f"\nAssignment b = a:")
print(f"  a:\n{a}")
print(f"  b:\n{b}")
print(f"  Same object: {a is b}")

# .copy(): independent data
a = np.array([[1, 2], [3, 4]])
c = a.copy()
c[0, 0] = 99
print(f"\n.copy() c = a.copy():")
print(f"  a:\n{a}")
print(f"  c:\n{c}")
print(f"  Same object: {a is c}")
print(f"  Share memory: {np.shares_memory(a, c)}")

# copy.deepcopy()
a = np.array([[1, 2], [3, 4]])
d = copy.deepcopy(a)
d[0, 0] = 99
print(f"\ncopy.deepcopy():")
print(f"  a:\n{a}")
print(f"  d:\n{d}")

# np.array() with copy parameter
a = np.array([1, 2, 3])
b = np.array(a)  # copy by default
c = np.array(a, copy=False)  # no copy if possible
d = np.array(a, copy=True)   # explicit copy
print(f"\nnp.array(a): shares: {np.shares_memory(a, b)}")
print(f"np.array(a, copy=False): shares: {np.shares_memory(a, c)}")
print(f"np.array(a, copy=True): shares: {np.shares_memory(a, d)}")

# np.copy
e = np.copy(a)
e[0] = 99
print(f"\nnp.copy: original {a}, copy {e}")

# Object array deepcopy caveats
obj_arr = np.array([1, [2, 3], {'a': 1}], dtype=object)
obj_copy = obj_arr.copy()
obj_arr[1].append(99)
print(f"\nObject array shallow copy:")
print(f"  Original: {obj_arr}")
print(f"  Copy: {obj_copy}")
obj_deep = copy.deepcopy(obj_arr)
print(f"  Deep copy: {obj_deep}")