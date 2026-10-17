"""dtype and type conversion."""

import numpy as np

# Common dtypes
print("Common dtypes:")
print(f"  int8: {np.dtype(np.int8)}, range: -128 to 127")
print(f"  int32: {np.dtype(np.int32)}")
print(f"  int64: {np.dtype(np.int64)}")
print(f"  float32: {np.dtype(np.float32)}")
print(f"  float64: {np.dtype(np.float64)}")
print(f"  bool: {np.dtype(bool)}")
print(f"  complex128: {np.dtype(np.complex128)}")

# Astype conversion
arr = np.array([1.7, 2.9, 3.1])
print(f"\nOriginal: {arr}, dtype={arr.dtype}")
print(f"astype(int): {arr.astype(int)}  (truncation, not rounding)")
print(f"astype(str): {arr.astype(str)}")

# Safe upcasting
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = int_arr.astype(np.float64)
print(f"\nint32 -> float64: {float_arr}, dtype={float_arr.dtype}")

# Dangerous downcasting (may overflow)
big = np.array([1000, 2000, 3000], dtype=np.int64)
small = big.astype(np.int8)
print(f"\nOverflow example: {big} -> int8 -> {small}")

# Using dtype parameter during creation
a = np.array([1, 2, 3], dtype=np.float32)
print(f"\nCreated as float32: {a}, dtype={a.dtype}")

# Checking dtype properties
print(f"\nfloat32 is float: {np.issubdtype(np.float32, np.floating)}")
print(f"int32 is integer: {np.issubdtype(np.int32, np.integer)}")