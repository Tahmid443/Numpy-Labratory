"""Overflow and underflow."""

import numpy as np

# Integer overflow
print("Integer overflow:")
a = np.array([200], dtype=np.int8)
b = a + 100
print(f"  int8 200 + 100 = {b[0]} (wraps around)")

a = np.array([127], dtype=np.int8)
b = a + 1
print(f"  int8 127 + 1 = {b[0]}")

# Python int has arbitrary precision
print(f"\nPython int: {2**63}")
try:
    print(f"np.int64(2**63): {np.int64(2**63)}")
except OverflowError as e:
    print(f"np.int64(2**63) error: {e}")

# Float overflow -> inf
print(f"\nFloat overflow:")
a = np.array([1e308], dtype=np.float64)
print(f"  1e308 * 10 = {a * 10}")

# Float underflow -> 0
print(f"\nFloat underflow:")
a = np.array([1e-308], dtype=np.float64)
print(f"  1e-308 / 10 = {a / 10}")
print(f"  1e-400: {np.array(1e-400)} (becomes 0)")

# Detecting inf/nan
a = np.array([1.0, np.inf, -np.inf, np.nan])
print(f"\na: {a}")
print(f"isinf: {np.isinf(a)}")
print(f"isnan: {np.isnan(a)}")
print(f"isfinite: {np.isfinite(a)}")

# Using np.seterr
old_settings = np.seterr(all='warn')
try:
    result = np.array([1.0]) / np.array([0.0])
    print(f"\n1/0 with warnings: {result}")
finally:
    np.seterr(**old_settings)

# Warning as exception
np.seterr(divide='raise')
try:
    np.array([1.0]) / np.array([0.0])
except FloatingPointError as e:
    print(f"\nCaught: {e}")
finally:
    np.seterr(divide='warn')

# Use larger dtype to avoid overflow
a = np.array([100_000], dtype=np.int32)
b = np.array([100_000], dtype=np.int32)
print(f"\nint32 100000 * 100000 = {a * b} (overflow!)")
print(f"int64: {a.astype(np.int64) * b.astype(np.int64)}")

# Float32 vs float64 overflow
a32 = np.array([1e38], dtype=np.float32)
print(f"\nfloat32 1e38 * 10: {a32 * 10}")
a64 = np.array([1e38], dtype=np.float64)
print(f"float64 1e38 * 10: {a64 * 10}")