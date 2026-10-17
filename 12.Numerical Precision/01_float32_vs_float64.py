"""float32 vs float64."""

import numpy as np

# Memory
a32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
a64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print(f"float32 itemsize: {a32.itemsize} bytes")
print(f"float64 itemsize: {a64.itemsize} bytes")

# Precision
print(f"\nPrecision test:")
value = 1/3
print(f"  Value: {value}")
print(f"  float64: {np.float64(value)}")
print(f"  float32: {np.float32(value)}")
print(f"  Difference: {abs(float(np.float64(value)) - float(np.float32(value))):.2e}")

# Machine epsilon
print(f"\nMachine epsilon:")
print(f"  float32 eps: {np.finfo(np.float32).eps}")
print(f"  float64 eps: {np.finfo(np.float64).eps}")

# Max/min values
print(f"\nfloat32 range: {np.finfo(np.float32).min:.2e} to {np.finfo(np.float32).max:.2e}")
print(f"float64 range: {np.finfo(np.float64).min:.2e} to {np.finfo(np.float64).max:.2e}")

# Type promotion
a = np.array([1.0], dtype=np.float32)
b = np.array([2.0], dtype=np.float64)
print(f"\nfloat32 + float64 -> {(a + b).dtype}")

# Sum of many small numbers
n = 1_000_000
arr32 = np.ones(n, dtype=np.float32) / n
arr64 = np.ones(n, dtype=np.float64) / n
print(f"\nSum of {n} tiny floats:")
print(f"  float32 sum: {arr32.sum()}")
print(f"  float64 sum: {arr64.sum()}")

# Operations preserve/upcast
print(f"\nfloat32 * float32: {(a32 * a32).dtype}")
print(f"float32 + int: {(a32 + 1).dtype}")