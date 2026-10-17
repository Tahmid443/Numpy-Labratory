"""Rounding methods and tolerance comparisons."""

import numpy as np

x = np.array([1.4, 1.5, 1.6, 2.5, 2.4, -1.5, -2.5])

# Rounding
print(f"x: {x}")
print(f"np.round(x): {np.round(x)}")  # banker's rounding
print(f"np.floor(x): {np.floor(x)}")
print(f"np.ceil(x): {np.ceil(x)}")
print(f"np.trunc(x): {np.trunc(x)}")
print(f"np.rint(x): {np.rint(x)}")

# Round to decimals
print(f"\nnp.round([1.2345, 2.3456], 2): {np.round([1.2345, 2.3456], 2)}")
print(f"np.round([1.2345, 2.3456], -1): {np.round([1.2345, 2.3456], -1)}")

# Tolerance
print(f"\nComparing floats:")
a = 1.0000001
b = 1.0000002
print(f"  a == b: {a == b}")
print(f"  np.isclose(a, b): {np.isclose(a, b)}")
print(f"  np.isclose with rtol=1e-6: {np.isclose(a, b, rtol=1e-6)}")

# isclose parameters
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0 + 1e-9, 2.0 - 1e-9, 3.0 + 1e-5])
print(f"\na: {a}")
print(f"b: {b}")
print(f"allclose: {np.allclose(a, b)}")
print(f"allclose default: {np.allclose(a, b)}")
print(f"allclose(rtol=1e-4): {np.allclose(a, b, rtol=1e-4)}")
print(f"allclose(atol=1e-4): {np.allclose(a, b, atol=1e-4)}")

# Quantization (round to nearest 0.25)
def quantize(x, step):
    return np.round(x / step) * step

print(f"\nQuantize to 0.25: {quantize(np.array([0.1, 0.4, 0.6, 0.9]), 0.25)}")

# Sum precision issues
arr = np.array([0.1] * 10)
print(f"\nSum of ten 0.1s: {arr.sum()}")
print(f"Expected: 1.0")
print(f"Difference: {arr.sum() - 1.0}")

# Use math.fsum for exact sum
import math
print(f"math.fsum: {math.fsum([0.1] * 10)}")