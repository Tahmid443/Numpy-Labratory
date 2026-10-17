"""Universal functions (ufuncs) - fast element-wise operations."""

import numpy as np

a = np.array([0, 1, 2, 3])
b = np.array([1, 4, 9, 16])

# Math ufuncs
print(f"np.sqrt(b): {np.sqrt(b)}")
print(f"np.exp(a): {np.exp(a)}")
print(f"np.log(np.array([1, np.e, np.e**2])): {np.log(np.array([1, np.e, np.e**2]))}")
print(f"np.sin(a): {np.sin(a)}")
print(f"np.cos(a): {np.cos(a)}")
print(f"np.tan(a): {np.tan(a)}")

# Rounding
x = np.array([1.4, 1.5, 1.6, 2.5, 2.4])
print(f"\nnp.floor(x): {np.floor(x)}")
print(f"np.ceil(x): {np.ceil(x)}")
print(f"np.round(x): {np.round(x)}")
print(f"np.trunc(x): {np.trunc(x)}")

# Arithmetic ufuncs
print(f"\nnp.add(2, 3): {np.add(2, 3)}")
print(f"np.subtract(5, 2): {np.subtract(5, 2)}")
print(f"np.multiply(3, 4): {np.multiply(3, 4)}")
print(f"np.divide(10, 4): {np.divide(10, 4)}")
print(f"np.power(2, 10): {np.power(2, 10)}")
print(f"np.mod(10, 3): {np.mod(10, 3)}")

# Comparison ufuncs
print(f"\nnp.greater(5, 3): {np.greater(5, 3)}")
print(f"np.maximum(a, b): {np.maximum(a, b)}")
print(f"np.minimum(a, b): {np.minimum(a, b)}")

# Absolute and sign
arr = np.array([-5, -3, 0, 2, 4])
print(f"\nnp.abs({arr}): {np.abs(arr)}")
print(f"np.sign({arr}): {np.sign(arr)}")

# Accumulation
print(f"\nnp.add.accumulate([1, 2, 3, 4]): {np.add.accumulate([1, 2, 3, 4])}")
print(f"np.add.reduce([1, 2, 3, 4]): {np.add.reduce([1, 2, 3, 4])}")

# Custom ufunc with out parameter
result = np.empty_like(a, dtype=float)
np.sqrt(b, out=result)
print(f"\nUsing out parameter: {result}")