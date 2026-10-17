"""How floating point numbers are represented (IEEE 754)."""

import numpy as np
import struct

def float_to_bits(f):
    """Show IEEE 754 bit representation."""
    if isinstance(f, np.float32):
        return format(struct.unpack('I', struct.pack('f', f))[0], '032b')
    else:
        return format(struct.unpack('Q', struct.pack('d', f))[0], '064b')

# Python float is double (float64)
x = 0.1
print(f"0.1 in float64 bits:")
bits = float_to_bits(x)
print(f"  Sign: {bits[0]}")
print(f"  Exponent: {bits[1:12]}")
print(f"  Mantissa: {bits[12:]}")

# 0.1 is not exactly representable
print(f"\n0.1 + 0.2 == 0.3: {0.1 + 0.2 == 0.3}")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"np.isclose: {np.isclose(0.1 + 0.2, 0.3)}")

# Repeating binary fractions
print(f"\n1/3 in binary repeats:")
print(f"  repr(1/3): {1/3!r}")
print(f"  Decimal: {format(1/3, '.20f')}")

# Exact floats
print(f"\nExact powers of 2:")
for x in [0.5, 0.25, 0.125, 0.0625]:
    print(f"  {x}: {x!r}")

# Integers up to 2^53 exact in float64
print(f"\n2^53 = {2**53}, 2^53 + 1 = {2**53 + 1}")
print(f"float64(2^53) = {float(2**53)}")
print(f"float64(2^53 + 1) = {float(2**53 + 1)}")
print(f"Equal? {float(2**53) == float(2**53 + 1)}")

# Comparison with tolerance
print(f"\nComparison with tolerance:")
a = 0.1 + 0.2
b = 0.3
print(f"  abs diff: {abs(a - b):.2e}")
print(f"  np.isclose: {np.isclose(a, b)}")
print(f"  np.allclose: {np.allclose(a, b)}")