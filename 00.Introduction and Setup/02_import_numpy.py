"""Standard NumPy import conventions."""

# Standard convention
import numpy as np

# Check version
print(f"NumPy version: {np.__version__}")

# Common attributes
print(f"NumPy config: {np.show_config()}")

# Basic usage
a = np.array([1, 2, 3])
print(f"Array: {a}")
print(f"Type: {type(a)}")

# Alternative imports (not recommended)
# import numpy
# from numpy import array, zeros, ones