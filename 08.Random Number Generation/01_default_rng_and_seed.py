"""Modern NumPy random API - default_rng and seeding."""

import numpy as np

# Modern approach (recommended)
rng = np.random.default_rng()
print(f"Random float: {rng.random()}")
print(f"Random integers: {rng.integers(0, 10, 5)}")

# Seeded for reproducibility
rng = np.random.default_rng(42)
print(f"\nSeeded rng 1: {rng.random(3)}")

rng = np.random.default_rng(42)
print(f"Seeded rng 2: {rng.random(3)} (same!)")

# Different seed = different values
rng = np.random.default_rng(43)
print(f"Seed 43: {rng.random(3)}")

# Legacy RandomState (avoid for new code)
np.random.seed(42)
print(f"\nLegacy RandomState: {np.random.rand(3)}")

# Generator object
rng = np.random.default_rng(123)
print(f"\nGenerator object type: {type(rng)}")
print(f"First call:  {rng.random(3)}")
print(f"Second call: {rng.random(3)} (different)")