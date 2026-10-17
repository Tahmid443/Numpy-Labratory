"""Random integers and floats."""

import numpy as np

rng = np.random.default_rng(42)

# Random floats in [0, 1)
print(f"random(5): {rng.random(5)}")

# 2D
print(f"random((2, 3)):\n{rng.random((2, 3))}")

# Random integers [low, high)
print(f"\nintegers(0, 10, 5): {rng.integers(0, 10, 5)}")
print(f"integers(1, 100, (2, 3)):\n{rng.integers(1, 100, (2, 3))}")

# With endpoint=True
print(f"\nintegers(1, 6, 5, endpoint=True): {rng.integers(1, 6, 5, endpoint=True)}")

# Random from continuous distribution
print(f"\nstandard_normal(5): {rng.standard_normal(5)}")

# Uniform in [low, high]
print(f"\nuniform(0, 10, 5): {rng.uniform(0, 10, 5)}")

# Random choice (with/without replacement)
a = np.array([1, 2, 3, 4, 5])
print(f"\nchoice(5): {rng.choice(5)}")
print(f"choice(a, 3): {rng.choice(a, 3)}")
print(f"choice(a, 3, replace=False): {rng.choice(a, 3, replace=False)}")

# With probabilities
print(f"\nchoice with probs: {rng.choice(a, 5, p=[0.1, 0.1, 0.1, 0.1, 0.6])}")

# Permutation and shuffle
b = np.arange(10)
print(f"\nOriginal: {b}")
print(f"permutation: {rng.permutation(b)}")
print(f"Original unchanged: {b}")

rng.shuffle(b)
print(f"After shuffle (in-place): {b}")