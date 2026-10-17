"""NaN, infinity, and finite checks."""

import numpy as np

a = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.0])
print(f"Array: {a}")

# Checks
print(f"\nnp.isnan: {np.isnan(a)}")
print(f"np.isinf: {np.isinf(a)}")
print(f"np.isposinf: {np.isposinf(a)}")
print(f"np.isneginf: {np.isneginf(a)}")
print(f"np.isfinite: {np.isfinite(a)}")

# NaN behavior
print(f"\nNaN behavior:")
print(f"  nan == nan: {np.nan == np.nan}")
print(f"  np.nan is np.nan: {np.nan is np.nan}")
print(f"  math.isnan(np.nan): {np.isnan(np.nan)}")

# Comparisons with NaN are False
b = np.array([1.0, np.nan, 3.0])
print(f"\nb: {b}")
print(f"  b > 1.5: {b > 1.5}")
print(f"  b == np.nan: {b == np.nan}")

# Filter out NaN/inf
print(f"\nFiltering:")
print(f"  Finite values: {a[np.isfinite(a)]}")
print(f"  Not NaN: {a[~np.isnan(a)]}")

# NaN in aggregations
print(f"\nAggregations:")
print(f"  sum: {b.sum()}")
print(f"  mean: {b.mean()}")
print(f"  nansum: {np.nansum(b)}")
print(f"  nanmean: {np.nanmean(b)}")

# Inf propagation
print(f"\nInf propagation:")
c = np.array([1.0, np.inf])
print(f"  c: {c}")
print(f"  c.sum(): {c.sum()}")
print(f"  c.mean(): {c.mean()}")
print(f"  1/c: {1/c}")

# Set NaN
d = np.array([1.0, 2.0, 3.0])
d[1] = np.nan
print(f"\nSet NaN: {d}")

# Replace
print(f"Replace NaN with 0: {np.nan_to_num(d)}")
print(f"Replace NaN with mean: {np.nan_to_num(d, nan=np.nanmean(d))}")

# Creating NaN
print(f"\nCreating special values:")
print(f"  np.nan: {np.nan}")
print(f"  np.inf: {np.inf}")
print(f"  -np.inf: {-np.inf}")
print(f"  np.float64('nan'): {np.float64('nan')}")

# Operations with inf
print(f"\ninf + inf: {np.inf + np.inf}")
print(f"inf - inf: {np.inf - np.inf}")
print(f"inf * 0: {np.inf * 0}")