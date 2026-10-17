"""NaN-aware aggregation functions."""

import numpy as np

a = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
print(f"Array with NaN: {a}")

# Standard functions propagate NaN
print(f"\nStandard functions:")
print(f"  sum: {a.sum()}")
print(f"  mean: {a.mean()}")
print(f"  min: {np.min(a)}")
print(f"  max: {np.max(a)}")

# NaN-aware versions
print(f"\nNaN-aware functions:")
print(f"  nansum: {np.nansum(a)}")
print(f"  nanmean: {np.nanmean(a)}")
print(f"  nanmin: {np.nanmin(a)}")
print(f"  nanmax: {np.nanmax(a)}")
print(f"  nanstd: {np.nanstd(a)}")
print(f"  nanvar: {np.nanvar(a)}")
print(f"  nanmedian: {np.nanmedian(a)}")

# All-NaN slice
b = np.array([[1.0, np.nan], [np.nan, np.nan]])
print(f"\nMatrix with NaNs:\n{b}")
print(f"nansum(axis=0): {np.nansum(b, axis=0)}")
print(f"nansum(axis=1): {np.nansum(b, axis=1)}")

# nanargmin/max
print(f"\nnanargmax: {np.nanargmax(a)}")

# Detect NaN
print(f"\nisnan(a): {np.isnan(a)}")
print(f"Count NaN: {np.isnan(a).sum()}")

# Replace NaN
c = a.copy()
c[np.isnan(c)] = 0
print(f"\nReplaced NaN with 0: {c}")

# Inf handling
d = np.array([1.0, np.inf, -np.inf, 4.0])
print(f"\nWith inf: {d}")
print(f"isfinite: {np.isfinite(d)}")
print(f"isinf: {np.isinf(d)}")

# Filter finite values
finite = d[np.isfinite(d)]
print(f"Finite values: {finite}")