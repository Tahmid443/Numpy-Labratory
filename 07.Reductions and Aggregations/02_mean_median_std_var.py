"""Statistical functions."""

import numpy as np

a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"Array: {a}")

# Mean, median
print(f"mean: {a.mean()}")
print(f"median: {np.median(a)}")

# Standard deviation, variance
print(f"std: {a.std()}")
print(f"var: {a.var()}")

# With ddof (delta degrees of freedom)
print(f"std (ddof=1): {a.std(ddof=1)}  # sample std")
print(f"var (ddof=1): {a.var(ddof=1)}")

# Percentiles and quantiles
print(f"\n25th percentile: {np.percentile(a, 25)}")
print(f"50th percentile: {np.percentile(a, 50)}")
print(f"75th percentile: {np.percentile(a, 75)}")
print(f"[25, 50, 75]: {np.percentile(a, [25, 50, 75])}")
print(f"quantile(0.5): {np.quantile(a, 0.5)}")

# 2D stats
m = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\nMatrix:\n{m}")
print(f"mean(): {m.mean()}")
print(f"mean(axis=0): {m.mean(axis=0)}")
print(f"mean(axis=1): {m.mean(axis=1)}")
print(f"std(axis=0): {m.std(axis=0)}")
print(f"median(axis=1): {np.median(m, axis=1)}")

# Average with weights
weights = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(f"\naverage with weights: {np.average(a, weights=weights)}")

# Peaks and valleys
print(f"\nargmax: {a.argmax()}")
print(f"argmin: {a.argmin()}")

# Correlation
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
print(f"\ncorrcoef(x, y):\n{np.corrcoef(x, y)}")