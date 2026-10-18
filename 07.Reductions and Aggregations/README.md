# NumPy — Reductions and Aggregations

Notes for **NumPy Laboratory → 07. Reductions and Aggregations**, covering how to collapse arrays into summary values: sums, extrema, statistics, the `axis` parameter, argmin/argmax, cumulative ops, and NaN-safe aggregates.

---

## 📁 Project Structure

```
07.Reductions and Aggregations/
├── 01_sum_min_max.py
├── 02_mean_median_std_var.py
├── 03_axis_parameter.py
├── 04_argmin_argmax.py
├── 05_cumulative_operations.py
├── 06_nan_aware_aggregations.py
└── 07_statistics_challenges.py
```

---

## 1️⃣ Sum, Min, Max & Variants

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

a.sum()      # 31
a.min()      # 1
a.max()      # 9
a.prod()     # 51840

np.sum(a); np.min(a); np.max(a)
np.amin(a); np.amax(a)   # aliases for min/max
np.ptp(a)                # peak-to-peak = max - min = 8
```

### 2-D reductions with `axis`

```python
m = np.array([[1, 2, 3], [4, 5, 6]])

m.sum()              # 21        (all elements)
m.sum(axis=0)        # [5 7 9]   (down columns)
m.sum(axis=1)        # [6 15]    (across rows)
m.max(axis=0)        # [4 5 6]
```

### `keepdims=True`

```python
m.sum(axis=1)                  # shape (2,)
m.sum(axis=1, keepdims=True)   # shape (2, 1) — broadcast-friendly
```

### `initial=` and `where=`

```python
a.sum(initial=100)             # 131 — start summing from 100
a.sum(where=a > 3)             # 20  — only sum elements > 3
```

### Element-wise combiners

```python
b = np.array([10, 20, 30, 40, 50, 60, 70, 80])

np.maximum(a, b)   # element-wise max
np.fmax(a, b)      # like maximum but ignores NaN
np.minimum(a, b)
np.fmin(a, b)
```

---

## 2️⃣ Mean, Median, Std, Var

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

a.mean()             # 3.875
np.median(a)         # 3.5
a.std()              # population std (ddof=0)
a.var()              # population variance
a.std(ddof=1)        # sample std
a.var(ddof=1)        # sample variance
```

> ⚠️ **`ddof` matters:** NumPy's default is **`ddof=0`** (population). For a **sample statistic**, use `ddof=1`.

### Percentiles & quantiles

```python
np.percentile(a, 25)          # 25th percentile
np.percentile(a, [25, 50, 75])
np.quantile(a, 0.5)           # equivalent to median
```

### 2-D stats

```python
m = np.array([[1, 2, 3], [4, 5, 6]])
m.mean(axis=0)               # column means → [2.5 3.5 4.5]
m.mean(axis=1)               # row means    → [2 5]
m.std(axis=0)
np.median(m, axis=1)
```

### Weighted average & correlation

```python
weights = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.average(a, weights=weights)

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
np.corrcoef(x, y)             # returns 2×2 correlation matrix
```

---

## 3️⃣ The `axis` Parameter

**The mental model:** `axis=k` means *"collapse axis k"* — the result has that axis **removed**.

| `ndim` | `axis=0` | `axis=1` | `axis=2` |
|---|---|---|---|
| 1-D | scalar | — | — |
| 2-D | down columns → shape `(cols,)` | across rows → shape `(rows,)` | — |
| 3-D | shape `(d1, d2)` | shape `(d0, d2)` | shape `(d0, d1)` |

```python
b = np.arange(24).reshape(2, 3, 4)

b.sum(axis=0).shape     # (3, 4)
b.sum(axis=1).shape     # (2, 4)
b.sum(axis=2).shape     # (2, 3)
b.sum(axis=(0, 1)).shape  # (4,)   — collapse multiple axes at once
b.sum(axis=-1).shape      # (2, 3) — same as axis=2
```

### Practical: normalize by column mean

```python
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
col_means = data.mean(axis=0)   # shape (3,)
data - col_means                # broadcasts → row-wise subtraction
```

---

## 4️⃣ `argmin` / `argmax` — Index of Extremes

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

a.argmax()    # 5  (first occurrence of 9)
a.argmin()    # 1  (first occurrence of 1)
a[a.argmax()] # 9
```

### 2-D — flat index by default

```python
m = np.array([[1, 5, 3], [4, 2, 8]])

m.argmax()                        # 5 (flat, row-major)
np.unravel_index(m.argmax(), m.shape)   # (1, 2)
```

### Per-axis index

```python
m.argmax(axis=0)   # index of max per column → [1 0 1]
m.argmax(axis=1)   # index of max per row    → [1 2]
```

### NaN-safe

```python
c = np.array([1.0, np.nan, 3.0, 2.0])
c.argmax()            # NaN propagates → unpredictable
np.nanargmax(c)       # 2  (index of 3.0)
np.nanargmin(c)       # 0
```

### Sorting & top-k

```python
d = np.array([3, 1, 4, 1, 5])
np.argsort(d)              # [1 3 0 2 4] — indices that sort
d[np.argsort(d)]           # sorted values
d[np.argsort(-d)]          # descending

# Top-k without full sort
e = np.array([10, 5, 20, 15, 30, 25, 8])
k = 3
top_idx = np.argpartition(e, -k)[-k:]
e[top_idx]
```

---

## 5️⃣ Cumulative Operations

| Function | Effect |
|---|---|
| `np.cumsum(a)` | running sum |
| `np.cumprod(a)` | running product |
| `np.maximum.accumulate(a)` | running max |
| `np.minimum.accumulate(a)` | running min |
| `np.diff(a)` | consecutive differences |
| `np.ediff1d(a)` | flattened differences |

```python
a = np.array([1, 2, 3, 4, 5])
np.cumsum(a)                          # [1 3 6 10 15]
np.cumprod(a)                         # [1 2 6 24 120]
np.maximum.accumulate([3,1,4,1,5,9])  # [3 3 4 4 5 9]
np.diff([3,1,4,1,5])                  # [-2 3 -3 4]
np.diff(a, n=2)                       # second differences
```

### With axis

```python
m = np.array([[1, 2, 3], [4, 5, 6]])
np.cumsum(m, axis=0)      # cumulative down columns
np.cumsum(m, axis=1)      # cumulative across rows
np.diff(m, axis=0)
```

### Real-world example: cumulative position

```python
steps = np.array([1, 1, -1, 1, -1, -1, 1, 1, 1])
np.cumsum(steps)   # [1 2 1 2 1 0 1 2 3]
```

---

## 6️⃣ NaN-Aware Aggregations

**Standard reductions propagate NaN** — `np.sum([1, np.nan])` is `nan`.

**`np.nan*` functions ignore NaN:**

| Standard | NaN-aware |
|---|---|
| `np.sum` | `np.nansum` |
| `np.mean` | `np.nanmean` |
| `np.min` | `np.nanmin` |
| `np.max` | `np.nanmax` |
| `np.std` | `np.nanstd` |
| `np.var` | `np.nanvar` |
| `np.median` | `np.nanmedian` |
| `np.argmax` | `np.nanargmax` |

```python
a = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])

a.sum()                # nan
np.nansum(a)           # 13.0
np.nanmean(a)          # 3.25
np.nanmax(a)           # 6.0
```

### All-NaN slices → 0

```python
b = np.array([[1.0, np.nan], [np.nan, np.nan]])
np.nansum(b, axis=0)   # [1.0, 0.0]  ← all-NaN column → 0
np.nansum(b, axis=1)   # [1.0, 0.0]
```

### Detecting & replacing NaN / inf

```python
np.isnan(a)                    # boolean mask
np.isnan(a).sum()              # count of NaN

c = a.copy()
c[np.isnan(c)] = 0             # replace NaN with 0

np.isfinite(d)                 # True where not NaN or inf
np.isinf(d)                    # True for +inf / -inf
d[np.isfinite(d)]              # filter finite values
```

> ⚠️ `np.nanmean` of an **all-NaN slice** returns `nan` **and emits a RuntimeWarning**. Handle this explicitly if it occurs.

---

## 7️⃣ Practice Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | Mean, median, std | `a.mean()`, `np.median(a)`, `a.std()` |
| 2 | 5 largest values | `np.sort(a)[-5:]` |
| 3 | Row-wise means | `m.mean(axis=1)` |
| 4 | Counts per quartile | `np.percentile` + boolean masks |
| 5 | Z-score | `(a - a.mean()) / a.std()` |
| 6 | Correlation matrix | `np.corrcoef(data)` |
| 7 | IQR outliers | `q1 - 1.5*iqr`, `q3 + 1.5*iqr` |
| 8 | Weighted avg | `np.average(values, weights=w)` |

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| Sum / prod | `a.sum()`, `a.prod()` |
| Min / max | `a.min()`, `a.max()` |
| Range | `np.ptp(a)` |
| Mean / median | `a.mean()`, `np.median(a)` |
| Std / var | `a.std(ddof=…)`, `a.var(ddof=…)` |
| Percentile | `np.percentile(a, [25, 50, 75])` |
| Weighted avg | `np.average(a, weights=…)` |
| Correlation | `np.corrcoef(x, y)` |
| Argmax / argmin | `a.argmax()`, `a.argmin()` |
| Sort indices | `np.argsort(a)` |
| Top-k | `np.argpartition(a, -k)[-k:]` |
| Running sum | `np.cumsum(a)` |
| Diff | `np.diff(a)` |
| NaN-safe | `np.nansum`, `np.nanmean`, `np.nanmax`, … |

---

## ✅ Takeaways

1. **`axis=k` collapses axis `k`** — the resulting array has that axis **removed**. Multi-axis: `axis=(0, 1)`.
2. **`keepdims=True`** keeps the collapsed axis as size 1 → enables clean broadcasting (e.g., row/col normalization).
3. **`ddof=0` by default** — use `ddof=1` for sample statistics.
4. `argmax`/`argmin` return the **flat index** in N-D arrays; combine with `np.unravel_index` for coordinates.
5. Use `np.argpartition` for **top-k** — `O(n)` instead of `O(n log n)`.
6. **NaN propagates** through standard reductions — always use `np.nan*` variants when data may contain NaN.
7. Combine `np.isfinite` / `np.isnan` masks with boolean indexing to clean data.

---
