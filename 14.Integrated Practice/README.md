# NumPy — Integrated Practice

Notes for **NumPy Laboratory → 14. Integrated Practice**, applying everything learned so far to real-world style problems: student marks analysis, sales data, image manipulation, Monte Carlo simulation, and a mini data-analysis pipeline.

---

## 📁 Project Structure

```
14.Integrated Practice/
├── 01_student_marks_analysis.py
├── 02_sales_data_analysis.py
├── 03_image_array_manipulation.py
├── 04_monte_carlo_simulation.py
└── 05_numpy_mini_project.py
```

---

## 1️⃣ Student Marks Analysis

**Data:** `(students, subjects)` matrix of marks.

```python
marks = rng.integers(40, 100, size=(30, 5))
```

### Key operations

| Task | NumPy expression |
|---|---|
| Subject means | `marks.mean(axis=0)` |
| Subject std / min / max | `marks.std(axis=0)`, `.min(axis=0)`, `.max(axis=0)` |
| Student totals | `marks.sum(axis=1)` |
| Student averages | `marks.mean(axis=1)` |
| Top-k students | `np.argsort(-averages)[:3]` |
| Pass count | `(marks >= 40).sum()` |
| Best subject | `subject_means.argmax()` |
| Subject correlations | `np.corrcoef(marks.T)` |

### Grade assignment — nested `np.where`

```python
grades = np.where(averages >= 90, 'A',
         np.where(averages >= 80, 'B',
         np.where(averages >= 70, 'C',
         np.where(averages >= 60, 'D', 'F'))))
```

> 💡 For **more than 2–3 branches**, prefer `np.select` over nested `np.where` — far more readable.

### Grade distribution

```python
for g in ['A', 'B', 'C', 'D', 'F']:
    print(g, (grades == g).sum())
```

### Mental model

- `axis=0` → **per-subject** statistics (down columns)
- `axis=1` → **per-student** statistics (across rows)
- `argsort(-values)` → descending rank order

---

## 2️⃣ Sales Data Analysis

**Data:** `(12 months, 5 products)` matrix.

```python
sales = rng.integers(100, 1000, size=(12, 5))
```

### Core aggregations

| Task | Expression |
|---|---|
| Total per product | `sales.sum(axis=0)` |
| Total per month | `sales.sum(axis=1)` |
| Best product | `products[sales.sum(axis=0).argmax()]` |
| Best/worst month | `months[sales.sum(axis=1).argmax()]` |

### Quarterly totals — reshape + sum

```python
quarters = sales.reshape(4, 3, 5).sum(axis=1)   # → (4, 5)
```

### Month-over-month growth

```python
monthly = sales.sum(axis=1)
growth  = np.diff(monthly) / monthly[:-1] * 100
```

> 💡 `np.diff` computes consecutive differences in one shot — a staple for time-series growth rates.

### Product correlations

```python
np.corrcoef(sales.T)     # (5, 5) correlation matrix
```

`sales.T` because `corrcoef` treats **rows as variables**.

---

## 3️⃣ Image Array Manipulation

**Data:** `(H, W, 3)` `uint8` array — every image is just a 3-D array.

### Grayscale conversion

```python
weights = np.array([0.2989, 0.5870, 0.1140])   # ITU-R BT.601
gray = img @ weights                            # (H, W, 3) @ (3,) → (H, W)
```

This is a **batched matrix-vector product** — a natural application of broadcasting and `@`.

### Brightness, contrast, invert

```python
brightened = np.clip(img.astype(int) + 50, 0, 255).astype(np.uint8)
contrast   = np.clip(img.astype(float) * 1.5, 0, 255).astype(np.uint8)
inverted   = 255 - img
```

> ⚠️ **Always cast to a wider type before arithmetic** — `uint8 + 50` **wraps around** to small values.

### Geometric transforms via slicing

| Operation | Expression |
|---|---|
| Horizontal flip | `img[:, ::-1]` |
| Vertical flip | `img[::-1]` |
| 90° rotate | `np.rot90(img)` |
| Center crop | `img[h//4:3*h//4, w//4:3*w//4]` |

### Nearest-neighbor resize — with broadcasting

```python
new_h, new_w = 50, 50
row_idx = (np.arange(new_h) * h / new_h).astype(int)
col_idx = (np.arange(new_w) * w / new_w).astype(int)
resized = img[row_idx][:, col_idx]    # advanced indexing
```

### Thresholding and histogram

```python
binary = (gray > 128).astype(np.uint8) * 255
hist, _ = np.histogram(gray, bins=8, range=(0, 256))
```

> 💡 Every image operation — resize, crop, rotate, threshold — is a NumPy expression. This is why computer vision libraries (OpenCV, PIL, scikit-image) all interoperate through NumPy arrays.

---

## 4️⃣ Monte Carlo Simulation

Three real applications of random simulation in one script.

### π estimation

```python
x, y = rng.random(n), rng.random(n)
inside = (x**2 + y**2) <= 1
pi_est = 4 * inside.mean()
```

**Convergence:** error shrinks as `1/√n` — quadrupling samples halves error.

| n | Error (approx) |
|---|---|
| 1 000 | ~0.03 |
| 10 000 | ~0.01 |
| 100 000 | ~0.003 |
| 1 000 000 | ~0.001 |

### Option pricing (Black–Scholes via Monte Carlo)

```python
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
call_payoff = np.maximum(ST - K, 0)
call_price  = np.exp(-r * T) * call_payoff.mean()
se          = call_payoff.std() / np.sqrt(n)      # standard error
```

Key tricks:

- `rng.standard_normal(n)` for `z`
- `np.maximum(ST - K, 0)` for payoff
- **Discount** back by `exp(-rT)`
- Report **standard error** `σ / √n` — crucial for interpreting a Monte Carlo result

### Random walk

```python
steps = rng.choice([-1, 1], size=(n_paths, n_steps))
paths = np.cumsum(steps, axis=1)
final = paths[:, -1]
```

### Portfolio Value-at-Risk (VaR) with correlated assets

The key step is **generating correlated normals via Cholesky**:

```python
corr = np.array([[1, .3, .2], [.3, 1, .4], [.2, .4, 1]])
L = np.linalg.cholesky(corr)

z = rng.standard_normal((n, 3))
returns = z @ L.T         # ← correlated normals
```

Then scale and weight:

```python
returns *= vols
returns += drift
port_returns = returns @ weights

var_95 = np.percentile(port_returns, 5)
cvar_95 = port_returns[port_returns <= var_95].mean()
```

> 💡 **`L.T` matters** — `z @ L.T` produces rows with covariance `L·Lᵀ = corr`. Getting this transposed backwards is a **classic Monte Carlo bug**.

---

## 5️⃣ Mini Project — Data Analyzer Class

A small but complete **pipeline** wrapping NumPy operations.

### Dataset

```python
{
    'age':       int array,
    'income':    float array,
    'score':     float array,
    'category':  string array,
    'purchased': 0/1 int array,
}
```

### API

| Method | Purpose |
|---|---|
| `generate_dataset(n)` | Synthetic data via `rng` |
| `summary()` | mean / std / quartiles per numeric column |
| `filter(mask)` | boolean mask → new `DataAnalyzer` |
| `groupby(col, value_col)` | per-group count, mean, std |
| `correlation_matrix()` | pairwise correlation of numeric columns |
| `find_outliers(col, method)` | IQR or z-score |

### Summary statistics

```python
for col, values in data.items():
    if values.dtype.kind in 'if':       # numeric
        print(values.mean(), values.std())
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
```

### Group-by without `pandas`

```python
def groupby(self, group_col, value_col):
    groups, values = self.data[group_col], self.data[value_col]
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        result[g] = {
            'count': mask.sum(),
            'mean':  values[mask].mean(),
            'std':   values[mask].std(),
        }
    return result
```

### Correlation matrix

```python
numeric_cols = {k: v for k, v in data.items() if v.dtype.kind in 'if'}
matrix = np.column_stack(list(numeric_cols.values()))
corr = np.corrcoef(matrix.T)
```

### Outlier detection

**IQR method** (robust to non-normal data):

```python
q1, q3 = np.percentile(values, [25, 75])
iqr    = q3 - q1
outliers = (values < q1 - 1.5*iqr) | (values > q3 + 1.5*iqr)
```

**Z-score method** (for roughly Gaussian data):

```python
outliers = np.abs((values - values.mean()) / values.std()) > 3
```

### Chained filtering

```python
mask = (data['age'] > 50) & (data['purchased'] == 1)
subset = analyzer.filter(mask)
```

> 💡 This mini project shows how far NumPy alone can go — you can build a real analysis pipeline without pandas. The key concepts are: boolean masks, `np.unique` + masks for group-by, `np.percentile` for quantiles, and `np.corrcoef` for correlations.

---

## 📊 Quick Reference — NumPy Pattern Cheatsheet

| Task | Pattern |
|---|---|
| Per-row stats | `arr.mean(axis=1)` |
| Per-column stats | `arr.mean(axis=0)` |
| Top-k by metric | `np.argsort(-metric)[:k]` |
| Multi-branch classification | `np.select` / nested `np.where` |
| Group-by aggregate | `for g in np.unique(col): mask = col == g` |
| Growth / rate of change | `np.diff(x) / x[:-1]` |
| Reshape to time chunks | `arr.reshape(n_chunks, chunk_size, …).sum(axis=1)` |
| Correlation matrix | `np.corrcoef(data.T)` |
| Grayscale image | `img @ [0.299, 0.587, 0.114]` |
| Crop / flip / rotate | slicing, `[::-1]`, `np.rot90` |
| Clamp to valid range | `np.clip(x, 0, 255)` |
| Correlated random vectors | `z @ np.linalg.cholesky(corr).T` |
| Monte Carlo estimate | `np.mean(f(samples))` |
| Standard error | `samples.std() / np.sqrt(n)` |
| Value at risk | `np.percentile(returns, 5)` |
| CVaR / expected shortfall | `returns[returns <= var].mean()` |
| Outliers (IQR) | `q1-1.5*iqr`, `q3+1.5*iqr` |
| Robust summary | `np.percentile(x, [25, 50, 75])` |

---

## ✅ Takeaways

1. **Almost every real-world problem becomes axis-wise aggregation + masking + broadcasting.**
2. **`axis=0` = columns, `axis=1` = rows** — this is the most reused mental model.
3. **Boolean masks + `np.unique`** replace `pandas.groupby` surprisingly well.
4. **Images are 3-D arrays** — grayscale is a matvec, resize is fancy indexing, crop is slicing.
5. **Monte Carlo is trivial in NumPy**: generate millions of samples, reduce with `mean` / `percentile` / `std`.
6. **Correlated samples** require Cholesky: `z @ L.T`.
7. **Always report standard error** (`σ/√n`) with Monte Carlo estimates — otherwise the number is meaningless.
8. **Outlier detection** has two standard methods: **IQR** (robust) and **z-score** (Gaussian assumption).
9. **Chained boolean masks** (`(a > x) & (b == y)`) express complex filters cleanly.
10. **`np.clip` before casting back to `uint8`** prevents overflow in image operations.

---
