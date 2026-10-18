# NumPy — Revision and Assessment

Notes for **NumPy Laboratory → 15. Revision and Assessment**, consolidating everything from the course through topic-wise questions, output prediction, debugging drills, a mixed problem set, and a final graded assessment.

---

## 📁 Project Structure

```
15.Revision and Assessment/
├── 01_topic_wise_questions.py
├── 02_output_prediction.py
├── 03_debugging_exercises.py
├── 04_mixed_problem_set.py
└── 05_final_assessment.py
```

---

## 1️⃣ Topic-Wise Questions — the 15 Essentials

| # | Topic | Question | Answer |
|---|---|---|---|
| 1 | Creation | 3×3 identity | `np.eye(3)` |
| 2 | Indexing | Every other from index 1 | `a[1::2]` |
| 3 | Broadcasting | Add `(3,)` to each row of `(4, 3)` | `M + v` |
| 4 | Reductions | Per-column mean of `(4, 5)` | `M.mean(axis=0)` |
| 5 | Reshape | `arange(12) → (2, 2, 3)` | `a.reshape(2, 2, 3)` |
| 6 | Boolean mask | Values strictly between 3 and 7 | `a[(a > 3) & (a < 7)]` |
| 7 | `where` | Replace values > 5 with 0 | `np.where(a > 5, 0, a)` |
| 8 | Linalg | Solve `2x+y=5`, `x+3y=7` | `np.linalg.solve(A, b)` |
| 9 | Axis | Row sums of `(3, 4)` | `M.sum(axis=1)` |
| 10 | Argmax | Max index per row | `M.argmax(axis=1)` |
| 11 | Transpose | Swap axes of `(3, 4)` | `M.T.shape` → `(4, 3)` |
| 12 | Random | 100 standard normals, seed 42 | `default_rng(42).standard_normal(100)` |
| 13 | Cumulative | Running sum | `np.cumsum(a)` |
| 14 | NaN | Mean ignoring NaN | `np.nanmean(a)` |
| 15 | Stack | Three `(3,)` → `(3, 3)` | `np.stack([a, b, c])` |

> 💡 If you can answer all 15 from memory, you have the working vocabulary of NumPy.

---

## 2️⃣ Output Prediction — Mental Execution Drills

For `a = [1, 2, 3, 4, 5]`, `b = [[1, 2], [3, 4]]`, `m = arange(6).reshape(2, 3)`:

| Expression | Result | Rule to remember |
|---|---|---|
| `a[2]` | `3` | 0-based |
| `a[-1]` | `5` | negative wraps |
| `a[1:3]` | `[2, 3]` | stop exclusive |
| `a[::2]` | `[1, 3, 5]` | step 2 |
| `a[::-1]` | `[5, 4, 3, 2, 1]` | reverse |
| `a[a > 2]` | `[3, 4, 5]` | mask filter |
| `a * 2` | `[2, 4, 6, 8, 10]` | element-wise |
| `a.sum()` | `15` | scalar reduce |
| `a.mean()` | `3.0` | float result |
| `a.argmax()` | `4` | index of max |
| `b[0, 1]` | `2` | `[row, col]` |
| `b[:, 0]` | `[1, 3]` | column 0 |
| `b.sum(axis=0)` | `[4, 6]` | **down columns** |
| `b.sum(axis=1)` | `[3, 7]` | **across rows** |
| `b.T` | `[[1, 3], [2, 4]]` | swap axes |
| `m.shape` | `(2, 3)` | tuple |
| `m.sum()` | `15` | total |
| `m.sum(axis=1)` | `[3, 12]` | row sums |
| `m[0]` | `[0, 1, 2]` | first row |
| `m[:, 1]` | `[1, 4]` | column 1 |

### The single most-asked axis rule

> **`axis=0` collapses rows → result has one value per column.**  
> **`axis=1` collapses columns → result has one value per row.**

---

## 3️⃣ Debugging Exercises — Common Pitfalls

### ❌ Bug 1 — Incompatible broadcasting

```python
np.ones((3, 4)) + np.ones((3, 3))       # ValueError
```

**Fix:** make last dims match → `np.ones((3, 4))`.

### ❌ Bug 2 — Fancy indexing returns a copy

```python
b = a[[1, 2, 3]]
b[0] = 99                                # a unchanged!
```

**Fix:** assign to the original — `a[[1, 2, 3]] = [20, 30, 40]`.

### ❌ Bug 3 — Comparing to NaN

```python
a == np.nan                              # always False
```

**Fix:** `np.isnan(a)`.

### ❌ Bug 4 — Wrong axis for reduction

```python
m.mean(axis=1)                           # row means
                                         # — wanted column means
m.mean(axis=0)                           # ✅
```

### ❌ Bug 5 — Integer division surprise

```python
a / 2      # float  ✅
a // 2     # int    (truncated)
```

Remember: `/` is true division (`float` result); `//` is floor division.

### ❌ Bug 6 — `reshape` returns a view

```python
b = a.reshape(2, 3)
b[0, 0] = 99                             # a is modified!
```

**Fix:** `a.copy().reshape(2, 3)`.

### ❌ Bug 7 — `*` vs `@`

```python
A * B      # element-wise
A @ B      # matrix product
```

### ❌ Bug 8 — Chained comparison

```python
a > 2 < 6          # parses as (a > 2) < 6  → nonsense
```

**Fix:** `(a > 2) & (a < 6)` — always parenthesize each condition.

---

## 4️⃣ Mixed Problem Set — the 15 Key Patterns

| # | Problem | Idiom |
|---|---|---|
| 1 | Set every 3rd of 20 zeros to 5 | `a[::3] = 5` |
| 2 | Normalize to sum 1 | `a / a.sum()` |
| 3 | Chain matmul | `A @ B @ C` |
| 4 | Drop NaN & stats | `clean = a[~np.isnan(a)]` |
| 5 | One-hot encode | `np.eye(n)[labels]` |
| 6 | Cumulative product of non-zeros | `np.cumprod(a[a != 0])` |
| 7 | Top-3 rows by sum | `np.argsort(-m.sum(1))[:3]` |
| 8 | Moving average | `np.convolve(a, ones/k, 'valid')` |
| 9 | Pairwise distances | `pts[:, None] - pts[None]` |
| 10 | One-hot decode | `one_hot.argmax(axis=1)` |
| 11 | Diagonal matrix | `np.diag(v)` |
| 12 | Lower triangle | `np.tril(m)` |
| 13 | Z-scores per column | `(m - m.mean(0)) / m.std(0)` |
| 14 | Rolling max | `sliding_window_view(a, 3).max(-1)` |
| 15 | Sign-change indices | `np.where(np.diff(np.sign(a)) != 0)[0]` |

### Key tricks worth memorizing

- **`np.sign` + `np.diff` + `np.where`** — find sign changes.
- **`sliding_window_view`** — the modern, allocation-friendly alternative to convolve for rolling stats.
- **`np.eye(n)[labels]`** — the canonical one-hot.
- **`pts[:, None, :] - pts[None, :, :]`** — the canonical pairwise difference.
- **`a / a.sum()`** — normalize to a probability distribution.

---

## 5️⃣ Final Assessment — What's Covered

The assessment checks **10 skills**, one or two per major topic:

| Section | Skill | Verification |
|---|---|---|
| 1.1 | Identity matrix | `shape == (4, 4)`, `det ≈ 1` |
| 1.2 | `arange` with step | equals `[5, 10, 15, 20, 25]` |
| 2.1 | Diagonal extraction | `m.diagonal()` |
| 2.2 | Boolean filtering | evens from `1..10` |
| 3.1 | Broadcasting multiplication table | `outer(1..9, 1..9)` |
| 4.1 | Row sums, column means, max | `axis=1`, `axis=0`, `.max()` |
| 5.1 | Reshape + transpose | `(6, 2)`, `[0, 1] == 2` |
| 6.1 | Solve linear system | `x=1, y=2` |
| 7.1 | Random with seed | length 1000, mean ≈ 0 |
| 8.1 | Float comparison | `np.isclose(0.1+0.2, 0.3)` |

### Scoring interpretation

| Score | Message |
|---|---|
| 10/10 | 🎉 Perfect — NumPy mastered |
| ≥ 8/10 | 👍 Excellent — review missed topics |
| ≥ 6/10 | 📚 Good — target weak areas |
| < 6/10 | 📖 Keep practicing |

---

## 📊 Comprehensive Cheatsheet

### Creation

```python
np.array(seq)          np.zeros(shape)        np.ones(shape)
np.full(shape, v)      np.empty(shape)        np.eye(n)
np.arange(start, stop, step)                  np.linspace(start, stop, n)
```

### Attributes

```python
a.shape    a.ndim    a.size    a.dtype    a.itemsize    a.nbytes
```

### Indexing

```python
a[i]        a[i, j]          a[:, k]         a[i, :]
a[a > x]    a[[i, j]]        a[np.newaxis, :]  a[:, np.newaxis]
```

### Reshaping

```python
a.reshape(...)    a.ravel()    a.flatten()    a.T
np.expand_dims    np.squeeze   np.concatenate   np.stack
```

### Reductions

```python
a.sum()     a.mean()      a.std()        a.var()
a.min()     a.max()       a.argmax()     a.argmin()
a.cumsum()  np.diff(a)    np.percentile(a, q)
np.nanmean  np.nansum     np.nanstd
```

### Arithmetic & logic

```python
a + b     a * b     a ** 2     a / b    a // b    a % b
(a > x) & (a < y)     ~(a > x)     (a < x) | (a > y)
np.where(cond, x, y)  np.clip(a, lo, hi)  np.select(...)
```

### Linear algebra

```python
A @ B           np.dot(A, B)        np.linalg.solve(A, b)
np.linalg.inv   np.linalg.det       np.linalg.eig      np.linalg.eigh
np.linalg.svd   np.linalg.qr        np.linalg.norm     np.linalg.cond
```

### Random

```python
rng = np.random.default_rng(seed)
rng.random(n)           rng.integers(lo, hi, n)
rng.normal(loc, scale)  rng.choice(a, k, replace=False)
rng.shuffle(a)          rng.permutation(a)
```

### Numerical

```python
np.isclose(a, b)     np.allclose(a, b)
np.isnan(a)          np.isinf(a)        np.isfinite(a)
np.finfo(np.float64).eps               np.seterr / np.errstate
```

---

## ✅ Final Takeaways

1. **`axis=k` collapses axis `k`** — this single idea powers 80% of NumPy work.
2. **Broadcasting = right-aligned shape matching** — use `newaxis` to fix mismatches.
3. **Slicing → view; fancy/boolean → copy** — check `.base` or `np.shares_memory`.
4. **`@` is matmul, `*` is element-wise** — the most impactful Python/NumPy gotcha.
5. **Always `np.isclose` for float comparison** — `==` on floats is a bug.
6. **NaN poisons reductions** — use `np.nan*` variants.
7. **Vectorization is 10–10 000× faster** than Python loops for arrays.
8. **`np.eye(n)[labels]`** for one-hot, **`a[:, None] op b`** for outer/pairwise, **`np.sign` + `np.diff`** for sign changes — memorize the three idioms.
9. **Correlated samples require Cholesky** — `z @ L.T`.
10. **When debugging arrays, always print `.shape`, `.dtype`, and a small sample** — 90% of bugs show up immediately.

---

## 🎓 You've Completed the NumPy Laboratory

You've covered:

- Array creation, metadata, dtype, strides
- Indexing, slicing, masking, fancy indexing
- Vectorized operations and ufuncs
- Broadcasting rules and patterns
- Reshaping, transposing, concatenating, splitting
- Reductions, statistics, cumulative operations
- Random generation and simulation
- Matrix multiplication and linear algebra
- Memory layout, views, copies, contiguity
- Numerical precision, overflow, NaN/inf
- Vectorization vs. Python loops
- Real-world integrated projects

### Recommended next steps

- 📊 **pandas** — labeled data, `DataFrame`, join/groupby/merge
- 🔬 **SciPy** — signal processing, optimization, stats
- 🤖 **scikit-learn** — ML pipelines on top of NumPy
- ⚡ **Numba / Cython** — accelerate the remaining slow loops
- 🧠 **PyTorch / JAX** — NumPy-like tensor APIs for deep learning
- 🎨 **Matplotlib** — visualize the arrays you've built

---
