# NumPy — Indexing, Slicing & Masking

Notes for **NumPy Laboratory → 03. Indexing, Slicing and Masking**, covering how to access, modify, and filter array elements using basic indexing, slices, boolean masks, fancy indexing, and conditional selection.

---

## 📁 Project Structure

```
03.Indexing Slicing and Masking/
├── 01_basic_indexing.py
├── 02_slicing_1d_arrays.py
├── 03_slicing_2d_and_3d_arrays.py
├── 04_boolean_indexing.py
├── 05_fancy_indexing.py
├── 06_conditional_selection.py
└── 07_indexing_challenges.py
```

---

## 1️⃣ Basic Indexing

### 1-D arrays

```python
a = np.array([10, 20, 30, 40, 50])
a[0]     # 10
a[2]     # 30
a[-1]    # 50  (negative = from end)
a[1] = 99  # mutate in place
```

### 2-D arrays

```python
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b[0, 0]   # 1
b[1, 2]   # 6
b[-1, -1] # 9

b[0]       # first row → [1, 2, 3]
b[:, 1]    # second column → [2, 5, 8]
b[-1, :]   # last row
```

> 💡 Prefer `b[i, j]` over `b[i][j]` — the chained form creates an intermediate view and is **less efficient**.

### 3-D arrays

```python
c = np.arange(24).reshape(2, 3, 4)
c[0]        # first 2-D slice, shape (3, 4)
c[0, 1, 2]  # scalar
c[1, -1, :] # last row of last 2-D slice
```

---

## 2️⃣ Slicing 1-D Arrays

**Syntax:** `a[start:stop:step]` — start inclusive, stop exclusive.

```python
a = np.arange(10)
a[2:5]    # [2 3 4]
a[:3]     # [0 1 2]
a[7:]     # [7 8 9]
a[::2]    # [0 2 4 6 8]
a[1::3]   # [1 4 7]
a[::-1]   # reversed
a[-3:]    # last three
a[:-3]    # all but last three
```

### ⚠️ Slices are **views**, not copies

```python
view = a[2:5]
view[0] = 999
# a is modified too!
```

Use `.copy()` for an independent array:

```python
safe = a[2:5].copy()
safe[0] = 999   # a unchanged
```

### Slice assignment

```python
c[2:5] = 0        # set a range
d[::2] = -1       # set every other element
```

---

## 3️⃣ Slicing 2-D and 3-D Arrays

For multi-dim arrays, slicing works **per axis**:

```python
a = np.arange(1, 13).reshape(3, 4)

a[0:2, 1:3]     # 2x2 block
a[::2, ::2]     # every other row and column
a[:, 1]         # column 1
a[1, :]         # row 1
a[:, -1]        # last column
a[::-1, :]      # flip rows
a[::-1, ::-1]   # flip both axes (180° rotation)
```

### 3-D slicing

```python
b = np.arange(24).reshape(2, 3, 4)
b[0]           # shape (3, 4)
b[:, 0]        # shape (2, 4) — first "row" of each 2-D slice
b[:, :, 1]     # shape (2, 3) — second column of each 2-D slice
b[1, 1:, :]    # shape (2, 4)
```

### Slice assignment on rows/cols

```python
c = np.zeros((3, 3))
c[1, :] = 5   # fill row 1
c[:, 2] = 7   # fill column 2
```

---

## 4️⃣ Boolean Indexing (Masking)

A **boolean mask** is an array of `True`/`False` selecting matching elements.

```python
a = np.array([1, 5, 2, 8, 3, 9, 4])

mask = a > 4           # [F T F T F T F]
a[mask]                # [5 8 9]
a[a > 4]               # same, directly
```

### Combining conditions

| Operator | Use |
|---|---|
| `&` | AND |
| `\|` | OR |
| `~` | NOT |

```python
a[(a > 2) & (a < 8)]   # [5 3 4]
a[(a < 3) | (a > 8)]   # [1 2 9]
a[~(a > 4)]            # [1 2 3 4]
```

> ⚠️ Always wrap conditions in **parentheses** — operator precedence would otherwise break them.

### 2-D masking

```python
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b[b > 5]           # [6 7 8 9]  (flattened 1-D result)
b[b % 2 == 0]      # [2 4 6 8]
```

### Boolean assignment

```python
c[c > 4] = 0       # replace matching elements
```

### Aggregations on masks

```python
(a > 4).sum()   # count of True values
(a > 8).any()   # at least one True
(a > 0).all()   # all True
np.where(a > 4)[0]  # indices of matches
```

---

## 5️⃣ Fancy Indexing

Index with **arrays/lists of integers** — not slices.

```python
a = np.array([10, 20, 30, 40, 50])

a[[0, 2, 4]]          # [10 30 50]
a[[4, 0, 2]]          # order preserved
a[[-1, 0]]            # negatives allowed
a[[0, 0, 1, 1, 1]]    # repeated indices allowed
```

### 2-D fancy indexing

```python
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

b[[0, 2]]              # rows 0 and 2
b[[0, 2], [1, 0]]      # pairs: (0,1) and (2,0) → [2 7]
```

For **cross-product** of row/column indices, use `np.ix_`:

```python
b[np.ix_([0, 2], [0, 2])]
# [[1 3]
#  [7 9]]
```

### Fancy assignment

```python
c[[1, 3]] = 0
d[[0, 1, 4]] = [999, 888, 777]
```

> ⚠️ **Fancy indexing returns a copy**, not a view. Mutating it won't affect the original.

---

## 6️⃣ Conditional Selection

| Function | Purpose |
|---|---|
| `np.where(cond, x, y)` | Vectorized if/else |
| `np.select(conds, choices)` | Multi-condition selection |
| `np.choose(indices, choices)` | Index-driven choice |
| `np.clip(a, lo, hi)` | Clamp values to range |
| `np.isin(a, values)` | Membership mask |
| `np.piecewise(x, conds, funcs)` | Piecewise functions |

### `np.where` — vectorized if/else

```python
np.where(a > 5, 100, 0)          # 100 if >5 else 0
np.where(a > 5, 100, np.where(a > 2, 50, 0))  # nested
```

### `np.select` — multiple conditions

```python
conditions = [a < 3, a < 6, a < 9]
choices    = ['low', 'medium', 'high']
np.select(conditions, choices, default='very high')
```

### `np.clip` — clamp to a range

```python
np.clip(a, 3, 7)   # values < 3 → 3, > 7 → 7
```

### `np.isin` — membership filter

```python
mask = np.isin(b, [3, 4, 5])
b[mask]
```

### `np.piecewise` — different function per region

```python
x = np.linspace(-2, 2, 9)
y = np.piecewise(x, [x < 0, x >= 0], [lambda x: x**2, lambda x: x**3])
```

---

## 7️⃣ Practice Challenges

| # | Task | Solution |
|---|---|---|
| 1 | Even numbers 0–19 | `a[a % 2 == 0]` |
| 2 | Values > 4 in 3×3 matrix | `m[m > 4]` |
| 3 | First and last column | `m[:, [0, -1]]` |
| 4 | Reverse each row | `m[:, ::-1]` |
| 5 | Middle 3×3 of 5×5 | `m[1:4, 1:4]` |
| 6 | Replace negatives with 0 | `a[a < 0] = 0` |
| 7 | 3 random unique indices | `a[rng.choice(100, 3, replace=False)]` |
| 8 | Anti-diagonal of 4×4 | `m[::-1, :][np.eye(4, dtype=bool)]` |

---

## 📊 Quick Reference

| Technique | Example | Returns |
|---|---|---|
| Basic index | `a[2]`, `b[1, 2]` | Scalar / view |
| Slice | `a[2:5]`, `b[:, 1]` | **View** |
| Boolean mask | `a[a > 4]` | **Copy** |
| Fancy index | `a[[0, 2, 4]]` | **Copy** |
| `np.where` | `np.where(a > 4, 1, 0)` | New array |
| `np.select` | Multi-cond selection | New array |
| `np.isin` | Membership | Boolean mask |
| `np.clip` | Range clamp | New array |

---

## ✅ Takeaways

1. **Basic indexing and slicing produce views** — mutating them mutates the original. Use `.copy()` when you need independence.
2. **Boolean and fancy indexing produce copies.**
3. Combine conditions with `&`, `|`, `~` — **always parenthesize**.
4. `np.where` is the vectorized if/else; `np.select` handles many branches.
5. `np.ix_` builds the **cross-product** for fancy row/column selection.
6. On masks: `.sum()`, `.any()`, `.all()` give fast aggregate info.

---
