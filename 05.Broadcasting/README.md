# NumPy — Broadcasting

Notes for **NumPy Laboratory → 05. Broadcasting**, covering the rules that let NumPy combine arrays of different shapes, how to control it with `newaxis`, how to diagnose errors, and how it powers fast, loop-free numeric code.

---

## 📁 Project Structure

```
05.Broadcasting/
├── 01_broadcasting_rules.py
├── 02_scalar_and_array_broadcasting.py
├── 03_1d_and_2d_broadcasting.py
├── 04_broadcasting_with_newaxis.py
├── 05_broadcasting_errors.py
└── 06_broadcasting_challenges.py
```

---

## 1️⃣ Broadcasting Rules

Broadcasting lets NumPy operate on arrays of **different shapes** without copying data.

**The three rules:**

1. **Align from the right.** If arrays have different `ndim`, pad the smaller shape with `1`s **on the left**.
2. **Check compatibility.** Two dimensions are compatible if they are **equal** or **one of them is 1**.
3. **Expand.** Size-1 dimensions stretch to match the other array's size — no data is actually copied.

### Rule-by-rule examples

```python
# Same shape → element-wise
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b                    # [5 7 9]

# Scalar → broadcast to any shape
a + 10                   # [11 12 13]

# (3, 3) + (3,) → (3, 3)
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([10, 20, 30])
m + v
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]
```

### Column × Row → grid

```python
col = np.array([[1], [2], [3]])        # shape (3, 1)
row = np.array([[10, 20, 30, 40]])     # shape (1, 4)
(col + row).shape                       # (3, 4)
```

### Padding the shape on the left

```python
a = np.ones((2, 3, 4))

a + np.ones(4)            # (4,) → (1, 1, 4) → (2, 3, 4) ✅
a + np.ones((3, 4))       # (3, 4) → (1, 3, 4) → (2, 3, 4) ✅
a + np.ones((2, 1, 4))    # already compatible → (2, 3, 4) ✅
```

### Incompatible shapes fail

```python
np.ones((3, 4)) + np.ones((3, 3))
# ValueError: operands could not be broadcast together with shapes (3,4) (3,3)
```

> 💡 **Mental model:** stack shapes right-aligned; a dimension "works" if it's equal to the other or one of them is `1`.

---

## 2️⃣ Scalar & Array Broadcasting

A scalar is treated as a **0-D array** and broadcasts to any shape.

```python
a = np.array([1, 2, 3])
a + 5         # [6 7 8]
a * 3         # [3 6 9]
a ** 2        # [1 4 9]
10 / a        # [10. 5. 3.33]

m = np.arange(6).reshape(2, 3)
m + 100       # shape (2, 3)
m > 2         # boolean mask
```

### 3-D + scalar

```python
b = np.arange(24).reshape(2, 3, 4)
(b + 1).shape  # (2, 3, 4)
```

### dtype note

```python
a.dtype                 # int64
(a + 5).dtype           # int64     ← int + int = int
(a + 5.0).dtype         # float64   ← int + float = float
```

> ⚠️ Broadcasting can **upcast dtype** silently. Watch out when memory matters.

---

## 3️⃣ Broadcasting Between 1-D and 2-D

### Row-wise add

```python
m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([10, 20, 30])

m + v
# Each column j gets v[j] added to every row
```

### Column-wise add

Use a **column vector** with shape `(n, 1)`:

```python
col = np.array([[100], [200], [300]])   # shape (3, 1)
m + col
# Each row i gets col[i] added
```

### Outer product via broadcasting

```python
a = np.array([1, 2, 3])
b = np.array([10, 20])

outer = a[:, np.newaxis] * b   # shape (3, 2)
```

### Row-wise normalization (mean-centered)

```python
data = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

row_means = data.mean(axis=1, keepdims=True)  # shape (3, 1)
data - row_means                               # broadcast subtract
```

> 💡 `keepdims=True` is the idiomatic way to keep a reduction dimension so it broadcasts cleanly.

### Column-wise normalization

```python
col_means = data.mean(axis=0)   # shape (3,)
data - col_means                # broadcasts to (3, 3)
```

---

## 4️⃣ Controlling Broadcasting with `newaxis`

`np.newaxis` (alias for `None`) inserts a new axis of length 1 — the primary tool for **aligning mismatched shapes**.

```python
a = np.array([1, 2, 3])

a[:, np.newaxis].shape    # (3, 1)
a[np.newaxis, :].shape    # (1, 3)

np.expand_dims(a, 0).shape  # (1, 3)
np.expand_dims(a, 1).shape  # (3, 1)
```

### Outer product

```python
x = np.array([1, 2, 3])
y = np.array([10, 20])

outer = x[:, np.newaxis] * y[np.newaxis, :]   # (3, 2)
```

### Pairwise differences

```python
a = np.array([1, 5, 3])
b = np.array([2, 4])

a[:, np.newaxis] - b
# [[-1 -3]
#  [ 3  1]
#  [ 1 -1]]
```

### Pairwise distance matrix

```python
points = np.array([[1, 2], [4, 6], [7, 8]])

diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=-1))
# shape (3, 3) — symmetric with zeros on diagonal
```

### 3-way outer product

```python
a = np.arange(3)[:, np.newaxis, np.newaxis]   # (3, 1, 1)
b = np.arange(4)[np.newaxis, :, np.newaxis]   # (1, 4, 1)
c = np.arange(5)[np.newaxis, np.newaxis, :]   # (1, 1, 5)

result = a + b + c                             # (3, 4, 5)
```

---

## 5️⃣ Common Broadcasting Errors & Fixes

### ❌ Mismatched inner dimensions

```python
np.ones((2, 3)) + np.ones((2, 4))
# ValueError: shapes (2,3) and (2,4) not aligned
```

**Fix:** reshape so the last dims match, or use `newaxis`.

### ❌ 1-D arrays of different lengths

```python
np.ones(3) + np.ones(4)
# ValueError
```

**Fix (if you wanted outer product):**

```python
np.ones(3)[:, np.newaxis] + np.ones(4)   # → (3, 4)
```

### ❌ Forgot to add a column axis

```python
m = np.arange(6).reshape(2, 3)   # (2, 3)
col = np.array([10, 20])         # (2,)  ← wrong orientation
m + col                          # ValueError
```

**Fix:**

```python
m + col[:, np.newaxis]           # (2, 1) → (2, 3) ✅
```

### ❌ `np.where` with incompatible branches

```python
np.where(np.ones(3) > 0, np.ones(4), 0)
# ValueError — branch shapes must broadcast
```

**Fix:** make `x` and `y` broadcastable to the condition shape.

---

## 6️⃣ Practice Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | `(3,3) + (3,)` | right-aligned row broadcast |
| 2 | Pairwise Euclidean distances | `A[:, None, :] - B[None, :, :]` |
| 3 | Multiplication table 1–9 | `arange(1,10)[:,None] * arange(1,10)` |
| 4 | Standardize each column | `(m - m.mean(0)) / m.std(0)` |
| 5 | Cosine similarity matrix | normalize rows, then `n @ n.T` |
| 6 | 8×8 checkerboard | `(i[:,None] + j[None,:]) % 2` |
| 7 | Sum of outer product | `(a[:,None] * b).sum()` |
| 8 | Lower-triangular `i*j` | `np.where(i >= j, i*j, 0)` |

---

## 📊 Quick Reference

| Goal | Pattern |
|---|---|
| Add vector to each row | `M + v` (v shape `(cols,)`) |
| Add vector to each column | `M + v[:, np.newaxis]` |
| Outer product | `a[:, None] * b[None, :]` |
| Pairwise differences (1-D) | `a[:, None] - b` |
| Pairwise differences (N-D) | `A[:, None, :] - B[None, :, :]` |
| Keep reduced axis | `arr.mean(axis=k, keepdims=True)` |
| Insert an axis | `arr[np.newaxis, ...]`, `np.expand_dims` |
| Check compatibility | right-align shapes mentally |
| Diagnose an error | print both `.shape`s |

---

## ✅ Takeaways

1. **Broadcasting = implicit expansion of size-1 dimensions** — no data is copied.
2. Align shapes **right to left**; dimensions must be **equal or 1**.
3. `np.newaxis` / `np.expand_dims` is the standard tool for aligning axes.
4. `keepdims=True` keeps reductions broadcast-friendly (row-wise / column-wise normalization).
5. `a[:, None] op b` produces an **outer product** — the backbone of pairwise ops (distances, similarity matrices).
6. **Error debugging checklist:** print both shapes → check right-aligned alignment → add a `newaxis` where needed.

---
