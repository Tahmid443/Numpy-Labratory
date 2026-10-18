# NumPy — Reshaping and Transposing

Notes for **NumPy Laboratory → 06. Reshaping and Transposing**, covering how to change an array's shape without changing its data — `reshape`, `ravel`/`flatten`, `transpose`/`.T`, `squeeze`/`expand_dims`, and `concatenate`/`stack`/`split`.

---

## 📁 Project Structure

```
06.Reshaping and Transposing/
├── 01_reshape.py
├── 02_flatten_and_ravel.py
├── 03_transpose_and_T.py
├── 04_squeeze_and_expand_dims.py
├── 05_concatenate_stack_split.py
└── 06_shape_transformation_challenges.py
```

---

## 1️⃣ `reshape` — Change Shape, Keep Data

```python
a = np.arange(12)

a.reshape(3, 4)        # 3×4
a.reshape(2, 3, 2)     # 2×3×2 (3-D)
a.reshape(-1)          # flatten to 1-D
```

### Using `-1` to infer a dimension

```python
a.reshape(4, -1).shape    # (4, 3)
a.reshape(-1, 6).shape    # (2, 6)
a.reshape(2, 3, -1).shape # (2, 3, 2)
```

> 💡 Only **one** `-1` is allowed per call — NumPy infers it from `size ÷ other_dims`.

### View vs copy

```python
f = np.arange(6)
g = f.reshape(2, 3)
g[0, 0] = 99
# f is modified too — reshape returned a VIEW
```

- If the array is **contiguous**, `reshape` returns a **view** (free).
- If the array is **non-contiguous** (e.g. a slice with a step), `reshape` must return a **copy** — so mutations won't propagate.

### Row-major vs column-major reshape

```python
a.reshape(3, 4)             # C order (default)
a.reshape(3, 4, order='F')  # F order
```

---

## 2️⃣ `ravel` vs `flatten` — Going 1-D

| | `ravel()` | `flatten()` |
|---|---|---|
| Return | **View** when possible | **Always a copy** |
| Speed | Fast | Slower (allocates) |
| Safe to mutate? | Modifies original | Independent |

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

r = a.ravel()       # view
r[0] = 999          # a is modified

f = a.flatten()     # copy
f[0] = 999          # a unchanged
```

### Order parameter

```python
c = np.array([[1, 2], [3, 4]])
c.ravel()               # [1 2 3 4]     (row-major)
c.ravel(order='F')      # [1 3 2 4]     (column-major)
```

### Performance

```python
# 1000 iterations on a 1000×1000 array
big.ravel()     # ~ms    (view, no copy)
big.flatten()   # slower (allocates each call)
```

> 💡 **Rule of thumb:** use `ravel` unless you need a guaranteed independent copy.

---

## 3️⃣ `transpose` and `.T`

For 2-D arrays, transposing **swaps rows and columns**.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
a.T                                     # shape (3, 2)
np.transpose(a)                         # same as a.T
```

> ⚠️ **`.T` is a view** — it swaps strides, doesn't move data. Mutating `a.T` mutates `a`.

### Higher dimensions

```python
c = np.arange(24).reshape(2, 3, 4)
c.T.shape             # (4, 3, 2) — reverses ALL axes
c.transpose(2, 0, 1).shape  # (4, 2, 3) — custom permutation
c.swapaxes(0, 2).shape      # (4, 3, 2) — swap two specific axes
np.moveaxis(c, 0, -1).shape # (3, 4, 2) — move an axis to a new position
```

### Practical uses

```python
# Matrix multiplication
A @ B.T

# Check symmetry
np.array_equal(S, S.T)

# Deep learning: (N, H, W, C) → (N, C, H, W)
images.transpose(0, 3, 1, 2)
```

---

## 4️⃣ `squeeze` and `expand_dims` — Add / Remove Length-1 Axes

### `expand_dims` — insert an axis

```python
a = np.array([1, 2, 3])           # shape (3,)

np.expand_dims(a, 0).shape        # (1, 3)
np.expand_dims(a, 1).shape        # (3, 1)
np.expand_dims(a, -1).shape       # (3, 1)
```

Equivalent to `a[np.newaxis, :]` and `a[:, np.newaxis]`.

### `squeeze` — remove length-1 axes

```python
b = np.array([[[1, 2, 3]]])       # shape (1, 1, 3)
np.squeeze(b).shape                # (3,)

c = np.zeros((1, 3, 1, 4, 1))     # shape (1, 3, 1, 4, 1)
np.squeeze(c).shape                # (3, 4)

np.squeeze(c, axis=0).shape        # (3, 1, 4, 1) — only remove axis 0
np.squeeze(c, axis=2).shape        # (1, 3, 4, 1)
```

> ⚠️ `squeeze(axis=k)` raises `ValueError` if `shape[k] != 1`. Only size-1 axes can be squeezed.

### `atleast_*` — guarantee a minimum ndim

```python
s = np.array(5)                    # 0-D

np.atleast_1d(s).shape             # (1,)
np.atleast_2d(s).shape             # (1, 1)
np.atleast_3d(s).shape             # (1, 1, 1)

v = np.array([1, 2, 3])            # (3,)
np.atleast_2d(v)                   # [[1, 2, 3]] shape (1, 3)
np.atleast_3d(v)                   # shape (1, 3, 1)
```

`atleast_*` accepts **multiple arrays** and returns a tuple of promoted arrays.

---

## 5️⃣ `concatenate`, `stack`, and `split`

### `concatenate` — join along an **existing** axis

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate([a, b])                  # [1 2 3 4 5 6]

m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

np.concatenate([m1, m2], axis=0)        # vertical stack
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

np.concatenate([m1, m2], axis=1)        # horizontal stack
# [[1 2 5 6]
#  [3 4 7 8]]
```

### `stack` — join along a **new** axis

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.stack([a, b]).shape         # (2, 3)   — axis 0 (default)
np.stack([a, b], axis=1).shape # (3, 2)
```

### Helper stacks

| Function | Effect |
|---|---|
| `np.vstack([a, b])` | Concatenate on axis 0 |
| `np.hstack([a, b])` | Concatenate on axis 1 (or 0 for 1-D) |
| `np.dstack([a, b])` | Stack along axis 2 |
| `np.column_stack([a, b])` | Stack 1-D arrays as columns |
| `np.row_stack([a, b])` | Same as `vstack` |

### Split

```python
c = np.arange(12)

np.split(c, 3)              # → three equal parts, shape (4,) each
np.split(c, [2, 5, 8])      # → [0:2], [2:5], [5:8], [8:]

m = np.arange(12).reshape(3, 4)
np.hsplit(m, 2)             # → two (3, 2) halves
np.vsplit(m, 3)             # → three (1, 4) strips
```

### `array_split` — uneven splits allowed

```python
np.split(np.arange(10), 3)          # ❌ ValueError (10 not divisible by 3)
np.array_split(np.arange(10), 3)    # ✅ → [0:4], [4:7], [7:]
```

### `append` — avoid in loops

```python
np.append(a, 4)          # returns a new array
np.append(a, [4, 5])
```

> ⚠️ `np.append` **allocates a new array** each call — never use it inside a loop. Prefer `list.append` + final `np.array(...)`, or `np.concatenate` once.

---

## 6️⃣ Practice Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | 12-element 1-D → 3×4 | `arange(12).reshape(3, 4)` |
| 2 | Flatten 2-D in F-order | `m.ravel(order='F')` |
| 3 | `(2,3,4)` → `(6,4)` | `a.reshape(6, 4)` |
| 4 | Stack two 1-D as columns | `np.column_stack([a, b])` |
| 5 | Split 1-D into chunks of 4 | `np.split(a, 5)` (20 ÷ 5) |
| 6 | Remove length-1 axes | `np.squeeze(a)` |
| 7 | `(N,H,W,C)` → `(N,C,H,W)` | `images.transpose(0, 3, 1, 2)` |
| 8 | Tile 1-D 3 times | `np.tile(a, 3)` |

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| Change shape | `reshape(..., -1)` |
| Flatten (view) | `ravel()` |
| Flatten (copy) | `flatten()` |
| Transpose | `.T`, `np.transpose`, `swapaxes`, `moveaxis` |
| Insert axis | `np.expand_dims(arr, k)` / `arr[np.newaxis, ...]` |
| Remove length-1 axes | `np.squeeze(arr)` |
| Guarantee min ndim | `np.atleast_1d/2d/3d` |
| Join existing axis | `np.concatenate` |
| Join new axis | `np.stack`, `vstack`, `hstack`, `dstack` |
| Split | `np.split`, `array_split`, `hsplit`, `vsplit` |
| Repeat | `np.tile`, `np.repeat` |

---

## ✅ Takeaways

1. `reshape` is **free when contiguous** (returns a view); otherwise it copies.
2. `-1` infers one dimension — only **one** per call.
3. `ravel` → **view** if possible; `flatten` → **always a copy**.
4. `.T` **doesn't move data** — it swaps strides. Mutating `a.T` mutates `a`.
5. `expand_dims` / `squeeze` are the idiomatic axis manipulators; use `keepdims=True` to prevent the need to squeeze in the first place.
6. `concatenate` uses existing axes; `stack` adds a new one.
7. Prefer `array_split` over `split` when the size isn't divisible.
8. Never use `np.append` inside a loop — allocate once instead.

---

