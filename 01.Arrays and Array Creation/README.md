# NumPy — Arrays and Array Creation

Notes for **NumPy Laboratory → 01. Arrays and Array Creation**, covering every common way to build a NumPy array — from Python sequences, filler functions, sequences (`arange`/`linspace`), and diagonal matrices.

---

## 📁 Project Structure

```
01.Arrays and Array Creation/
├── 01_array_from_list.py
├── 02_zeros_ones_full.py
├── 03_arange_and_linspace.py
├── 04_identity_and_diagonal.py
└── 05_array_creation_challenges.py
```

---

## 1️⃣ Creating Arrays from Python Sequences

```python
import numpy as np

np.array([1, 2, 3, 4, 5])          # from list
np.array((1, 2, 3))                # from tuple
np.array([[1, 2, 3], [4, 5, 6]])   # 2D from nested list → shape (2, 3)
np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D → shape (2, 2, 2)
np.array([1, 2, 3], dtype=np.float32)  # explicit dtype
np.array(range(10))                # from range
np.array("hello")                  # 0-D string array, dtype='<U5'
```

**Key points:**
- `np.array(...)` **infers `dtype`** from the input.
- Mixed types are **upcast** (e.g., `[1, 2.5, 3]` → `float64`).
- Nested lists create **multi-dimensional arrays** — `shape` and `ndim` reflect depth.
- Use `dtype=` to control memory and precision (`float32` vs `float64`).

---

## 2️⃣ Filled Arrays — `zeros`, `ones`, `full`, `empty`

| Function | Purpose | Example |
|---|---|---|
| `np.zeros(shape)` | All zeros | `np.zeros((2, 3))` |
| `np.ones(shape)` | All ones | `np.ones((3, 2), dtype=int)` |
| `np.full(shape, val)` | All = `val` | `np.full((2, 2), 3.14)` |
| `np.empty(shape)` | Uninitialized | `np.empty(3)` |

**`*_like` variants** — copy the shape **and** dtype of an existing array:

```python
template = np.array([[1, 2], [3, 4]], dtype=np.float32)

np.zeros_like(template)   # same shape, dtype=float32, zeros
np.ones_like(template)    # same shape, ones
np.full_like(template, 9) # same shape, filled with 9
```

> 💡 `np.empty` is **faster** but returns **arbitrary** memory contents — only use it if you will overwrite every element.

---

## 3️⃣ Sequences — `arange`, `linspace`, `logspace`

### `np.arange(start, stop, step)` — *step-based*

```python
np.arange(10)          # [0 1 2 3 4 5 6 7 8 9]
np.arange(2, 10)       # [2 3 4 5 6 7 8 9]
np.arange(0, 1, 0.1)   # [0.  0.1 0.2 ... 0.9]
np.arange(10, 0, -2)   # [10 8 6 4 2]
```

### `np.linspace(start, stop, num)` — *count-based*

```python
np.linspace(0, 1, 5)                  # [0.   0.25 0.5  0.75 1.  ]
np.linspace(0, 10, 11)                # [0. 1. 2. ... 10.]
np.linspace(0, 1, 5, endpoint=False)  # [0.   0.2 0.4 0.6 0.8]
```

### `np.logspace(start, stop, num, base=10)`

```python
np.logspace(0, 3, 4)           # [1.   10.  100. 1000.]
np.logspace(1, 3, 3, base=2)   # [2. 4. 8.]
```

### ⚖️ `arange` vs `linspace`

| | `arange` | `linspace` |
|---|---|---|
| Controlled by | **step size** | **number of points** |
| Endpoint | **excluded** by default | **included** by default |
| Best for | integer ranges, fixed step | evenly divided ranges, floats |

> ⚠️ Prefer `linspace` for floating-point ranges — `arange` with float steps can suffer from rounding errors.

---

## 4️⃣ Identity, Diagonal & Eye

### `np.identity(n)`
Square matrix with `1`s on the main diagonal.

```python
np.identity(3)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### `np.eye(n, m, k)` — flexible identity
- Supports **non-square** shapes and **diagonal offset** `k`.
- `k > 0` → superdiagonal; `k < 0` → subdiagonal.

```python
np.eye(3)       # 3x3 identity
np.eye(3, k=1)  # ones on superdiagonal
np.eye(3, k=-1) # ones on subdiagonal
np.eye(3, 4)    # 3x4 non-square
```

### `np.diag(v, k=0)` — dual purpose
- Pass a **1-D array** → returns a **2-D diagonal matrix**.
- Pass a **2-D array** → returns its **diagonal as 1-D**.

```python
np.diag([1, 2, 3, 4])                 # 4x4 diagonal matrix
np.diag(np.arange(9).reshape(3,3))    # extracts main diagonal
np.diag([1, 2, 3, 4], k=1)            # superdiagonal matrix
```

### `np.diagflat(list)` — flatten input into a diagonal matrix

```python
np.diagflat([[1, 2], [3, 4]])
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]
```

---

## 5️⃣ Practice Challenges

| # | Task | Solution |
|---|---|---|
| 1 | Array `[10, 20, 30, 40, 50]` | `np.array([10, 20, 30, 40, 50])` |
| 2 | 4×4 zeros matrix | `np.zeros((4, 4))` |
| 3 | Even numbers 0–20 | `np.arange(0, 21, 2)` |
| 4 | 5 evenly spaced 5→25 | `np.linspace(5, 25, 5)` |
| 5 | Identity 6×6 | `np.identity(6)` |
| 6 | 3×3 all sevens | `np.full((3, 3), 7)` |
| 7 | `[[1,2,3],[4,5,6]]` via arange + reshape | `np.arange(1, 7).reshape(2, 3)` |
| 8 | 5×5 checkerboard of 0/1 | `np.indices((5, 5)).sum(axis=0) % 2` |

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| From Python data | `np.array(seq)` |
| Zeros / ones / constant | `np.zeros`, `np.ones`, `np.full` |
| Match existing array | `np.zeros_like`, `np.ones_like`, `np.full_like` |
| Uninitialized | `np.empty` |
| Fixed step | `np.arange` |
| Fixed count | `np.linspace` |
| Log-spaced | `np.logspace` |
| Identity | `np.identity`, `np.eye` |
| Diagonal | `np.diag`, `np.diagflat` |

---

## ✅ Takeaways

1. `np.array(...)` is the universal entry point — dtype is inferred or set explicitly.
2. Use `zeros`, `ones`, `full` for fast, predictable initialization.
3. Choose `arange` when you know the **step**; `linspace` when you know the **count**.
4. `eye` handles offsets and non-square cases; `diag` both **builds** and **extracts** diagonals.
5. `*_like` helpers keep shape + dtype consistent with a template array.

---
