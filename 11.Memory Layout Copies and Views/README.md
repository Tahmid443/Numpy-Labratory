# NumPy — Memory Layout, Copies & Views

Notes for **NumPy Laboratory → 11. Memory Layout, Copies and Views**, covering when NumPy returns views vs. copies, how assignment and `copy()` differ, how to detect shared memory, and why contiguity matters for performance.

---

## 📁 Project Structure

```
11.Memory Layout Copies and Views/
├── 01_views_vs_copies.py
├── 02_copy_and_deep_copy_behavior.py
├── 03_memory_sharing.py
├── 04_contiguous_arrays.py
└── 05_memory_behavior_challenges.py
```

---

## 1️⃣ Views vs. Copies

A **view** shares the underlying buffer with its source. A **copy** owns its own data.

### What returns a **view**

| Operation | View or copy? |
|---|---|
| Basic slicing `a[1:4]` | ✅ View |
| `a.reshape(...)` (when contiguous) | ✅ View |
| `a.T` / `a.transpose(...)` | ✅ View |
| `a.ravel()` (when contiguous) | ✅ View |
| `a.view(...)` | ✅ View (reinterpret) |

### What returns a **copy**

| Operation | View or copy? |
|---|---|
| Basic indexing of a single element `a[2]` | 🟡 Scalar copy (immutable) |
| **Fancy indexing** `a[[1, 3, 5]]` | ❌ Copy |
| **Boolean indexing** `a[a > 5]` | ❌ Copy |
| `a.flatten()` | ❌ Copy (always) |
| `a.copy()` / `np.copy(a)` | ❌ Copy (always) |
| `a.astype(...)` | ❌ Copy (unless dtype unchanged and `copy=False`) |

### Demonstrating the difference

```python
a = np.arange(10)

b = a[1:4]
b[0] = 99                     # ← a is modified (VIEW)

c = a[[1, 3, 5]]
c[0] = 99                     # ← a unchanged (COPY)

d = a[a > 5]
d[0] = 99                     # ← a unchanged (COPY)
```

### Detecting a view with `.base`

```python
a = np.arange(12)
b = a[:5]
c = a.copy()

b.base is a                   # True  — view
c.base is a                   # False — copy (base is None)
```

> 💡 **Rule of thumb:** *basic slicing → view; fancy/boolean → copy.* When in doubt, check `x.base` or use `np.shares_memory(x, y)`.

---

## 2️⃣ Copy and Deep-Copy Behavior

### Assignment `=` does **not** copy

```python
a = np.array([[1, 2], [3, 4]])
b = a
b[0, 0] = 99
# a is modified — a and b reference the SAME array
a is b                        # True
```

### `.copy()` — independent data

```python
c = a.copy()
c[0, 0] = 99
# a unchanged
np.shares_memory(a, c)        # False
```

### `copy.deepcopy` — matters for object arrays

For normal numeric arrays, `.copy()` and `copy.deepcopy()` behave the same. For **object arrays**, `.copy()` is shallow — the Python objects are still shared:

```python
obj = np.array([1, [2, 3], {'a': 1}], dtype=object)

shallow = obj.copy()
obj[1].append(99)             # ← mutates BOTH obj and shallow!

deep = copy.deepcopy(obj)     # independent nested objects
```

> ⚠️ If your array holds Python objects (lists, dicts), you need `copy.deepcopy` for full independence.

### `np.array(..., copy=...)`

```python
a = np.array([1, 2, 3])

np.array(a)                   # copy by default
np.array(a, copy=False)       # view — no copy if possible
np.array(a, copy=True)        # explicit copy
```

### `np.copy` — straightforward copy

```python
e = np.copy(a)
```

---

## 3️⃣ Detecting and Controlling Memory Sharing

### `np.shares_memory` vs. `np.may_share_memory`

```python
a = np.arange(5)
b = a[:3]
c = a.copy()

np.shares_memory(a, b)        # True  — definitely share
np.shares_memory(a, c)        # False

np.may_share_memory(a, b)     # True  — conservative over-approximation
```

- **`shares_memory`** — precise, but can be slow for large arrays (checks overlap).
- **`may_share_memory`** — fast, may return `True` even when they don't overlap.

### The `.base` chain

Views form a chain back to the original buffer:

```python
a = np.arange(24)
b = a.reshape(2, 3, 4)
c = b[0]
d = c[:, 1]

d.base is c                   # True
c.base is b                   # True
b.base is a                   # True
a.base is None                # True (a owns the data)
```

### Memory-mapped files

For very large arrays that don't fit in RAM, use `mmap_mode`:

```python
np.save('big.npy', arr)

loaded = np.load('big.npy', mmap_mode='r')   # lazy, OS-paged
```

- Returns a `memmap` array — reads on demand from disk.
- `mmap_mode='r'` (read-only), `'r+'` (read/write), `'w+'`, `'c'` (copy-on-write).

> 💡 `memmap` is the standard way to work with datasets larger than RAM in NumPy.

### Which operations share?

```python
a = np.arange(10)

np.shares_memory(a, a.reshape(2, 5))     # True  — reshape
np.shares_memory(a, a.astype(float))     # False — dtype change
np.shares_memory(a, a.view(np.float32))  # True  — reinterpret (same bytes)
```

---

## 4️⃣ Contiguous Arrays

A **contiguous** array stores its elements in **adjacent memory**, which enables cache-friendly access and lets BLAS / SIMD routines run at full speed.

### The `flags` dictionary

```python
a = np.arange(12).reshape(3, 4)

a.flags['C_CONTIGUOUS']      # True  — row-major, contiguous
a.flags['F_CONTIGUOUS']      # False
a.flags['OWNDATA']           # True  — owns its buffer
a.flags['WRITEABLE']         # True
```

### What breaks contiguity?

| Operation | Result |
|---|---|
| `a.T` | Not C-contiguous (but F-contiguous!) |
| `a[:, 1]` | Not contiguous (strided) |
| `a[::2]` | Not contiguous |
| `a.reshape(...)` on non-contiguous | Not contiguous (and copies) |

```python
b = a.T
b.flags['C_CONTIGUOUS']      # False
b.flags['F_CONTIGUOUS']      # True

c = a[:, 1]
c.flags['C_CONTIGUOUS']      # False
```

### Forcing contiguity

```python
np.ascontiguousarray(x)      # returns C-contiguous (copies if needed)
np.asfortranarray(x)         # returns F-contiguous
```

Use these before passing to libraries that expect a specific layout (BLAS, C extensions, `sklearn`, etc.).

### Performance impact

```python
N = 5000
contig     = rng.random((N, N))
non_contig = contig.T        # column-major view

contig.sum(axis=0)           # fast — sequential memory
non_contig.sum(axis=0)       # slower — strided access
```

> ⚠️ **Reductions along a non-contiguous axis are slower.** Reorder axes or copy if this is a hot path.

### Creating contiguous arrays directly

```python
np.zeros((3, 3), order='C')  # default
np.zeros((3, 3), order='F')  # Fortran layout
np.zeros((3, 3), order='A')  # either (let NumPy decide)
```

---

## 5️⃣ Practice Challenges

| # | Task | Answer |
|---|---|---|
| 1 | Does `a[5:10]` share memory? | **True** — basic slice → view |
| 2 | `a[2:5] = 99` modifies `a`? | **Yes** — slice assignment mutates in place |
| 3 | Does `a[[0, 1, 2]]` share memory? | **False** — fancy index → copy |
| 4 | Does `a.T` share memory? | **True** — transpose is a view |
| 5 | Does `a.ravel()` share memory? | **True** (for contiguous input) |
| 6 | Make a non-contiguous slice contiguous | `np.ascontiguousarray(b)` |
| 7 | Basic vs. fancy share? | `(True, False)` |
| 8 | `nbytes` of `(100, 200)` float32 | `100 × 200 × 4 = 80 000` bytes |

---

## 📊 Quick Reference

| Goal | Function |
|---|---|
| Check view vs. copy | `x.base is not None` |
| Check overlap | `np.shares_memory(a, b)` |
| Fast overlap check | `np.may_share_memory(a, b)` |
| Independent copy | `a.copy()`, `np.copy(a)` |
| Deep copy (object arrays) | `copy.deepcopy(a)` |
| Control copy | `np.array(a, copy=True/False)` |
| Check layout | `a.flags['C_CONTIGUOUS']`, `a.flags['F_CONTIGUOUS']` |
| Force C-order | `np.ascontiguousarray(a)` |
| Force F-order | `np.asfortranarray(a)` |
| Create with layout | `np.zeros(shape, order='C'/'F')` |
| Large-array I/O | `np.load(path, mmap_mode='r')` |
| Memory used | `a.nbytes` |

---

## ✅ Takeaways

1. **Basic slicing, reshape, transpose, and ravel return views** — mutation propagates to the original.
2. **Fancy/boolean indexing and `flatten()` return copies** — safe but allocating.
3. `=` **never copies** — it aliases. Use `.copy()` when you need independence.
4. **`copy.deepcopy`** is only necessary for **object-dtype** arrays (or nested Python containers).
5. Use **`np.shares_memory`** for a precise check, **`np.may_share_memory`** for a fast one.
6. `.base` traces the view chain back to the original owning buffer.
7. **C-contiguity** matters — strided access is slower; use `np.ascontiguousarray` before passing to libraries that require it.
8. **`memmap`** is the standard tool for arrays larger than RAM.
9. `nbytes = size × itemsize` — the definitive memory-footprint formula.

---
