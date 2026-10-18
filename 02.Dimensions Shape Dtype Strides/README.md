# NumPy — Dimensions, Shape, Dtype & Strides

Notes for **NumPy Laboratory → 02. Dimensions, Shape, Dtype, Strides**, covering the metadata that describes every NumPy array: `ndim`, `shape`, `size`, `dtype`, `itemsize`, `nbytes`, `strides`, and memory layout.

---

## 📁 Project Structure

```
02.Dimensions Shape Dtype Strides/
├── 01_ndim_and_shape.py
├── 02_size_and_itemsize.py
├── 03_dtype_and_type_conversion.py
├── 04_strides_and_memory_layout.py
└── 05_array_metadata_challenges.py
```

---

## 1️⃣ `ndim` and `shape`

Every array carries:

| Attribute | Meaning |
|---|---|
| `ndim` | Number of **dimensions** (axes) |
| `shape` | **Tuple** giving length along each axis |
| `size` | Total number of elements (product of shape) |

```python
a0 = np.array(42)                             # 0-D  → ndim=0, shape=()
a1 = np.array([1, 2, 3])                      # 1-D  → ndim=1, shape=(3,)
a2 = np.array([[1, 2, 3], [4, 5, 6]])         # 2-D  → ndim=2, shape=(2, 3)
a3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3-D → shape=(2, 2, 2)
```

**Key points:**
- A **scalar** has `shape = ()` — an empty tuple, **not** `None`.
- `shape` is a **tuple**, so it is immutable and indexable: `a2.shape[0]` → rows, `a2.shape[1]` → cols.
- A **0-D array** still has `.shape`, `.ndim`, `.size`. Extract the scalar with `.item()`.

---

## 2️⃣ `size`, `itemsize`, `nbytes`

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.size       # 6        → total elements
arr.itemsize   # 8        → bytes per element (int64 → 8)
arr.nbytes     # 48       → size × itemsize
```

**Relationship:**

```
nbytes = size × itemsize
```

**itemsize by dtype:**

| dtype | itemsize |
|---|---|
| `int8` | 1 byte |
| `int16` | 2 bytes |
| `int32` | 4 bytes |
| `int64` | 8 bytes |
| `float32` | 4 bytes |
| `float64` | 8 bytes |

> 💡 A `100×100` `float32` array has `10 000 × 4 = 40 000 bytes ≈ 39 KB`. Always check `nbytes` before allocating huge arrays.

---

## 3️⃣ `dtype` and Type Conversion

### Common dtypes

| Category | Examples |
|---|---|
| Signed ints | `int8`, `int16`, `int32`, `int64` |
| Unsigned ints | `uint8`, `uint16`, `uint32`, `uint64` |
| Floats | `float16`, `float32`, `float64` |
| Complex | `complex64`, `complex128` |
| Others | `bool`, `str_`, `object_` |

### `astype()` — explicit conversion

```python
arr = np.array([1.7, 2.9, 3.1])
arr.astype(int)     # [1 2 3]  ← truncation, NOT rounding
arr.astype(str)     # ['1.7' '2.9' '3.1']
```

### Upcasting vs downcasting

```python
np.array([1, 2, 3], dtype=np.int32).astype(np.float64)  # safe ✅

np.array([1000, 2000, 3000], dtype=np.int64).astype(np.int8)
# → wraps around → garbage values (overflow) ⚠️
```

### Setting dtype at creation

```python
np.array([1, 2, 3], dtype=np.float32)
np.zeros((2, 3), dtype=np.int16)
```

### Type checking

```python
np.issubdtype(np.float32, np.floating)  # True
np.issubdtype(np.int32, np.integer)     # True
```

> ⚠️ **`astype` truncates, not rounds.** Use `np.round(arr).astype(int)` for rounding. Downcasting can silently overflow.

---

## 4️⃣ Strides & Memory Layout

### What are strides?

`strides` tells NumPy **how many bytes to jump** to move one step along each axis.

```python
a = np.arange(12).reshape(3, 4)
a.strides          # (32, 8)  ← for int64
# strides[0] = 32 bytes = 4 elements (jump to next row)
# strides[1] =  8 bytes = 1 element  (jump to next column)
```

### C-order (row-major) — default

```python
arr_c = np.array([[1, 2, 3], [4, 5, 6]], order='C')
arr_c.flags['C_CONTIGUOUS']   # True
```

- Elements in a row are **contiguous** in memory.
- Last axis varies fastest.
- This is NumPy's default and matches C arrays.

### F-order (column-major)

```python
arr_f = np.array([[1, 2, 3], [4, 5, 6]], order='F')
arr_f.flags['F_CONTIGUOUS']   # True
```

- Elements in a **column** are contiguous.
- First axis varies fastest.
- Common when interfacing with Fortran / BLAS / LAPACK.

### Why strides matter

- **Transpose is free** — it just **swaps strides** instead of moving data.
- **Views vs copies** — slicing produces views sharing memory; `strides` explain the view.
- **Performance** — operations over contiguous axes are much faster (cache-friendly).

### Transpose effect

```python
at = a.T
at.shape      # (4, 3)
at.strides    # (8, 32)  ← swapped; no data copied
```

---

## 5️⃣ Array Metadata — Full Inspection

A minimal inspector:

```python
def inspect(arr):
    return {
        'shape': arr.shape,
        'ndim': arr.ndim,
        'size': arr.size,
        'dtype': str(arr.dtype),
        'itemsize': arr.itemsize,
        'nbytes': arr.nbytes,
        'strides': arr.strides,
        'C_CONTIGUOUS': arr.flags['C_CONTIGUOUS'],
    }
```

**Example output for `np.zeros((3, 4))`:**

```
shape: (3, 4)
ndim: 2
size: 12
dtype: float64
itemsize: 8
nbytes: 96
strides: (32, 8)
C_CONTIGUOUS: True
```

---

## 📊 Quick Reference

| Attribute / Method | Returns |
|---|---|
| `arr.ndim` | Number of dimensions |
| `arr.shape` | Tuple of axis lengths |
| `arr.size` | Total elements |
| `arr.dtype` | Data type |
| `arr.itemsize` | Bytes per element |
| `arr.nbytes` | Total bytes (= `size × itemsize`) |
| `arr.strides` | Bytes to jump per axis step |
| `arr.flags` | Memory layout flags (`C_CONTIGUOUS`, `F_CONTIGUOUS`, etc.) |
| `arr.astype(dtype)` | Convert dtype (truncates) |
| `np.issubdtype(dt, kind)` | Check dtype category |

---

## ✅ Takeaways

1. `shape` is always a **tuple**; a scalar array has `shape == ()`.
2. `nbytes = size × itemsize` — use this to estimate memory footprint.
3. `astype` **truncates**, not rounds; always check for overflow when downcasting.
4. `strides` describe memory jumps; NumPy's power comes from **views over strides** (e.g., transpose needs no copy).
5. **C-order** (row-major) is the default; **F-order** matters for Fortran/BLAS interoperability.

---
