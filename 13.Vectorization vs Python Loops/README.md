# NumPy — Vectorization vs. Python Loops

Notes for **NumPy Laboratory → 13. Vectorization vs Python Loops**, covering why NumPy's compiled loops beat Python loops, how to benchmark fairly with `timeit`, memory differences, and patterns for converting loop-based code into vectorized operations.

---

## 📁 Project Structure

```
13.Vectorization vs Python Loops/
├── 01_python_loop_implementation.py
├── 02_numpy_vectorized_implementation.py
├── 03_benchmark_with_timeit.py
├── 04_memory_and_performance_comparison.py
└── 05_optimization_challenges.py
```

---

## 1️⃣ Python Loop Implementations

Every operation written as a Python loop pays the cost of the interpreter for **each iteration**.

```python
def py_add(a, b):
    return [x + y for x, y in zip(a, b)]

def py_square(a):
    return [x ** 2 for x in a]

def py_sum(a):
    total = 0
    for x in a:
        total += x
    return total

def py_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def py_matrix_multiply(A, B):
    n, m, k = len(A), len(B[0]), len(B)
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for l in range(k):
                result[i][j] += A[i][l] * B[l][j]
    return result
```

### Why Python loops are slow

- **Interpreter overhead per iteration** — each `+`, `*`, comparison is a Python bytecode dispatch.
- **Dynamic typing** — every value is a heap-allocated `PyObject`.
- **No vectorization** — the CPU can't use SIMD instructions.
- **No cache locality** — objects are scattered on the heap.

---

## 2️⃣ NumPy Vectorized Implementations

The same operations expressed as single NumPy calls run **in compiled C**, without per-element Python overhead.

```python
import numpy as np

a = np.arange(n)
b = np.arange(n)

a + b          # element-wise add
a ** 2         # square
a.sum()        # sum
a @ b          # dot product
A @ B          # matrix multiply

np.sqrt(np.exp(matrix))    # composed ufuncs
```

### Why vectorized code is fast

| Factor | Python loop | NumPy vectorized |
|---|---|---|
| Iteration | Python bytecode | Compiled C loop |
| Types | Dynamic (per-element) | Fixed `dtype` |
| SIMD | ❌ | ✅ (AVX, SSE) |
| Cache | Poor | Contiguous access |
| Memory | Per-object | Single buffer |

### Typical speedups

| Operation | Python | NumPy | Speedup |
|---|---|---|---|
| Add 100k elements | ~5 ms | ~0.1 ms | **~50×** |
| Square 100k | ~5 ms | ~0.1 ms | **~50×** |
| Sum 100k | ~1 ms | ~0.01 ms | **~100×** |
| Dot product | ~10 ms | ~0.1 ms | **~100×** |
| 100×100 matmul | ~1 s | ~0.05 ms | **~20 000×** |
| `sqrt(exp)` 1000×1000 | (never) | ~10 ms | ∞ |

> 💡 **Matrix operations** gain the most — NumPy calls BLAS/LAPACK, which are hand-tuned assembly (often 10×+ faster than naive C).

---

## 3️⃣ Benchmarking with `timeit`

`timeit` runs code **many times** and reports accurate timing — essential for reliable comparisons.

### Basic usage

```python
import timeit

t = timeit.timeit(lambda: np_arr + 1, number=1000)
# number = number of repetitions
```

### With setup (avoids including setup time)

```python
t = timeit.timeit(
    'a + 1',
    setup='import numpy as np; a = np.arange(10000)',
    number=1000
)
```

### Comparing implementations

```python
t_py = timeit.timeit(lambda: [x + 1 for x in py_list], number=1000)
t_np = timeit.timeit(lambda: np_arr + 1, number=1000)

print(f"Speedup: {t_py / t_np:.1f}x")
```

### ⚠️ Benchmarking pitfalls

| Pitfall | Fix |
|---|---|
| Too few iterations | Use `number >= 100` |
| Timing setup / import | Use `setup=` |
| Warm-up cache effects | Run a few warm-up iterations first |
| Micro-differences from CPU frequency | Use `timeit.repeat` + `min` |
| NumPy threads off for small arrays | Ensure array is large enough to amortize overhead |
| Comparing on cold cache | Same array reused across runs |

### Reliable pattern — `timeit.repeat`

```python
results = timeit.repeat(
    lambda: np_arr + 1,
    number=1000,
    repeat=5
)
best = min(results)      # use the minimum — least interference
```

### Scaling with size

```python
for size in [100, 1000, 10_000, 100_000]:
    setup = f'import numpy as np; a = np.arange({size})'
    t = timeit.timeit('a + 1', setup=setup, number=1000)
    print(f"size={size}: {t:.4f}s")
```

> ⚠️ **Small arrays may favor Python lists** — the fixed overhead of creating a NumPy array (~1 µs) can dominate at `n < ~50`.

---

## 4️⃣ Memory & Performance Comparison

### Memory — the biggest gap

For a **list of 1 million ints**:

| Storage | Size |
|---|---|
| Python `list(range(n))` | ~**36 MB** (8 B pointer + 28 B object per int) |
| NumPy `np.arange(n, dtype=np.int64)` | **8 MB** (8 B per element) |

- **Python list** → 8-byte pointer + ~28-byte `PyLong` object **per element**.
- **NumPy array** → `itemsize` bytes **per element**, contiguous.

**A Python list of ints is ~4–5× larger** than the equivalent NumPy int64 array.

> 💡 Check actual memory with `arr.nbytes` (NumPy) vs. `sys.getsizeof` + per-object size (Python).

### Performance — repeatable patterns

Benchmarks across **1 million** elements:

| Operation | Python | NumPy | Speedup |
|---|---|---|---|
| `a + 5` | ~50 ms | ~1 ms | **~50×** |
| `a * 2` | ~50 ms | ~1 ms | **~50×** |
| `a ** 2` | ~60 ms | ~1 ms | **~60×** |
| `sum(a)` | ~10 ms | ~0.5 ms | **~20×** |
| `mean(a)` | ~10 ms | ~0.5 ms | **~20×** |
| `a + b` (element-wise) | ~50 ms | ~1 ms | **~50×** |

### Array creation speed

```python
list(range(n))   # ~10 ms for n=100k
np.arange(n)     # ~0.1 ms
```

> 💡 **Create arrays once, reuse them.** In a hot loop, converting a list to an array each iteration kills any speedup.

---

## 5️⃣ Optimization Challenges — Loop → Vectorized

| # | Task | Vectorized approach |
|---|---|---|
| 1 | Sum of squares of evens | `(arr[arr % 2 == 0] ** 2).sum()` |
| 2 | Count above mean | `(a > a.mean()).sum()` |
| 3 | Piecewise function | `np.where(x < 0, x**2, x**3)` |
| 4 | Pairwise distance matrix | `points[:, None, :] - points[None, :, :]` |
| 5 | Moving average | `np.convolve(data, kernel, 'valid')` |
| 6 | Count occurrences | `np.bincount(data)` |
| 7 | Stable sigmoid | `np.where(x >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))` |
| 8 | Group-by sum | `{k: values[keys == k].sum() for k in np.unique(keys)}` |

### Moving average — the classic loop → convolve

```python
data   = rng.random(1000)
window = 10
kernel = np.ones(window) / window

# Vectorized
moving_avg = np.convolve(data, kernel, mode='valid')
```

### Group-by sum — loop replacement

```python
keys   = np.array([1, 2, 1, 2, 1, 2])
values = np.array([10, 20, 30, 40, 50, 60])

result = {
    k: values[keys == k].sum()
    for k in np.unique(keys)
}
# {1: 90, 2: 120}
```

---

## 📊 Quick Reference — Vectorization Recipes

| Python loop pattern | Vectorized NumPy |
|---|---|
| `[x + 1 for x in a]` | `a + 1` |
| `[x * y for x, y in zip(a, b)]` | `a * b` |
| `sum(x*x for x in a)` | `(a * a).sum()` or `a @ a` |
| `[x if c else y for x, c, y in ...]` | `np.where(cond, x, y)` |
| `[f(x) for x in a]` | `np.func(a)` (ufunc) |
| `sum(x for x in a if p(x))` | `a[p(a)].sum()` |
| `[x for x in a if p(x)]` | `a[p(a)]` |
| `min/max` | `a.min()`, `a.max()` |
| `sum(a[i]*b[i])` | `a @ b` |
| `apply moving window` | `np.convolve` |
| `count occurrences` | `np.bincount` |
| `unique values` | `np.unique` |
| `sort` | `np.sort` |
| `group by` | boolean mask + `.sum()` |

---

## ✅ Takeaways

1. **Vectorized NumPy is 10–10 000× faster** than Python loops for array math — the gap grows with size and with matrix operations.
2. **NumPy arrays use ~4–5× less memory** than Python lists of the same integers.
3. **`timeit.repeat` + `min`** is the reliable benchmarking pattern.
4. **Avoid per-element Python logic** — replace it with a ufunc, boolean mask, `np.where`, `np.einsum`, or `np.convolve`.
5. **Matrix operations gain the most** because NumPy calls BLAS/LAPACK.
6. **For tiny arrays** (< ~50 elements), a Python list may be faster due to NumPy's fixed overhead.
7. **Don't recreate arrays inside a hot loop** — the creation cost dominates.
8. The **canonical vectorization patterns**:
   - loop + condition → `np.where`
   - loop + filter → boolean mask
   - loop + reduce → `.sum()` / `.mean()`
   - loop + dot → `@` / `np.einsum`
   - loop + sliding window → `np.convolve`

---
