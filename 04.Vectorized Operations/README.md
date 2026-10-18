# NumPy — Vectorized Operations

Notes for **NumPy Laboratory → 04. Vectorized Operations**, covering arithmetic, comparisons, logical operators, universal functions (ufuncs), and conditional selection with `np.where` — all performed without explicit Python loops.

---

## 📁 Project Structure

```
04.Vectorized Operations/
├── 01_arithmetic_operations.py
├── 02_comparison_and_logical_operations.py
├── 03_universal_functions_ufuncs.py
├── 04_conditional_operations_where.py
└── 05_vectorization_challenges.py
```

---

## 1️⃣ Arithmetic Operations

Element-wise operations between arrays of the **same shape** (or arrays and scalars).

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

a + b        # [11 22 33 44]
a - b        # [-9 -18 -27 -36]
a * b        # [10 40 90 160]
a / b        # [0.1 0.1 0.1 0.1]
a // 2       # [0 1 1 2]      integer division
a % 3        # [1 2 0 1]      modulo
a ** 2       # [1 4 9 16]     power
2 ** a       # [2 4 8 16]     scalar base
-a           # [-1 -2 -3 -4]  unary negation
```

### In-place operations

```python
c = a.copy()
c += 10      # modifies c, no new array
d *= 2
```

### With scalars

```python
a + 100      # broadcast scalar
a * 10
a / 2
```

### 2-D operations

```python
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[10, 20], [30, 40]])
m1 + m2      # element-wise
m1 * m2      # element-wise (NOT matrix multiply!)
```

> ⚠️ `*` is element-wise. For matrix multiplication use `@` or `np.matmul`.

### Integer vs float division

```python
7 / 2        # 3.5  (float)
7 // 2       # 3    (int)
```

---

## 2️⃣ Comparison & Logical Operations

### Comparisons → boolean arrays

```python
a = np.array([1, 2, 3, 4, 5])

a > 3        # [F F F T T]
a < 3        # [T T F F F]
a == 3       # [F F T F F]
a != 3       # [T T F T T]
a >= 3       # [F F T T T]
a <= 3       # [T T T F F]
```

### Array-vs-array comparison

```python
b = np.array([5, 2, 4, 1, 3])
a > b        # element-wise
a == b
a != b
```

### Logical operators

| Operator | Meaning |
|---|---|
| `&` | element-wise AND |
| `\|` | element-wise OR |
| `~` | element-wise NOT |
| `^` | element-wise XOR |

```python
(a > 1) & (a < 5)   # [F T T T F]
(a < 2) | (a > 4)   # [T F F F T]
~(a > 3)            # [T T T F F]
```

> ⚠️ Always parenthesize each condition — `a > 1 & a < 5` parses wrong.

### Aggregated truth checks

```python
(a > 0).all()    # True  (all True?)
(a > 4).any()    # True  (any True?)
(a > 10).all()   # False
```

### Explicit logical functions

```python
np.logical_and(a > 1, a < 5)
np.logical_or(a < 2, a > 4)
np.logical_not(a > 3)
np.logical_xor(a > 2, b > 2)
```

### `np.isclose` — tolerance-aware float comparison

```python
np.isclose([1.0, 2.0], [1.0000001, 2.0])
# [True True]
```

> 💡 Never use `==` on floats — use `np.isclose` or `np.allclose`.

---

## 3️⃣ Universal Functions (ufuncs)

A **ufunc** is a compiled, element-wise function — the engine of NumPy speed.

### Math ufuncs

| Ufunc | Operation |
|---|---|
| `np.sqrt(x)` | Square root |
| `np.exp(x)` | e^x |
| `np.log(x)` | Natural log |
| `np.log10(x)` | Base-10 log |
| `np.sin`, `np.cos`, `np.tan` | Trig |
| `np.arcsin`, `np.arccos`, `np.arctan` | Inverse trig |
| `np.abs(x)` | Absolute value |
| `np.sign(x)` | -1 / 0 / 1 |

### Rounding

```python
x = np.array([1.4, 1.5, 1.6, 2.5, 2.4])

np.floor(x)   # [1. 1. 1. 2. 2.]
np.ceil(x)    # [2. 2. 2. 3. 3.]
np.round(x)   # [1. 2. 2. 2. 2.]
np.trunc(x)   # [1. 1. 1. 2. 2.]
```

> ⚠️ `np.round` uses **banker's rounding** — `2.5 → 2`, not `3`.

### Arithmetic ufuncs

| Ufunc | Equivalent |
|---|---|
| `np.add(a, b)` | `a + b` |
| `np.subtract(a, b)` | `a - b` |
| `np.multiply(a, b)` | `a * b` |
| `np.divide(a, b)` | `a / b` |
| `np.power(a, b)` | `a ** b` |
| `np.mod(a, b)` | `a % b` |

### Element-wise min/max

```python
np.maximum(a, b)   # element-wise max
np.minimum(a, b)   # element-wise min
```

### Accumulation

```python
np.add.accumulate([1, 2, 3, 4])   # [1 3 6 10]  (running sum)
np.add.reduce([1, 2, 3, 4])       # 10          (= sum)
np.multiply.accumulate([1, 2, 3]) # [1 2 6]
```

### `out=` — write into preallocated memory

```python
result = np.empty_like(a, dtype=float)
np.sqrt(b, out=result)
```

> 💡 `out=` avoids allocations in hot loops — a common performance trick.

---

## 4️⃣ Conditional Operations with `np.where`

### As a vectorized if/else (3 args)

```python
a = np.array([1, 5, 3, 8, 2, 9, 4])

np.where(a > 4, a, 0)          # keep value if >4 else 0
np.where(a > 4, 1, 0)          # 1 if >4 else 0
np.where(a > 4, 'big', 'small')
```

### As an index finder (1 arg)

```python
np.where(a > 4)                # (array([1, 3, 5]),)
np.where(a > 4)[0]             # [1 3 5]
```

### Multi-dim

```python
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.where(b > 5, b, 0)
np.where(b % 2 == 0, b, -1)
```

### Nested `where` (multi-branch)

```python
np.where(c < 3, 'low',
  np.where(c < 6, 'medium', 'high'))
```

### Broadcasting with `where`

```python
row = np.array([1, 2, 3])
col = np.array([[10], [20], [30]])
np.where(col > 15, row, 0)
# [[ 0  0  0]
#  [ 1  2  3]
#  [ 1  2  3]]
```

### Performance vs. Python loops

| Implementation | Time (1M elements) |
|---|---|
| Python list comprehension | ~0.06 s |
| `np.where` | ~0.004 s |
| **Speedup** | **~15–50×** |

---

## 5️⃣ Practice Challenges

| # | Task | Solution |
|---|---|---|
| 1 | `a² + b²` | `a**2 + b**2` |
| 2 | Normalize to [0, 1] | `(a - a.min()) / (a.max() - a.min())` |
| 3 | ReLU: `max(0, x)` | `np.maximum(0, a)` |
| 4 | Sigmoid | `1 / (1 + np.exp(-x))` |
| 5 | Count > mean | `(a > a.mean()).sum()` |
| 6 | Manhattan distance | `np.abs(p1 - p2).sum()` |
| 7 | One-hot encode | `np.eye(n)[labels]` |
| 8 | Clamp to range | `np.clip(a, 0, 10)` |

---

## 📊 Quick Reference

| Need | Approach |
|---|---|
| Element-wise math | `a + b`, `a * b`, `a ** 2` |
| In-place update | `a += 1` (avoids new array) |
| Boolean mask | `a > 4` |
| Combine conditions | `(a > 2) & (a < 8)` |
| All/any | `(a > 0).all()`, `(a > 4).any()` |
| Fast math function | `np.sqrt`, `np.exp`, `np.sin`, … |
| Conditional selection | `np.where(cond, x, y)` |
| Index finder | `np.where(cond)[0]` |
| Float comparison | `np.isclose`, `np.allclose` |
| Reduce | `np.add.reduce`, `np.sum` |
| Cumulative | `np.add.accumulate`, `np.cumsum` |

---

## ✅ Takeaways

1. **Everything in NumPy is element-wise by default** — no loops needed.
2. `+`, `-`, `*`, `/` operate per element; `*` is **not** matrix multiplication.
3. Use `&`, `|`, `~` (with parentheses) to combine boolean conditions.
4. **ufuncs** are compiled C functions — use them for speed (`np.sqrt`, `np.exp`, `np.maximum`, etc.).
5. `np.where` is the vectorized if/else; for multi-branch use `np.select` or nested `np.where`.
6. `out=` avoids reallocation — critical for large-array performance.
7. Never compare floats with `==`; use `np.isclose`.

---
