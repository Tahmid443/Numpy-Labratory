# NumPy — Matrix Multiplication

Notes for **NumPy Laboratory → 09. Matrix Multiplication**, covering the difference between element-wise and matrix products, the `@` / `np.matmul` / `np.dot` family, dimension compatibility, batched (broadcasting) multiplication, and practical linear-algebra patterns.

---

## 📁 Project Structure

```
09.Matrix Multiplication/
├── 01_elementwise_vs_matrix_product.py
├── 02_dot_and_matmul.py
├── 03_matrix_dimensions_and_compatibility.py
├── 04_batched_matrix_multiplication.py
└── 05_matrix_multiplication_challenges.py
```

---

## 1️⃣ Element-wise vs. Matrix Product

Two completely different operations share the `*` family syntax:

| Operation | Symbol | Meaning |
|---|---|---|
| **Element-wise** (Hadamard) | `A * B` | multiply **matching positions** |
| **Matrix product** | `A @ B`, `np.dot`, `np.matmul` | **row · column** inner products |

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A * B
# [[ 5 12]
#  [21 32]]     ← element-wise

A @ B
# [[19 22]
#  [43 50]]     ← matrix product
```

### Manual verification

```
C[0, 0] = A[0] · B[:, 0] = 1·5 + 2·7 = 19
C[0, 1] = A[0] · B[:, 1] = 1·6 + 2·8 = 22
```

### Non-square shapes

- Element-wise requires **equal shapes** (or broadcast-compatible).
- Matrix product requires `A.shape[-1] == B.shape[-2]`.

```python
C = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
D = np.array([[1, 2], [3, 4], [5, 6]]) # (3, 2)

C @ D        # (2, 2) ✅
C * D        # ❌ ValueError
```

---

## 2️⃣ `dot`, `matmul`, `@`, and Friends

### For 2-D arrays — all three are equivalent

```python
A @ B
np.dot(A, B)
np.matmul(A, B)
```

### For 1-D arrays — inner product

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)      # 32
np.matmul(a, b)   # 32
a @ b             # 32
```

> 💡 `1-D @ 1-D` = dot product (scalar). `2-D @ 2-D` = matrix product.

### For N-D arrays — `dot` and `matmul` **differ**

```python
X = np.arange(24).reshape(2, 3, 4)
Y = np.arange(20).reshape(2, 4, 5)

np.dot(X, Y).shape      # (2, 3, 2, 5)  ← outer-product style over leading axes
np.matmul(X, Y).shape   # (2, 3, 5)     ← batch-aware (what you usually want)
```

| | `np.dot` | `np.matmul` / `@` |
|---|---|---|
| 1-D · 1-D | inner product | inner product |
| 2-D · 2-D | matrix product | matrix product |
| N-D · N-D | last axis of `A` vs. second-to-last of `B` | **batched** over leading dims + broadcasting |
| Broadcasting | ❌ not batched | ✅ broadcast leading dims |

> ✅ **Rule of thumb:** use `@` for linear algebra. `np.dot` exists for legacy compatibility.

### Other product helpers

| Function | Meaning |
|---|---|
| `np.inner(a, b)` | sum over **last** axis of both |
| `np.outer(a, b)` | outer product → 2-D |
| `np.vdot(a, b)` | flattened dot (works with complex) |
| `np.tensordot(A, B, axes)` | contract over specified axes |
| `np.einsum('ij,jk->ik', A, B)` | general Einstein summation |

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.inner(a, b)                # 32
np.outer(a, b)                # 3×3
np.vdot(a, b)                 # 32

# einsum equivalent to A @ B
np.einsum('ij,jk->ik', A, B)
```

---

## 3️⃣ Dimension Compatibility

**The rule:** `(m, n) @ (n, p) → (m, p)`

```python
np.ones((2, 3)) @ np.ones((3, 4))   # (2, 4) ✅
np.ones((2, 3)) @ np.ones((4, 5))   # ❌
```

### 1-D handling in matmul

- `1-D @ 2-D` → treated as `(1, n) @ (n, p)` → result `(p,)`
- `2-D @ 1-D` → treated as `(m, n) @ (n, 1)` → result `(m,)`

```python
v = np.array([1, 2, 3])       # (3,)
M = np.ones((3, 4))

(v @ M).shape       # (4,)
(M.T @ v).shape     # (4,)
```

### Scalar vs. matrix product

```python
A * 2       # ✅ scalar multiply (element-wise)
A @ 2       # ❌ TypeError — `@` needs array-like
np.dot(A, 2) # ✅ still works (scalar multiply) — but confusing
```

> ⚠️ Never mix `np.dot(A, 2)` into code — it silently behaves like `A * 2` and confuses readers.

### Stack of matrices

```python
A = np.ones((3, 2, 2))   # 3 matrices of 2×2
B = np.ones((2, 2))      # single 2×2

(A @ B).shape            # (3, 2, 2) ← B broadcast to each A[i]
```

---

## 4️⃣ Batched Matrix Multiplication

**Batching** is `matmul`'s killer feature — you get a **stack** of matrix products in one call.

### Basic batch

```python
B = 4
A = rng.random((B, 2, 3))
B_ = rng.random((B, 3, 5))

(A @ B_).shape            # (4, 2, 5)
```

Equivalent (but slower) loop:

```python
np.array([A[i] @ B_[i] for i in range(B)])
```

### Broadcasting one matrix across a batch

```python
A = rng.random((3, 2, 3))    # 3 matrices of (2, 3)
S = rng.random((3, 5))       # single (3, 5)

(A @ S).shape                # (3, 2, 5)  ← S reused for all 3
```

### Broadcasting both sides

```python
A = rng.random((2, 1, 3, 4))
B = rng.random((1, 5, 4, 6))

(A @ B).shape                # (2, 5, 3, 6)
```

**Rules:** leading dimensions follow **standard broadcasting**; the **last two** dims must satisfy `(m, n) @ (n, p)`.

### Batched dot via `einsum`

```python
a = rng.random((4, 3))
b = rng.random((4, 3))

np.einsum('ij,ij->i', a, b)   # shape (4,)  — row-wise dot products
```

> 💡 `einsum` is often the fastest and clearest way to express batched reductions.

### Real-world: transform many 3-D points

```python
points    = rng.random((100, 3))          # 100 points
transform = rng.random((3, 3))            # 3×3 matrix

transformed = points @ transform.T        # (100, 3)
```

This pattern — `points @ T.T` or `points @ T` depending on row/column convention — is **the** way to apply a linear transform to N points without loops.

---

## 5️⃣ Practice Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | `A @ A` (A²) | plain matmul |
| 2 | `A @ A.T` | Gram matrix |
| 3 | Row norms squared | `np.einsum('ij,ij->i', A, A)` |
| 4 | Bilinear form `xᵀ A x` | `x @ A @ x` |
| 5 | Batch of matmuls | `(N, 3, 3) @ (N, 3, 2)` |
| 6 | Covariance matrix | `Xc.T @ Xc / (n - 1)` |
| 7 | Weighted sum of columns | `data @ weights` |
| 8 | Verify associativity | `np.allclose((A @ B) @ C, A @ (B @ C))` |

### Covariance example

```python
rng  = np.random.default_rng(42)
data = rng.random((100, 3))
Xc   = data - data.mean(axis=0)         # center
cov  = Xc.T @ Xc / (len(data) - 1)      # 3×3 covariance
```

### Weighted sum

```python
data    = np.array([[1, 2], [3, 4], [5, 6]])
weights = np.array([0.3, 0.7])

data @ weights          # [2.1, 3.9, 4.5]
```

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| Element-wise product | `A * B` |
| Matrix product | `A @ B` = `np.matmul(A, B)` |
| Dot product (1-D) | `np.dot(a, b)` or `a @ b` |
| Outer product | `np.outer(a, b)` = `a[:, None] * b` |
| Batched matmul | `A @ B` with `A.shape=(…, m, n)`, `B.shape=(…, n, p)` |
| Broadcast matmul | rely on standard broadcasting of leading dims |
| Batched row-wise dot | `np.einsum('ij,ij->i', A, B)` |
| Tensor contraction | `np.tensordot(A, B, axes)` |
| Custom einsum | `np.einsum(...)` |
| Verify result | `np.allclose(lhs, rhs)` |

---

## ✅ Takeaways

1. **`*` ≠ `@`** — `*` is element-wise, `@` is matrix product.
2. For 1-D arrays, `a @ b` is the **dot product** (scalar).
3. **Use `@`** — it's the modern operator. `np.dot` and `np.matmul` exist but `matmul` is preferred for N-D because it **batches**.
4. Shape rule: `(m, n) @ (n, p) → (m, p)`; 1-D is promoted/consumed along the appropriate axis.
5. **Batched matmul** (leading dims + broadcasting) is what makes NumPy fast for deep learning and graphics.
6. Common patterns:
   - Gram matrix: `A @ A.T`
   - Covariance: `Xc.T @ Xc / (n - 1)`
   - Point cloud transform: `points @ T.T`
7. Use `np.einsum` for custom batched contractions — often **clearer and faster** than looping.
8. Never mix `np.dot(A, scalar)` into code — use `A * scalar` for element-wise and reserve `@` for real matmuls.

---

