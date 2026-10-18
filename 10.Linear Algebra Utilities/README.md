# NumPy — Linear Algebra Utilities

Notes for **NumPy Laboratory → 10. Linear Algebra Utilities**, covering transposes and identity, determinants and inverses, solving linear systems, eigen-decomposition, norms and dot products, SVD/QR/Cholesky, and the practical `np.linalg` routines.

---

## 📁 Project Structure

```
10.Linear Algebra Utilities/
├── 01_transpose_and_identity.py
├── 02_determinant_and_inverse.py
├── 03_solve_linear_systems.py
├── 04_eigenvalues_and_eigenvectors.py
├── 05_norms_and_dot_products.py
├── 06_svd_and_matrix_rank.py
└── 07_linear_algebra_challenges.py
```

---

## 1️⃣ Transpose, Identity, and Trace

```python
A = np.array([[1, 2, 3], [4, 5, 6]])

A.T                          # shape (3, 2)
np.array_equal((A.T).T, A)   # True  — involution

np.eye(3)                    # 3×3 identity
```

### Useful facts

- `A @ A.T` is **always symmetric** (Gram matrix).
- `np.trace(A)` = sum of the diagonal elements.
- `A.T @ A` is symmetric and **positive semi-definite**.
- Complex matrices: use `C.conj().T` (Hermitian) instead of `C.T`.

```python
C = np.array([[1+1j, 2], [3, 4-1j]])
C.conj().T                    # Hermitian adjoint
```

---

## 2️⃣ Determinant and Inverse

### Determinant

```python
A = np.array([[1, 2], [3, 4]])
np.linalg.det(A)              # -2.0
```

> ⚠️ `det` is computed in floating point — a theoretically-zero determinant may appear as `1e-17` instead of exactly `0`. **Don't rely on `det == 0`** for singularity; use `np.linalg.matrix_rank` or `np.linalg.cond`.

### Inverse

```python
np.linalg.inv(A)              # A⁻¹
A @ np.linalg.inv(A)          # ≈ I
```

Raises `np.linalg.LinAlgError` if the matrix is **singular** (det ≈ 0).

### Pseudo-inverse (Moore–Penrose)

For **singular** or **non-square** matrices:

```python
np.linalg.pinv(B)             # works even when inv fails
```

> 💡 **Prefer `solve(A, b)` over `inv(A) @ b`** — it's faster and more numerically stable. Only use `inv` when you actually need the inverse matrix itself.

### Trace

```python
np.trace(C)                   # sum of diagonal
```

---

## 3️⃣ Solving Linear Systems

### Square systems — `np.linalg.solve`

Solves **A·x = b** directly (LU decomposition under the hood).

```python
A = np.array([[2, 1], [1, 3]])
b = np.array([4, 7])

x = np.linalg.solve(A, b)     # [1. 2.]
A @ x                         # ≈ b
```

### Multiple right-hand sides

```python
B = np.array([[1, 2], [3, 4], [5, 6]])     # shape (3, 2)

X = np.linalg.solve(A, B)                  # each column solves A x = B[:, i]
```

`A` must be **square** and **nonsingular**; `B` can be `(n,)` or `(n, k)`.

### Overdetermined / underdetermined — `np.linalg.lstsq`

```python
X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
```

| Return | Meaning |
|---|---|
| `X` | Least-squares solution |
| `residuals` | Sum of squared residuals (empty if `rank < n` or `m ≤ n`) |
| `rank` | Effective rank of `A` |
| `s` | Singular values of `A` |

> 💡 **Rule of thumb:** `solve` for square non-singular systems; `lstsq` for everything else.

---

## 4️⃣ Eigenvalues and Eigenvectors

### General case — `np.linalg.eig`

Returns `(eigenvalues, eigenvectors)` where `eigenvectors[:, i]` corresponds to `eigenvalues[i]`.

```python
A = np.array([[4, 2], [1, 3]])

vals, vecs = np.linalg.eig(A)

# A @ v = λ v
for i in range(len(vals)):
    assert np.allclose(A @ vecs[:, i], vals[i] * vecs[:, i])
```

### Symmetric / Hermitian — `np.linalg.eigh`

**Always prefer `eigh` for symmetric matrices** — faster, returns **real** eigenvalues, and guarantees **orthonormal** eigenvectors.

```python
S = np.array([[4, 1], [1, 4]])
vals, vecs = np.linalg.eigh(S)

vecs.T @ vecs                 # ≈ I  (orthonormal)
```

### Reconstruct from eigendecomposition

```python
V = vecs
D = np.diag(vals)
S ≈ V @ D @ V.T               # for symmetric matrices
```

### Matrix powers via eigendecomposition

```python
S2_eig = V @ np.diag(vals**2) @ V.T
np.allclose(S2_eig, S @ S)    # True
```

### Condition number

```python
np.linalg.cond(A)             # σ_max / σ_min
```

- `≈ 1` → well-conditioned (stable).
- `≫ 1` → ill-conditioned (sensitive to rounding).

### Complex eigenvalues

```python
C = np.array([[0, -1], [1, 0]])   # rotation by 90°
np.linalg.eig(C)[0]               # [1j, -1j]
```

---

## 5️⃣ Norms and Dot Products

### Vector norms — `np.linalg.norm`

| Ord | Name | Formula |
|---|---|---|
| `1` | L1 (Manhattan) | Σ\|xᵢ\| |
| `2` | L2 (Euclidean) | √Σxᵢ² |
| `np.inf` | Max | max\|xᵢ\| |
| `-np.inf` | Min | min\|xᵢ\| |

```python
v = np.array([3, 4])
np.linalg.norm(v, 1)          # 7
np.linalg.norm(v, 2)          # 5
np.linalg.norm(v, np.inf)     # 4
np.linalg.norm(v)             # 5 (default L2)
```

### Matrix norms

```python
A = np.array([[1, 2], [3, 4]])

np.linalg.norm(A, 'fro')      # Frobenius: √Σaᵢⱼ²
np.linalg.norm(A, 2)          # spectral: largest singular value
np.linalg.norm(A, 'nuc')      # nuclear: sum of singular values
```

### Axis-wise norms (for batches)

```python
M = np.array([[3, 4], [5, 12], [8, 15]])

np.linalg.norm(M, axis=1)     # per-row norms
np.linalg.norm(M, axis=0)     # per-column norms
```

### Normalizing

```python
v3 = np.array([1, 2, 3])
v3 / np.linalg.norm(v3)       # unit vector
```

### Dot products

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

np.dot(a, b)    # 32
np.inner(a, b)  # 32
np.vdot(a, b)   # 32 (flattened, complex-safe)
a @ b           # 32
```

### Cosine similarity

```python
cos_sim = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
angle   = np.arccos(np.clip(cos_sim, -1, 1))
```

**Cosine similarity matrix** — normalize each row, then matmul:

```python
vectors = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
normed = vectors / norms
cos_sim = normed @ normed.T    # (3, 3)
```

> 💡 `np.clip(cos_sim, -1, 1)` guards against `1.0000000002` → `arccos` domain errors.

---

## 6️⃣ SVD, QR, Cholesky, Rank

### Rank

```python
np.linalg.matrix_rank(A)      # numerical rank
```

Uses SVD internally with a tolerance — more reliable than `det`.

### SVD — `np.linalg.svd`

**A = U · diag(s) · Vᵀ**

```python
U, s, Vt = np.linalg.svd(A)

A ≈ U @ np.diag(s) @ Vt
```

Shape rules:

| Input shape | `full_matrices=True` (default) | `full_matrices=False` |
|---|---|---|
| `(m, n)` | `U: (m, m)`, `s: (min,)`, `Vt: (n, n)` | `U: (m, k)`, `s: (k,)`, `Vt: (k, n)` where `k = min(m, n)` |

Use `full_matrices=False` for **PCA**, **low-rank approximations**, and anything where memory matters.

### Low-rank approximation

```python
U, s, Vt = np.linalg.svd(A_big, full_matrices=False)
k = 3
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
error = np.linalg.norm(A_big - A_approx)
```

> 💡 This is exactly the **PCA / rank-k compression** algorithm.

### QR decomposition

**A = Q · R** with **Q orthogonal** and **R upper-triangular**.

```python
Q, R = np.linalg.qr(A)
np.allclose(Q @ R, A)         # True
Q.T @ Q                       # ≈ I
```

Common uses: least squares, orthogonalization, stable eigenvalue algorithms.

### Cholesky decomposition

**A = L · Lᵀ** — only for **symmetric positive-definite** matrices.

```python
S = np.array([[4, 2], [2, 3]])
L = np.linalg.cholesky(S)
np.allclose(L @ L.T, S)       # True
```

> ⚠️ Raises `LinAlgError` if the matrix is **not positive-definite**. Use `eigh` to check: all eigenvalues must be > 0.

### Condition number

```python
np.linalg.cond(A)             # σ_max / σ_min
```

---

## 7️⃣ Practice Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | Solve `x + 2y = 5`, `3x + 4y = 11` | `np.linalg.solve` |
| 2 | Determinant of 3×3 | `np.linalg.det` |
| 3 | Eigenvalues of symmetric matrix | `np.linalg.eigvalsh` |
| 4 | Row-normalize a matrix | `M / norm(M, axis=1, keepdims=True)` |
| 5 | Rank of rank-1 matrix | `np.linalg.matrix_rank` |
| 6 | Inverse + verify | `A_inv = inv(A)` |
| 7 | Projection of `a` onto `b` | `(a·b / b·b) * b` |
| 8 | Condition number of Hilbert matrix | `np.linalg.cond` |

### Projection example

```python
a = np.array([1, 2, 3])
b = np.array([1, 0, 1])

proj = (a @ b) / (b @ b) * b   # vector projection of a onto b
```

### Hilbert matrix — notoriously ill-conditioned

```python
n = 5
H = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
np.linalg.cond(H)              # ~ 4.7e5 for n=5, explodes with n
```

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| Transpose | `A.T`, `A.conj().T` (Hermitian) |
| Identity | `np.eye(n)` |
| Trace | `np.trace(A)` |
| Determinant | `np.linalg.det(A)` |
| Inverse | `np.linalg.inv(A)` |
| Pseudo-inverse | `np.linalg.pinv(A)` |
| Solve `Ax = b` | `np.linalg.solve(A, b)` |
| Least-squares | `np.linalg.lstsq(A, b)` |
| Eigen (general) | `np.linalg.eig(A)` |
| Eigen (symmetric) | `np.linalg.eigh(A)` |
| Eigenvalues only | `np.linalg.eigvals`, `eigvalsh` |
| Vector norm | `np.linalg.norm(v, ord)` |
| Matrix norm | `np.linalg.norm(A, 'fro'/'nuc'/2)` |
| Rank | `np.linalg.matrix_rank(A)` |
| SVD | `np.linalg.svd(A)` |
| QR | `np.linalg.qr(A)` |
| Cholesky | `np.linalg.cholesky(A)` |
| Condition number | `np.linalg.cond(A)` |

---

## ✅ Takeaways

1. **Prefer `solve(A, b)` over `inv(A) @ b`** — faster and numerically stable.
2. Use **`eigh` for symmetric** matrices — faster, real, orthonormal eigenvectors.
3. **`lstsq`** handles overdetermined systems; **`solve`** requires square + nonsingular.
4. **Never test singularity with `det == 0`** — use `matrix_rank` or `cond`.
5. **SVD** is the universal workhorse: rank, pseudo-inverse, PCA, low-rank approximation.
6. **QR** is the standard for least squares and orthogonalization.
7. **Cholesky** is the fast path for symmetric positive-definite matrices (used in Gaussian processes, Kalman filters).
8. `np.linalg.norm(A, axis=…)` gives batched norms without loops.
9. **Cosine similarity** via `normalize_rows(V) @ normalize_rows(V).T` — the standard way to compute pairwise similarity.
10. Always verify results with `np.allclose` — floating-point comparisons with `==` will fail.

---
