# NumPy — Numerical Precision

Notes for **NumPy Laboratory → 12. Numerical Precision**, covering float32 vs. float64, IEEE 754 representation, rounding and tolerance, overflow/underflow, NaN/inf handling, and numerically stable computing patterns.

---

## 📁 Project Structure

```
12.Numerical Precision/
├── 01_float32_vs_float64.py
├── 02_floating_point_representation.py
├── 03_rounding_and_tolerance.py
├── 04_overflow_underflow.py
├── 05_nan_inf_and_finite_checks.py
└── 06_numerical_stability_challenges.py
```

---

## 1️⃣ `float32` vs. `float64`

| Property | `float32` | `float64` |
|---|---|---|
| Bytes per element | **4** | **8** |
| Decimal digits (≈) | **7** | **15–16** |
| Machine epsilon | ≈ `1.19e-7` | ≈ `2.22e-16` |
| Max value | ≈ `3.4e38` | ≈ `1.8e308` |
| Min normal | ≈ `1.18e-38` | ≈ `2.23e-308` |

```python
np.finfo(np.float32).eps    # 1.19e-7
np.finfo(np.float64).eps    # 2.22e-16
np.finfo(np.float64).max    # 1.797e+308
```

### Precision demonstration

```python
value = 1 / 3
np.float32(value)   # 0.33333334   ← only ~7 correct digits
np.float64(value)   # 0.3333333333333333
```

### Type promotion

```python
a32 = np.array([1.0], dtype=np.float32)
a64 = np.array([2.0], dtype=np.float64)

(a32 + a64).dtype   # float64 — higher precision wins
(a32 + 1).dtype     # float64 — Python int promotes to float64
(a32 + a32).dtype   # float32 — stays float32
```

> ⚠️ **`float32 + int_scalar` gives `float64`**, not `float32`. This is a common surprise when you thought you were saving memory.

### When to use `float32`

| Use `float32` | Use `float64` |
|---|---|
| Deep learning (memory + GPU speed) | Scientific computing / stats |
| Image / signal data | Financial / high-precision |
| When 7 digits suffice | When ill-conditioned systems appear |
| Large arrays that fit in cache | Accumulators over millions of terms |

### Summing many tiny values

```python
n = 1_000_000
arr32 = np.ones(n, dtype=np.float32) / n
arr64 = np.ones(n, dtype=np.float64) / n

arr32.sum()   # ≈ 0.999...  (accumulated rounding error)
arr64.sum()   # ≈ 1.0       (much closer)
```

> 💡 `np.sum` on `float32` uses a `float32` accumulator. Use `arr.sum(dtype=np.float64)` to accumulate in higher precision.

---

## 2️⃣ Floating-Point Representation (IEEE 754)

A float is stored as **sign · 2^(exponent) · 1.mantissa**.

| Type | Total bits | Sign | Exponent | Mantissa |
|---|---|---|---|---|
| `float32` | 32 | 1 | 8 | 23 |
| `float64` | 64 | 1 | 11 | 52 |

### The classic `0.1 + 0.2 != 0.3`

```python
0.1 + 0.2           # 0.30000000000000004
0.1 + 0.2 == 0.3    # False
np.isclose(0.1 + 0.2, 0.3)   # True
```

**Why?** `0.1` and `0.2` have **infinite binary expansions** — they can't be represented exactly in any finite-precision binary format.

### Exactly representable numbers

Numbers of the form `k / 2ⁿ` **are** exact:

```python
0.5      # exact
0.25     # exact
0.125    # exact
1/3      # NOT exact
0.1      # NOT exact
```

### Integers beyond 2⁵³ lose precision

```python
2**53               # 9007199254740992
float(2**53)        # 9007199254740992.0
float(2**53 + 1)    # 9007199254740992.0  ← same!
float(2**53) == float(2**53 + 1)   # True
```

> 💡 Use `int64` when you need integer values above `2⁵³ ≈ 9×10¹⁵`.

---

## 3️⃣ Rounding and Tolerance

### The rounding functions

```python
x = np.array([1.4, 1.5, 1.6, 2.5, 2.4, -1.5, -2.5])

np.round(x)     # [1. 2. 2. 2. 2. -2. -2.]  ← banker's rounding
np.floor(x)     # [1. 1. 1. 2. 2. -2. -3.]
np.ceil(x)      # [2. 2. 2. 3. 3. -1. -2.]
np.trunc(x)     # [1. 1. 1. 2. 2. -1. -2.]
np.rint(x)      # same as round
```

> ⚠️ **`np.round` uses banker's rounding** — ties go to the **nearest even number**. `2.5 → 2`, `3.5 → 4`. This differs from schoolbook rounding.

### Rounding to decimals

```python
np.round([1.2345, 2.3456], 2)    # [1.23 2.35]
np.round([1.2345, 2.3456], -1)   # [ 0.  0.]
```

### `np.isclose` / `np.allclose`

```python
np.isclose(a, b, rtol=1e-5, atol=1e-8)
```

**The formula:** `|a - b| ≤ atol + rtol × |b|`

| Parameter | Meaning | Default |
|---|---|---|
| `rtol` | Relative tolerance | `1e-5` |
| `atol` | Absolute tolerance | `1e-8` |

```python
np.isclose(0.1 + 0.2, 0.3)         # True
np.allclose(a, b)                  # True if ALL elements close
```

> 💡 Use `rtol` for large values, `atol` for values near zero. Set both when the array spans a wide dynamic range.

### Quantization

```python
def quantize(x, step):
    return np.round(x / step) * step

quantize(np.array([0.1, 0.4, 0.6, 0.9]), 0.25)
# [0.   0.5  0.5  1.  ]
```

### Exact sum — `math.fsum`

```python
sum([0.1] * 10)           # 0.9999999999999999
math.fsum([0.1] * 10)     # 1.0  ← exact rounding
```

> 💡 `math.fsum` uses a compensated summation algorithm. For arrays, `np.sum(arr, dtype=np.float64)` helps when accumulating `float32`.

---

## 4️⃣ Overflow and Underflow

### Integer overflow — **wraps around silently**

```python
a = np.array([127], dtype=np.int8)
a + 1        # → -128  (wraps!)

a = np.array([200], dtype=np.int8)
a + 100      # → 44   (wraps!)
```

> ⚠️ NumPy **does not raise** on integer overflow — it wraps, mimicking C behavior. Choose a wide enough dtype (usually `int64`) or check bounds.

### Python int vs. NumPy int

```python
2**63                        # Python int, arbitrary precision
np.int64(2**63)              # OverflowError
```

### Float overflow → `inf`

```python
np.array([1e308], dtype=np.float64) * 10   # inf
np.array([1e38], dtype=np.float32) * 10    # inf
```

### Float underflow → `0.0`

```python
np.array([1e-308], dtype=np.float64) / 10  # 1e-309 (subnormal) → 0 with / again
np.array(1e-400)                            # 0.0 (compile-time underflow)
```

### Controlling error handling — `np.seterr`

```python
np.seterr(all='warn')     # default: warn
np.seterr(divide='raise') # raise FloatingPointError
np.seterr(over='ignore')  # silence overflow warnings
```

Options per category (divide, over, under, invalid): `'ignore'`, `'warn'`, `'raise'`, `'call'`.

> 💡 Use `np.errstate(divide='ignore')` as a **context manager** for localized suppression:
> ```python
> with np.errstate(divide='ignore'):
>     result = 1 / a
> ```

### Avoiding integer overflow

```python
a = np.array([100_000], dtype=np.int32)
b = np.array([100_000], dtype=np.int32)
a * b                        # overflow!

a.astype(np.int64) * b.astype(np.int64)   # correct
```

> 💡 For sums and products, **upscale to `int64` or `float64` before multiplying**.

---

## 5️⃣ NaN, Infinity, and Finite Checks

### The four special checkers

```python
a = np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -1.0])

np.isnan(a)     # [F T F F F F]
np.isinf(a)     # [F F T T F F]
np.isposinf(a)  # [F F T F F F]
np.isneginf(a)  # [F F F T F F]
np.isfinite(a)  # [T F F F T T]  ← = ~(isnan | isinf)
```

### NaN quirks

- `np.nan == np.nan` → **`False`** (IEEE 754 rule).
- `np.nan != np.nan` → **`True`**.
- `b == np.nan` always returns `False` — never use this to find NaNs. Use `np.isnan`.
- All comparisons with NaN return `False`.

### NaN propagates through aggregations

```python
b = np.array([1.0, np.nan, 3.0])
b.sum()          # nan
b.mean()         # nan
np.nansum(b)     # 4.0
np.nanmean(b)    # 2.0
```

### Infinity propagates too

```python
c = np.array([1.0, np.inf])
c.sum()          # inf
c.mean()         # inf
1 / c            # [1.0, 0.0]
```

### Filter and replace

```python
a[np.isfinite(a)]                       # keep finite values
a[~np.isnan(a)]                         # remove NaN
d[np.isnan(d)] = 0                      # replace in place

np.nan_to_num(d)                        # replace nan→0, inf→large
np.nan_to_num(d, nan=np.nanmean(d))     # replace with mean
```

### Operations with `inf`

| Expression | Result |
|---|---|
| `inf + inf` | `inf` |
| `inf - inf` | `nan` |
| `inf * 0` | `nan` |
| `1 / inf` | `0.0` |
| `inf / inf` | `nan` |

---

## 6️⃣ Numerical Stability Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | Test `0.1 + 0.2 == 0.3` | `False`, but `isclose` → `True` |
| 2 | `float32` vs `float64` π | ~1e-7 difference |
| 3 | Sum 100k tiny values | `float32` loses precision |
| 4 | `logsumexp` trick | subtract max before `exp` |
| 5 | Numerically stable sigmoid | branch on sign of `x` |
| 6 | Underflow behavior | `1e-300 / 1e10` |
| 7 | Safe tolerance | use `rtol=1e-9` |
| 8 | Associativity failure | `(a+b)+c ≠ a+(b+c)` in float |

### `logsumexp` — avoid overflow in `log(Σexp(x))`

```python
x = np.array([1000, 1001, 1002])

# Naive — overflows
np.log(np.sum(np.exp(x)))    # inf

# Stable — factor out the max
m = x.max()
m + np.log(np.sum(np.exp(x - m)))   # ≈ 1002.41
```

> 💡 `scipy.special.logsumexp` implements this safely — use it in production.

### Stable sigmoid

```python
def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),      # safe for large positive
        np.exp(x) / (1 + np.exp(x)) # safe for large negative
    )
```

Without the branch, `np.exp(-x)` overflows to `inf` for `x = -1000`.

### Associativity failure

```python
a, b, c = 1e16, -1e16, 1.0

(a + b) + c   # 1.0
a + (b + c)   # 0.0
```

Floating-point arithmetic is **not associative** — this causes subtle bugs in reordered reductions and parallel sums.

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| Dtype info | `np.finfo(np.float64).eps` |
| Round | `np.round`, `np.floor`, `np.ceil`, `np.trunc` |
| Compare with tolerance | `np.isclose`, `np.allclose` |
| Exact float sum | `math.fsum` |
| Check NaN | `np.isnan` |
| Check inf | `np.isinf`, `np.isposinf`, `np.isneginf` |
| Check finite | `np.isfinite` |
| Replace NaN/inf | `np.nan_to_num` |
| NaN-safe reductions | `np.nansum`, `np.nanmean`, … |
| Control error policy | `np.seterr`, `np.errstate` |
| Log-sum-exp | `m + log(sum(exp(x - m)))` |

---

## ✅ Takeaways

1. **`float32` is faster and lighter but only ~7 digits** — use `float64` for anything numerical that matters.
2. **`0.1 + 0.2 ≠ 0.3`** — always compare floats with `np.isclose` / `np.allclose`, never `==`.
3. **`np.round` uses banker's rounding** — ties go to the nearest even integer.
4. **Integers wrap silently on overflow** — use `int64` (or check bounds) for safety.
5. **`np.seterr` / `np.errstate`** let you control divide/overflow/underflow behavior.
6. **NaN poisons aggregations** — always use `np.nan*` variants when input may contain NaN.
7. **`inf` propagates**: `inf + 1 = inf`, `inf - inf = nan`, `inf * 0 = nan`.
8. **Watch accumulation dtype** in `float32` sums — pass `dtype=np.float64` to `np.sum` for accuracy.
9. **Use stable formulations**:
   - `logsumexp` (subtract max) instead of `log(sum(exp))`
   - `np.where(x >= 0, ...)` for stable sigmoid
   - Sort ascending when summing many terms of different magnitude
10. **Float arithmetic is not associative** — order of operations changes the result.

---
