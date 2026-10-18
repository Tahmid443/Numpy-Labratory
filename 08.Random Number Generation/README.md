# NumPy — Random Number Generation

Notes for **NumPy Laboratory → 08. Random Number Generation**, covering the modern `Generator` API, reproducibility with seeds, common distributions, sampling and shuffling, and Monte Carlo simulation patterns.

---

## 📁 Project Structure

```
08.Random Number Generation/
├── 01_default_rng_and_seed.py
├── 02_random_integers_and_floats.py
├── 03_random_distributions.py
├── 04_sampling_and_shuffling.py
└── 05_simulation_challenges.py
```

---

## 1️⃣ Modern API: `default_rng` and Seeding

**Use `np.random.default_rng(...)` — not the legacy `np.random.seed(...)`.**

```python
rng = np.random.default_rng()              # no seed — non-reproducible
rng.random()                               # single float in [0, 1)
rng.integers(0, 10, 5)                     # 5 ints in [0, 10)
```

### Reproducibility

```python
rng = np.random.default_rng(42)
rng.random(3)                              # e.g. [0.77 0.43 0.03]

rng = np.random.default_rng(42)
rng.random(3)                              # IDENTICAL — same seed
```

- Same seed → **same sequence**.
- Different seed → different sequence.
- **Successive calls advance the state** — you can't re-draw the same numbers without reseeding.

### Legacy vs. modern

| | `np.random.seed(...)` | `np.random.default_rng(...)` |
|---|---|---|
| Class | `RandomState` (legacy) | `Generator` (recommended) |
| New code | ❌ | ✅ |
| Speed | Slower | Faster |
| Statistical quality | Good | Better |
| API | `rand`, `randint` | `random`, `integers` |

> 💡 Prefer **passing `rng` as an argument** to functions rather than using a global seed — makes testing and parallelism cleaner.

---

## 2️⃣ Random Integers and Floats

```python
rng = np.random.default_rng(42)

rng.random(5)                    # floats in [0, 1)
rng.random((2, 3))               # 2×3 floats

rng.integers(0, 10, 5)           # ints in [0, 10)
rng.integers(1, 100, (2, 3))     # 2×3 ints in [1, 100)
rng.integers(1, 6, 5, endpoint=True)  # ints in [1, 6] (inclusive)
```

### Other useful draws

```python
rng.standard_normal(5)           # N(0, 1)
rng.uniform(0, 10, 5)            # U(0, 10)

rng.choice(5)                    # scalar index
rng.choice(a, 3)                 # 3 samples from a (with replacement)
rng.choice(a, 3, replace=False)  # without replacement
rng.choice(a, 5, p=[...])        # weighted sampling
```

### Shuffle vs. permutation

```python
b = np.arange(10)
rng.permutation(b)               # returns NEW shuffled array — b unchanged
rng.shuffle(b)                   # shuffles b IN PLACE
```

> ⚠️ `shuffle` **mutates** the array; `permutation` **returns a copy**.

---

## 3️⃣ Random Distributions

| Distribution | Method | Typical use |
|---|---|---|
| Normal | `rng.normal(loc, scale, size)` | Measurement noise, weights init |
| Standard normal | `rng.standard_normal(size)` | Same, μ=0, σ=1 |
| Uniform | `rng.uniform(low, high, size)` | Flat priors, random positions |
| Binomial | `rng.binomial(n, p, size)` | Coin flips, conversion rates |
| Poisson | `rng.poisson(lam, size)` | Event counts per interval |
| Exponential | `rng.exponential(scale, size)` | Inter-arrival times |
| Beta | `rng.beta(a, b, size)` | Proportions, probabilities |
| Gamma | `rng.gamma(shape, scale, size)` | Waiting times |
| Chi-square | `rng.chisquare(df, size)` | Goodness-of-fit |
| Multivariate normal | `rng.multivariate_normal(mean, cov, size)` | Correlated features |

### Example

```python
rng = np.random.default_rng(42)

normal   = rng.normal(0, 1, 1000)          # μ=0, σ=1
uniform  = rng.uniform(0, 1, 1000)
binomial = rng.binomial(n=10, p=0.5, 1000) # mean ≈ 5
poisson  = rng.poisson(lam=3, 1000)        # mean ≈ 3
exp      = rng.exponential(scale=2, 1000)  # mean ≈ 2
beta     = rng.beta(a=2, b=5, 1000)        # mean ≈ 2/7

# Correlated 2-D data
mv = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=1000)
mv.shape         # (1000, 2)
mv.mean(axis=0)  # ≈ [0, 0]
```

### Sanity checks

```python
normal.mean()   # ≈ 0
normal.std()    # ≈ 1
binomial.mean() # ≈ 5
poisson.mean()  # ≈ 3
```

---

## 4️⃣ Sampling and Shuffling

### With vs. without replacement

```python
rng.choice(10, size=5, replace=True)   # duplicates allowed
rng.choice(10, size=5, replace=False)  # unique draws
```

### Weighted sampling

```python
items   = ['A', 'B', 'C', 'D']
weights = [0.1, 0.2, 0.3, 0.4]

rng.choice(items, size=20, p=weights)
# 'D' appears ~40% of the time
```

> ⚠️ Probabilities must **sum to 1.0** and have **non-negative** values — otherwise `ValueError`.

### Stratified sampling

```python
data = np.arange(100)
rng.shuffle(data)
strata = data.reshape(5, 20)

samples = [rng.choice(s, 3, replace=False) for s in strata]
np.concatenate(samples)
```

### Bootstrap

```python
def bootstrap(data, n_bootstrap=3):
    return np.array([
        rng.choice(data, size=len(data), replace=True)
        for _ in range(n_bootstrap)
    ])

boots = bootstrap(np.arange(1, 11))
boots.mean(axis=1)    # distribution of the sample mean
```

### Train/test split

```python
all_data = np.arange(100)
rng.shuffle(all_data)
split = int(0.8 * len(all_data))
train, test = all_data[:split], all_data[split:]
```

---

## 5️⃣ Monte Carlo Simulation Challenges

| # | Task | Key idea |
|---|---|---|
| 1 | Estimate π | sample (x, y) in unit square; π ≈ 4·fraction inside circle |
| 2 | Coin flips | `rng.integers(0, 2, N).sum()` |
| 3 | Two-dice sums | `np.bincount(d1 + d2)` |
| 4 | Random walk | `np.cumsum(rng.choice([-1, 1], N))` |
| 5 | Birthday paradox | compare `np.unique(birthdays).size` to N |
| 6 | Roll until 6 | expected ~6 rolls |
| 7 | Buffon's needle | π estimate from crossings |
| 8 | Estimate e | expected stopping time of unit sum |

### Example — π by Monte Carlo

```python
rng = np.random.default_rng(42)
n = 1_000_000

x, y = rng.random(n), rng.random(n)
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * inside.mean()
# ≈ 3.141…
```

### Example — birthday paradox (23 people → ~50.7%)

```python
rng = np.random.default_rng(42)
n_trials = 10_000
shared = 0

for _ in range(n_trials):
    bdays = rng.integers(0, 365, 23)
    if len(np.unique(bdays)) < 23:
        shared += 1

shared / n_trials          # ≈ 0.507
```

---

## 📊 Quick Reference

| Need | Function |
|---|---|
| Create generator | `np.random.default_rng(seed)` |
| Float in [0, 1) | `rng.random(shape)` |
| Integers | `rng.integers(low, high, shape)` |
| Normal | `rng.normal(loc, scale, shape)` |
| Uniform | `rng.uniform(low, high, shape)` |
| Binomial | `rng.binomial(n, p, shape)` |
| Poisson | `rng.poisson(lam, shape)` |
| Multivariate normal | `rng.multivariate_normal(mean, cov, shape)` |
| Sample | `rng.choice(a, size, replace, p)` |
| Shuffle in place | `rng.shuffle(a)` |
| Shuffled copy | `rng.permutation(a)` |
| Unique values | `np.unique(a)` |
| Histogram of ints | `np.bincount(a)` |

---

## ✅ Takeaways

1. **Always use `np.random.default_rng(...)`** — modern, faster, statistically better than `RandomState`.
2. **Pass the generator explicitly** to functions rather than relying on global seeds.
3. **Same seed → same sequence**; **successive calls advance state**.
4. `rng.shuffle` mutates in place; `rng.permutation` returns a copy.
5. `rng.choice(..., p=...)` for weighted sampling — weights must sum to 1.
6. Bootstrap = sampling **with** replacement; train/test = sampling **without** — or simply `shuffle` + slice.
7. **Monte Carlo** is trivial in NumPy — generate millions of samples and reduce.
8. Use **`replace=False`** for unique draws; for large arrays, `permutation` + slice is faster than `choice(..., replace=False)`.

---
