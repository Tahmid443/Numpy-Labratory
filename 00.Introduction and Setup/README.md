# NumPy — Introduction and Setup

Notes for **NumPy Laboratory → 00. Introduction and Setup** covering installation, importing conventions, and why NumPy outperforms Python lists.

---

## 📁 Project Structure

```
00.Introduction and Setup/
├── 01_numpy_installation.py
├── 02_import_numpy.py
└── 03_numpy_vs_python_lists.py
```

---

## 1️⃣ Installing NumPy

**Quick install:**

```bash
pip install numpy
```

**Programmatic check & install** (`01_numpy_installation.py`):

```python
import subprocess
import sys

def check_numpy_installed():
    try:
        import numpy
        return True, numpy.__version__
    except ImportError:
        return False, None

def install_numpy():
    print("Installing NumPy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
```

**Key points:**
- Uses `importlib`-style try/except to detect if NumPy is available.
- Installs via `sys.executable -m pip` to ensure the *right* Python environment.
- Safe to run — skips install if NumPy already exists.

---

## 2️⃣ Importing NumPy

**Standard convention:**

```python
import numpy as np
```

**Useful checks:**

```python
print(np.__version__)   # e.g. 1.26.4
np.show_config()        # build configuration
```

**Basic usage:**

```python
a = np.array([1, 2, 3])
print(type(a))          # <class 'numpy.ndarray'>
```

**Conventions to follow / avoid:**

| ✅ Recommended | ❌ Not recommended |
|---|---|
| `import numpy as np` | `import numpy` (verbose) |
| — | `from numpy import array, zeros, ones` (pollutes namespace) |

---

## 3️⃣ NumPy Arrays vs Python Lists

### 🧠 Memory Usage

```python
import sys, numpy as np

py_list  = list(range(1000))
np_array = np.arange(1000)

sys.getsizeof(py_list)               # list of pointers
sum(sys.getsizeof(x) for x in py_list)  # + each int object
np_array.nbytes                       # contiguous raw bytes
np_array.itemsize                     # bytes per element
```

- Python lists store **pointers to objects** (extra overhead per element).
- NumPy arrays store **raw contiguous data** → much more compact.

### ⚡ Speed

```python
size = 1_000_000
py_list  = list(range(size))
np_array = np.arange(size)

# List comprehension
[x * 2 for x in py_list]

# Vectorized
np_array * 2
```

- NumPy is typically **10–100x faster** than pure Python loops for large arrays.
- Reason: vectorized C loops + no per-element type checking.

### 🚀 NumPy-Specific Features

| Feature | Example |
|---|---|
| Vectorized ops | `arr * 2` |
| Element-wise arithmetic | `arr + arr` |
| Broadcasting | `arr + 10` |
| Aggregations | `arr.sum()`, `arr.mean()` |
| Boolean masking | `arr[arr > 2]` |
| Multi-dimensional arrays | `np.zeros((2, 3))` |

---

## 📊 Quick Comparison

| Aspect | Python List | NumPy Array |
|---|---|---|
| Element types | Mixed | Homogeneous (dtype) |
| Memory | High (objects + pointers) | Low (contiguous buffer) |
| Speed | Slow (interpreted loop) | Fast (vectorized C) |
| Math ops | Manual loops | Built-in vectorized |
| Multi-dim | Nested lists | True n-D array |

---

## ✅ Takeaways

1. Install with `pip install numpy`; verify with `np.__version__`.
2. Always import as `import numpy as np`.
3. Prefer NumPy arrays over Python lists for **numerical / large-scale** computation.
4. Leverage **vectorization, broadcasting, and masking** instead of loops.

---

