"""Memory and performance comparison."""

import numpy as np
import sys
import time

def compare_memory():
    print("=" * 60)
    print("MEMORY COMPARISON")
    print("=" * 60)
    
    n = 1_000_000
    
    py_list = list(range(n))
    np_array = np.arange(n, dtype=np.int64)
    
    list_memory = sys.getsizeof(py_list) + sum(sys.getsizeof(x) for x in py_list[:1000]) * (n // 1000)
    print(f"\nPython list ({n} ints):")
    print(f"  Approx: {list_memory / 1e6:.1f} MB")
    print(f"  (List overhead + {sys.getsizeof(0)} bytes/int)")
    
    print(f"\nNumPy array ({n} int64):")
    print(f"  Actual: {np_array.nbytes / 1e6:.1f} MB")
    print(f"  Ratio: ~{list_memory / np_array.nbytes:.0f}x more memory for Python list")

def compare_performance():
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    n = 1_000_000
    py_list = list(range(n))
    np_array = np.arange(n)
    
    operations = [
        ("Add scalar", lambda x: [v + 5 for v in x] if isinstance(x, list) else x + 5),
        ("Multiply", lambda x: [v * 2 for v in x] if isinstance(x, list) else x * 2),
        ("Square", lambda x: [v ** 2 for v in x] if isinstance(x, list) else x ** 2),
        ("Sum", lambda x: sum(x) if isinstance(x, list) else x.sum()),
        ("Mean", lambda x: sum(x) / len(x) if isinstance(x, list) else x.mean()),
    ]
    
    print(f"\n{'Operation':<15} {'Python':<12} {'NumPy':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for name, op in operations:
        start = time.time()
        op(py_list)
        t_py = time.time() - start
        
        start = time.time()
        op(np_array)
        t_np = time.time() - start
        
        print(f"{name:<15} {t_py:<12.4f} {t_np:<12.4f} {t_py/t_np:<10.1f}x")

def compare_element_wise():
    print("\n" + "=" * 60)
    print("ELEMENT-WISE OPERATIONS (size=100000)")
    print("=" * 60)
    
    n = 100_000
    py_a = list(range(n))
    py_b = list(range(n))
    np_a = np.arange(n)
    np_b = np.arange(n)
    
    start = time.time()
    _ = [a + b for a, b in zip(py_a, py_b)]
    t_py = time.time() - start
    
    start = time.time()
    _ = np_a + np_b
    t_np = time.time() - start
    
    print(f"\nVector addition:")
    print(f"  Python: {t_py:.4f}s")
    print(f"  NumPy:  {t_np:.4f}s")
    print(f"  Speedup: {t_py/t_np:.0f}x")
    
    # Creating arrays
    print(f"\nArray creation:")
    start = time.time()
    _ = list(range(n))
    t_py = time.time() - start
    start = time.time()
    _ = np.arange(n)
    t_np = time.time() - start
    print(f"  Python list(range): {t_py:.4f}s")
    print(f"  NumPy arange:       {t_np:.4f}s")

if __name__ == "__main__":
    compare_memory()
    compare_performance()
    compare_element_wise()