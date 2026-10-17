"""Compare NumPy arrays with Python lists."""

import numpy as np
import time

def compare_memory():
    """Compare memory usage."""
    import sys
    py_list = list(range(1000))
    np_array = np.arange(1000)
    
    print("Memory usage:")
    print(f"  Python list: {sys.getsizeof(py_list)} bytes (elements) + "
          f"{sum(sys.getsizeof(x) for x in py_list)} bytes (objects)")
    print(f"  NumPy array: {np_array.nbytes} bytes")
    print(f"  NumPy itemsize: {np_array.itemsize} bytes per element")

def compare_speed():
    """Compare computation speed."""
    size = 1_000_000
    py_list = list(range(size))
    np_array = np.arange(size)
    
    # Python list multiplication
    start = time.time()
    result_list = [x * 2 for x in py_list]
    list_time = time.time() - start
    
    # NumPy multiplication
    start = time.time()
    result_np = np_array * 2
    np_time = time.time() - start
    
    print(f"\nSpeed comparison (size={size}):")
    print(f"  Python list: {list_time:.4f}s")
    print(f"  NumPy array: {np_time:.4f}s")
    print(f"  Speedup: {list_time / np_time:.1f}x")

def compare_features():
    """Show features unique to NumPy."""
    arr = np.array([1, 2, 3, 4, 5])
    
    print("\nNumPy features:")
    print(f"  Vectorized ops: {arr * 2}")
    print(f"  Element-wise: {arr + arr}")
    print(f"  Broadcasting: {arr + 10}")
    print(f"  Aggregations: sum={arr.sum()}, mean={arr.mean()}")
    print(f"  Boolean masking: {arr[arr > 2]}")
    print(f"  Multi-dim: {np.zeros((2, 3))}")

if __name__ == "__main__":
    compare_memory()
    compare_speed()
    compare_features()