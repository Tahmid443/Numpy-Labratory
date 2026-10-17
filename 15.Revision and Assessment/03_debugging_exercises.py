"""Debugging exercises."""

import numpy as np

def exercise_1():
    """Bug: trying to add incompatible shapes."""
    print("Exercise 1: Broadcasting error")
    try:
        a = np.ones((3, 4))
        b = np.ones((3, 3))
        print(a + b)
    except ValueError as e:
        print(f"  Original error: {e}")
    
    # Fix: use compatible shapes
    b = np.ones((3, 4))
    print(f"  Fixed: shape {a.shape} + {b.shape} = {(a + b).shape}")

def exercise_2():
    """Bug: modifying a slice doesn't affect original (thought it would)."""
    print("\nExercise 2: Copy vs view")
    a = np.array([1, 2, 3, 4, 5])
    b = a[[1, 2, 3]]  # fancy indexing → copy
    b[0] = 99
    print(f"  a: {a} (expected [1,99,3,4,5]? No - b is a copy)")
    print(f"  b: {b}")
    
    # Fix: modify original directly
    a = np.array([1, 2, 3, 4, 5])
    a[[1, 2, 3]] = [20, 30, 40]
    print(f"  Fixed: a = {a}")

def exercise_3():
    """Bug: using == with NaN."""
    print("\nExercise 3: NaN comparison")
    a = np.array([1.0, np.nan, 3.0])
    print(f"  a == np.nan: {a == np.nan}")
    print(f"  Fix: np.isnan(a): {np.isnan(a)}")

def exercise_4():
    """Bug: axis confusion in reduction."""
    print("\nExercise 4: Axis confusion")
    m = np.arange(6).reshape(2, 3)
    print(f"  m:\n{m}")
    print(f"  Want column means [2.0, 3.0, 4.0]")
    wrong = m.mean(axis=1)
    print(f"  Wrong (axis=1): {wrong}")
    right = m.mean(axis=0)
    print(f"  Correct (axis=0): {right}")

def exercise_5():
    """Bug: integer division surprise."""
    print("\nExercise 5: Integer division")
    a = np.array([1, 2, 3], dtype=int)
    print(f"  a / 2: {a / 2} (float)")
    print(f"  a // 2: {a // 2} (int)")
    b = np.array([1, 2, 3])
    print(f"  np.float64 array / 2: {b / 2}")

def exercise_6():
    """Bug: reshape changes data unexpectedly."""
    print("\nExercise 6: reshape view")
    a = np.arange(6)
    b = a.reshape(2, 3)
    b[0, 0] = 99
    print(f"  Original a: {a} (changed!)")
    print(f"  Fix: use .copy() before reshape")
    a = np.arange(6)
    b = a.copy().reshape(2, 3)
    b[0, 0] = 99
    print(f"  a: {a} (unchanged)")

def exercise_7():
    """Bug: matrix multiplication vs element-wise."""
    print("\nExercise 7: * vs @")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"  A * B (element-wise):\n{A * B}")
    print(f"  A @ B (matrix):\n{A @ B}")

def exercise_8():
    """Bug: chained comparison."""
    print("\nExercise 8: Chained comparison")
    a = np.array([1, 2, 3, 4, 5, 6, 7])
    print(f"  a > 2 < 6 (wrong): {a > 2 < 6}")
    print(f"  Fix: (a > 2) & (a < 6): {(a > 2) & (a < 6)}")

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING EXERCISES")
    print("=" * 60)
    for i in range(1, 9):
        globals()[f'exercise_{i}']()