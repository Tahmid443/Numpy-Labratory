"""Challenges: memory behavior."""

import numpy as np

def challenge_1():
    """Determine if slicing returns a view."""
    a = np.arange(20)
    b = a[5:10]
    return np.shares_memory(a, b)

def challenge_2():
    """Modify a slice and check original."""
    a = np.arange(10)
    a[2:5] = 99
    return a

def challenge_3():
    """Create a copy using indexing."""
    a = np.arange(10)
    b = a[[0, 1, 2]]
    return np.shares_memory(a, b)

def challenge_4():
    """Check if transpose shares memory."""
    a = np.arange(12).reshape(3, 4)
    b = a.T
    return np.shares_memory(a, b)

def challenge_5():
    """Verify ravel returns view when possible."""
    a = np.arange(12).reshape(3, 4)
    b = a.ravel()
    return np.shares_memory(a, b)

def challenge_6():
    """Find non-contiguous array and make it contiguous."""
    a = np.arange(12).reshape(3, 4)
    b = a[:, ::2]  # non-contiguous
    return b.flags['C_CONTIGUOUS'], np.ascontiguousarray(b).flags['C_CONTIGUOUS']

def challenge_7():
    """Determine memory of fancy vs basic index."""
    a = np.arange(10)
    basic = a[2:5]
    fancy = a[[2, 3, 4]]
    return (np.shares_memory(a, basic), np.shares_memory(a, fancy))

def challenge_8():
    """How much memory does an array's data use?"""
    a = np.zeros((100, 200), dtype=np.float32)
    return a.nbytes

if __name__ == "__main__":
    for i in range(1, 9):
        print(f"Challenge {i}: {globals()[f'challenge_{i}']()}")