"""Implement operations with Python loops."""

import time

def py_add(a, b):
    return [x + y for x, y in zip(a, b)]

def py_multiply(a, b):
    return [x * y for x, y in zip(a, b)]

def py_square(a):
    return [x ** 2 for x in a]

def py_sum(a):
    total = 0
    for x in a:
        total += x
    return total

def py_dot(a, b):
    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total

def py_matrix_multiply(A, B):
    n, m = len(A), len(B[0])
    k = len(B)
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for l in range(k):
                result[i][j] += A[i][l] * B[l][j]
    return result

if __name__ == "__main__":
    n = 100_000
    a = list(range(n))
    b = list(range(n))

    start = time.time()
    result = py_add(a, b)
    print(f"Python list add ({n} elements): {time.time() - start:.4f}s")

    start = time.time()
    result = py_square(a)
    print(f"Python square: {time.time() - start:.4f}s")

    start = time.time()
    result = py_sum(a)
    print(f"Python sum: {time.time() - start:.4f}s")

    start = time.time()
    result = py_dot(a, b)
    print(f"Python dot: {time.time() - start:.4f}s")

    # Matrix multiply
    n = 100
    A = [[i * n + j for j in range(n)] for i in range(n)]
    B = [[i * n + j for j in range(n)] for i in range(n)]
    start = time.time()
    result = py_matrix_multiply(A, B)
    print(f"Python matrix multiply ({n}x{n}): {time.time() - start:.4f}s")