"""Contiguous arrays and memory layout."""

import numpy as np

# C-contiguous (row-major)
a = np.arange(12).reshape(3, 4)
print(f"A:\n{a}")
print(f"C_CONTIGUOUS: {a.flags['C_CONTIGUOUS']}")
print(f"F_CONTIGUOUS: {a.flags['F_CONTIGUOUS']}")
print(f"OWNDATA: {a.flags['OWNDATA']}")
print(f"strides: {a.strides}")

# Transpose breaks C-contiguity
b = a.T
print(f"\nB = A.T:")
print(f"C_CONTIGUOUS: {b.flags['C_CONTIGUOUS']}")
print(f"F_CONTIGUOUS: {b.flags['F_CONTIGUOUS']}")
print(f"strides: {b.strides}")

# Column slice breaks contiguity
c = a[:, 1]
print(f"\nC = A[:, 1]:")
print(f"C_CONTIGUOUS: {c.flags['C_CONTIGUOUS']}")
print(f"strides: {c.strides}")

# Force contiguity
d = np.ascontiguousarray(b)
print(f"\nascontiguousarray(B):")
print(f"C_CONTIGUOUS: {d.flags['C_CONTIGUOUS']}")

# F-order
e = np.asfortranarray(a)
print(f"\nasfortranarray(A):")
print(f"F_CONTIGUOUS: {e.flags['F_CONTIGUOUS']}")

# Performance: contiguous access is faster
import time

N = 5000
contig = np.random.default_rng(42).random((N, N))
non_contig = contig.T  # column-contiguous

start = time.time()
_ = contig.sum(axis=0)
t_contig = time.time() - start

start = time.time()
_ = non_contig.sum(axis=0)
t_non = time.time() - start

print(f"\nSum along axis=0:")
print(f"  Contiguous: {t_contig:.4f}s")
print(f"  Non-contiguous: {t_non:.4f}s")

# Create F-contiguous array
f = np.zeros((3, 3), order='F')
print(f"\nF-order zeros:")
print(f"F_CONTIGUOUS: {f.flags['F_CONTIGUOUS']}")