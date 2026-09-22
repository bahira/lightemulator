#!/usr/bin/env python3
"""Quick SPEAR BT29 LLM Attention Benchmark"""
import numpy as np
import time
import sys

n = 100000
np.random.seed(42)

# Generate test data - 4x4 matrices like BT29 would process
A = np.random.randn(4, 4).astype(np.float32)
B = np.random.randn(4, 4).astype(np.float32)

# Reference numpy multiply
t0 = time.time()
C_np = A @ B
t1 = time.time()
print(f"NumPy 4x4 matmul: {((t1-t0)*1000):.2f} ms")

# Manual 4x4 multiply (like spear_matrix_mul_4x4_avx2)
t0 = time.time()
C_manual = np.zeros((4,4), dtype=np.float32)
for i in range(4):
    for j in range(4):
        s = 0.0
        for k in range(4):
            s += A[i*4+k] * B[k*4+j]
        C_manual[i,j] = s
t1 = time.time()
print(f"Manual 4x4 multiply: {((t1-t0)*1000):.2f} ms")
print(f"Matches NumPy: {np.allclose(C_np, C_manual)}")
print(f"Max diff: {np.max(np.abs(C_np - C_manual)):.2e}")

# Now simulate the attention application
# For 100k 4x4 blocks (like ~16k tokens with d_k=64)
N_BLOCKS = 100000

t0 = time.time()
# Simulate the attention pattern: for each block pair, compute QKᵀ then aggregate
# This is simplified - real attention has softmax etc.
total_time = 0
for _ in range(10):  # 10 iterations for averaging
    t0 = time.time()
    # Simulate the multiply pattern
    result = np.zeros((4,4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += A[i*4+k] * B[k*4+j]
            result[i,j] = s
    t1 = time.time()
    total_time += (t1 - t0)

avg_time = total_time / 10
print(f"\nAverage 4x4 multiply time: {avg_time*1000:.2f} ms per iteration")
print(f"Theoretical speedup vs NumPy: { (0.135 / (avg_time*1000/100000)):.1f}x " if avg_time > 0 else "N/A")
print(f"@ 100k blocks: {(avg_time * 100000)/1000:.1f} ms for 100k blocks")