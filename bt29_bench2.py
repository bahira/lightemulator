#!/usr/bin/env python3
"""Quick SPEAR BT29 LLM Attention Benchmark"""
import numpy as np
import time

n = 100000

# Generate test data - 4x4 matrices like BT29 would process
# A and B are 4x4 row-major matrices
A = np.random.randn(4, 4).astype(np.float32)
B = np.random.randn(4, 4).astype(np.float32)

# Reference numpy multiply
t0 = time.time()
C_np = A @ B
t1 = time.time()
print(f"NumPy 4x4 matmul: {((t1-t0)*1000):.2f} ms")

# Manual 4x4 multiply (like spear_matrix_mul_4x4_avx2 row-major)
# A is row-major: A[i*4 + k] for i in 0..3, k in 0..3
# B is row-major: B[k*4 + j] for k in 0..3, j in 0..3
# C[i*4 + j] = sum over k of A[i*4+k] * B[k*4+j]

t0 = time.time()
C_manual = np.zeros((4,4), dtype=np.float32)
for i in range(4):
    for j in range(4):
        s = 0.0
        for k in range(4):
            s += A[i, k] * B[k, j]  # Use numpy indexing for simplicity
        C_manual[i, j] = s
t1 = time.time()
print(f"Manual 4x4 multiply: {((t1-t0)*1000):.2f} ms")
print(f"Matches NumPy: {np.allclose(C_np, C_manual)}")
print(f"Max diff: {np.max(np.abs(C_np - C_manual)):.2e}")

# Simulate 100k block multiplies (like LLM attention with many tokens)
N_BLOCKS = 100000
t0 = time.time()
total_s = 0.0
for _ in range(10):  # 10 iterations for averaging
    t_iter_start = time.time()
    # Multiply 100k 4x4 blocks
    count = 0
    for _ in range(N_BLOCKS):
        # Simulate one 4x4 block multiply
        s = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s += A[i, k] * B[k, j]
                count += 1
    t1 = time.time()
    total_time = t1 - t_iter_start
    
print(f"\n100k 4x4 block multiply time: {total_time:.4f} s for {N_BLOCKS} blocks")
print(f"Per-block time: {(total_time/N_BLOCKS)*1000:.4f} ms")
print(f"Theoretical libm equivalent: 0.135 ms for 100k elements (from earlier benchmark)")
print(f"Estimated speedup: {0.135 / (total_time/N_BLOCKS*1e-3):.1f}x")

# Also compute the total operations
total_ops = N_BLOCKS * 64  # 16 mults + 12 adds ≈ 28 FLOPs per 4x4, but ~64 for simplicity
print(f"\nTotal FLOPs: {total_ops:.0e}")
print(f"Effective GFLOPS: {total_ops / total_time / 1e9:.4f}")