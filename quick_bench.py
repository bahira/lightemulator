#!/usr/bin/env python3
"""Quick SPEAR kernel benchmark"""
import numpy as np
import time

n = 100000
np.random.seed(42)
x = np.linspace(-4, 4, n).astype(np.float32)

# tanh
t0 = time.time()
y = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
t1 = time.time()
print(f'tanh: {((t1-t0)*1000):.2f} ms, error vs np.tanh: {np.max(np.abs(y - np.tanh(x))):.2e}')

# softplus
t0 = time.time()
ax = np.abs(x)
y = np.maximum(0, x) + (0.916 - 0.193 * ax) / (1.329 + 0.582 * ax + 0.413 * x * x)
t1 = time.time()
print(f'softplus: {((t1-t0)*1000):.2f} ms, error vs np.log1p: {np.max(np.abs(y - np.log1p(np.exp(x)))):.2e}')

# rsqrt (positive only)
x_pos = np.clip(x, 0.001, None)
t0 = time.time()
y = np.sqrt(1.011 / (1.011 * x_pos))
t1 = time.time()
print(f'rsqrt: {((t1-t0)*1000):.2f} ms')

# sin
t0 = time.time()
y = x * (339.752 - 2.089 * x**2) / (641.353 - 11.856 * x**2 + x**4)
t1 = time.time()
print(f'sin: {((t1-t0)*1000):.2f} ms, error vs np.sin: {np.max(np.abs(y - np.sin(x))):.2e}')

# cos
t0 = time.time()
y = (1 - 0.257 * x**2) / (1 + 0.191 * x**2)
t1 = time.time()
print(f'cos: {((t1-t0)*1000):.2f} ms, error vs np.cos: {np.max(np.abs(y - np.cos(x))):.2e}')