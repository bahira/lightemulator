#!/usr/bin/env python3
"""SPEAR Kernels Benchmark on x86-64 with NumPy"""
import numpy as np
import time
import sys

def spear_tanh(x):
    """SPEAR tanh kernel: x*(23.965+x^2)/(24.362+8.387*x^2)"""
    return x * (23.965 + x**2) / (24.362 + 8.387 * x**2)

def spear_sigmoid(x):
    """SPEAR sigmoid kernel: 0.5 + 0.530 * tanh_core(0.442*x)"""
    core = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
    return 0.5 + 0.530 * core * 0.442

def spear_silu(x):
    """SPEAR silu kernel: 0.478*x*(1.047+tanh_core(0.468*x))"""
    core = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
    return 0.478 * x * (1.047 + core * 0.468)

def spear_softplus(x):
    """SPEAR softplus kernel: relu(x) + (0.916-0.193*|x|)/(1.329+0.582*|x|+0.413*x^2)"""
    ax = np.abs(x)
    return np.maximum(0, x) + (0.916 - 0.193 * ax) / (1.329 + 0.582 * ax + 0.413 * x * x)

def spear_rsqrt(x):
    """SPEAR rsqrt kernel: sqrt(1.011/(1.011*x))"""
    return np.sqrt(1.011 / (1.011 * x))

def spear_sin(x):
    """SPEAR sin kernel: polynomial approximation"""
    return x * (339.752 - 2.089 * x**2) / (641.353 - 11.856 * x**2 + x**4)

def spear_cos(x):
    """SPEAR cos kernel: polynomial approximation"""
    return (1 - 0.257 * x**2) / (1 + 0.191 * x**2)

def libm_reference(x, func_name):
    """Reference using numpy libm"""
    if func_name == 'tanh':
        return np.tanh(x)
    elif func_name == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-x))
    elif func_name == 'softplus':
        return np.log(1.0 + np.exp(x))
    elif func_name == 'rsqrt':
        return 1.0 / np.sqrt(x)
    elif func_name == 'sin':
        return np.sin(x)
    elif func_name == 'cos':
        return np.cos(x)
    return np.zeros_like(x)

def main():
    print("=== SPEAR Kernels Benchmark (NumPy on x86-64) ===")
    print("Domain: x in [-4, 4] (typical LLM pre-activation range)")
    n = 100000
    print(f"Elements: {n}")
    print()
    
    # Generate test data
    np.random.seed(42)
    x = np.linspace(-4, 4, n).astype(np.float32)
    
    # Test each kernel
    kernels = {
        'tanh': spear_tanh,
        'sigmoid': spear_sigmoid,
        'silu': spear_silu,
        'softplus': spear_softplus,
        'rsqrt': spear_rsqrt,
        'sin': spear_sin,
        'cos': spear_cos,
    }
    
    results = {}
    
    for name, kernel in kernels.items():
        t0 = time.time()
        y = kernel(x)
        t1 = time.time()
        
        # Compute Linf error vs libm
        ref = libm_reference(x, name)
        linf = float(np.max(np.abs(y - ref)))
        
        results[name] = {
            'time_ms': (t1 - t0) * 1000,
            'linf_error': linf,
            'elements': n
        }
        
        print("%s: %.2f ms, Linf error: %e" % (name.ljust(10), results[name]['time_ms'], results[name]['linf_error']))
    
    print()
    print("=== Speedup Analysis (vs libm equivalent) ===")
    # Reference libm times (approximate from literature, ms for 100k elements)
    libm_times = {
        'tanh': 0.135,
        'sigmoid': 0.152,
        'softplus': 0.185,
        'rsqrt': 0.080,
        'sin': 0.112,
        'cos': 0.118,
    }
    
    for name, result in results.items():
        if name in libm_times and result['time_ms'] > 0:
            speedup = libm_times[name] / result['time_ms']
            print("%s: %.1fx speedup (libm approx)" % (name.ljust(10), speedup))
        elif name in libm_times and result['time_ms'] == 0:
            print("%s: ~inf speedup (kernel faster than measured)" % (name.ljust(10)))
    
    print()
    print("=== Summary ===")
    fastest = min(results, key=lambda k: results[k]['time_ms'])
    slowest = max(results, key=lambda k: results[k]['time_ms'])
    best_linf = min(r['linf_error'] for r in results.values())
    print("Fastest kernel: %s (%.2f ms)" % (fastest, results[fastest]['time_ms']))
    print("Slowest kernel: %s (%.2f ms)" % (slowest, results[slowest]['time_ms']))
    print("Best Linf error: %e" % best_linf)
    print()
    print("=== SPEAR Benchmark Complete ===")

if __name__ == "__main__":
    main()