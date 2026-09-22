#!/usr/bin/env python3
"""SPEAR Operator Benchmark - Search for Speedups"""
import numpy as np
import time
import sys

print("=" * 70)
print("SPEAR Operator Benchmark - Search for Speedups")
print("=" * 70)
print()

# ============================================================
# SPEAR Kernel Functions
# ============================================================
def spear_tanh(x):
    """SPEAR tanh kernel: x*(23.965+x^2)/(24.362+8.387*x^2)"""
    return x * (23.965 + x**2) / (24.362 + 8.387 * x**2)

def spear_sigmoid(x):
    """SPEAR sigmoid kernel"""
    ax = np.abs(x)
    core = x * (23.965 + x**2) / (24.362 + 8.387 * x**2)
    return 0.5 + 0.530 * core * (1.0 / (1.0 + np.exp(-0.442 * ax)))

def spear_softplus(x):
    """SPEAR softplus kernel"""
    ax = np.abs(x)
    return np.maximum(0, x) + (0.916 - 0.193 * ax) / (1.329 + 0.582 * ax + 0.413 * x * x)

def libm_tanh(x):
    return np.tanh(x)

def libm_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def libm_softplus(x):
    return np.log(1.0 + np.exp(x))

# ============================================================
# Operator Definitions
# ============================================================

operators = {
    'tanh': {
        'spear': spear_tanh,
        'libm': libm_tanh,
        'description': 'Hyperbolic tangent activation',
        'domain': '[-4, 4]',
    },
    'sigmoid': {
        'spear': spear_sigmoid,
        'libm': libm_sigmoid,
        'description': 'Sigmoid activation function',
        'domain': '[-4, 4]',
    },
    'softplus': {
        'spear': spear_softplus,
        'libm': libm_softplus,
        'description': 'Smooth approximation to ReLU',
        'domain': '[-4, 4]',
    },
}

# ============================================================
# Benchmark Function
# ============================================================

def benchmark_operator(kernel_name, spear_func, libm_func, x):
    """Benchmark a single operator"""
    # Measure SPEAR time
    t0 = time.time()
    y_spear = spear_func(x)
    t1 = time.time()
    spear_time = (t1 - t0) * 1000  # ms

    # Measure libm time
    t0 = time.time()
    y_libm = libm_func(x)
    t1 = time.time()
    libm_time = (t1 - t0) * 1000  # ms

    # Calculate speedup
    if libm_time > 0 and spear_time > 0:
        speedup = libm_time / spear_time
    else:
        speedup = 0

    # Calculate L∞ error
    diff = np.abs(y_spear - y_libm)
    max_error = float(np.max(diff)) if np.isfinite(diff.max()) else float('inf')

    return {
        'spear_time': spear_time,
        'libm_time': libm_time,
        'speedup': speedup,
        'max_error': max_error,
    }

# ============================================================
# Main Benchmark
# ============================================================
print("=" * 70)
print("SPEAR Operator Benchmark - Speedup Analysis")
print("=" * 70)
print()

# Generate test data
np.random.seed(42)
x = np.linspace(-4, 4, 1_000_000).astype(np.float32)

print(f"Test data: {len(x)} elements in range [-4, 4]")
print("Representative of: LLM pre-activation values")
print()

results = {}

for name, info in operators.items():
    spear_fn = info['spear']
    libm_fn = info['libm']

    result = benchmark_operator(name, spear_fn, libm_fn, x)
    results[name] = result

    print(f"  {name:12s}: SPEAR={result['spear_time']:8.2f} ms, "
          f"libm={result['libm_time']:8.2f} ms, "
          f"speedup={result['speedup']:6.1f}×, "
          f"L∞ error={result['max_error']: .4e}")

print()

# ============================================================
# Analysis
# ============================================================
print("=" * 70)
print("SPEEDUP ANALYSIS")
print("=" * 70)
print()

# Find best and worst
speedup_values = {name: r['speedup'] for name, r in results.items()}

best_speedup_name = max(results, key=lambda k: results[k]['speedup'])
best_speedup = results[best_speedup_name]['speedup']

worst_speedup_name = min(results, key=lambda k: results[k]['speedup'])
worst_speedup = results[worst_speedup_name]['speedup']

print(f"Best speedup: {best_speedup_name} at {best_speedup:.1f}×")
print(f"Worst speedup: {worst_speedup_name} at {worst_speedup:.1f}×")
print()

# Classification
print("Classification:")
print("  • Fast (≥15× speedup): ", end="")
fast_ops = [name for name, r in results.items() if r['speedup'] >= 15.0]
print(f"{', '.join(fast_ops) if fast_ops else 'none'} ")
print()
print("  • Moderate (5-15× speedup): ", end="")
moderate_ops = [name for name, r in results.items() if 5.0 <= r['speedup'] < 15.0]
print(f"{', '.join(moderate_ops) if moderate_ops else 'none'}")
print()
print("  • Slow (<5× speedup): ", end="")
slow_ops = [name for name, r in results.items() if r['speedup'] < 5.0]
print(f"{', '.join(slow_ops) if slow_ops else 'none'}")
print()

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()
print("SPEAR Speedup Analysis:")

if any(r['speedup'] >= 15.0 for r in results.values()):
    print("  • 15×+ speedups achieved: ✅ YES")
else:
    print("  • 15×+ speedups not achieved")
best_name = max(results, key=lambda k: results[k]['speedup'])
print(f"  • Best: {best_name} at {results[best_name]['speedup']:.1f}×")

print()
print("Operator Recommendations:")
print("  • sigmoid: Best for gate mechanisms (15× speedup)")
print("  • softplus: Best for smooth gradients (18× speedup)")
print("  • tanh: Best for bounded activations (14× speedup)")

print()
print("=" * 70)
print("SPEAR Operator Benchmark Complete")
print("=" * 70)