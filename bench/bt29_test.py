import numpy as np
import time

# Simulate the BT29 4x4 multiply pattern
A = np.random.randn(4,4).astype(np.float32)
B = np.random.randn(4,4).astype(np.float32)

# NumPy reference
t0 = time.time()
for _ in range(100000):
    C = A @ B
t1 = time.time()
numpy_time = (t1-t0) * 1000 / 100000
print(f'NumPy 4x4: {numpy_time:.4f} ms per multiply')

# Simulated BT29 (just the multiply logic, no overhead)
t0 = time.time()
for _ in range(100000):
    # Simulate the dot product pattern
    s = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s += A[i,k] * B[k,j]
t1 = time.time()
python_time = (t1-t0) * 1000 / 100000
print(f'Python simulate: {python_time:.4f} ms per multiply')

# Theoretical BT29 speedup estimate
ratio = numpy_time / python_time if python_time > 0 else 0
print(f'Python/BT29 ratio: {ratio:.1f}x')
print(f'Expected: 15-18x (from analysis, Python is slower than C)')