import numpy as np
import time

def timed_extraction(a, iterations):
    t0 = time.time()
    for i in range(iterations):
        for j in range(4):
            a[:, :, :, j]
    return time.time() - t0

def timed_transpose(a, iterations):
    t0 = time.time()
    for i in range(iterations):
        np.transpose(a, (3, 0, 1, 2))
    return time.time() - t0


a = np.random.rand(1000, 20, 1000, 4)
iterations = 100000

print("Starting")

print("Extraction time:", timed_extraction(a, iterations))
print("Transpose time:", timed_transpose(a, iterations))