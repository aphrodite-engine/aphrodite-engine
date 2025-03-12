# ruff: noqa
import random
import string
import timeit
from typing import Callable

import c_utils
import matplotlib.pyplot as plt
import numpy as np
# Import both implementations
from utils import extract_intermediate_diff as py_diff
from utils import find_all_indices as py_indices
from utils import find_common_prefix as py_prefix
from utils import find_common_suffix as py_suffix


def generate_random_string(length: int) -> str:
    """Generate a random string of given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits + '{}[]":', k=length))

def benchmark_function(func: Callable, *args, number: int = 10000) -> float:
    """Measure execution time of a function."""
    return timeit.timeit(lambda: func(*args), number=number) / number

def run_benchmarks():
    # Test cases with increasing sizes
    sizes = [10, 100, 1000, 10000]
    results = {
        'prefix': {'python': [], 'c': []},
        'suffix': {'python': [], 'c': []},
        'diff': {'python': [], 'c': []},
        'indices': {'python': [], 'c': []}
    }

    for size in sizes:
        # Generate test strings
        s1 = generate_random_string(size)
        s2 = s1[:size//2] + generate_random_string(size//2)  # Share prefix
        text = "hello " * (size // 5)  # For find_all_indices

        # Benchmark prefix
        py_time = benchmark_function(py_prefix, s1, s2)
        c_time = benchmark_function(c_utils.find_common_prefix, s1, s2)
        results['prefix']['python'].append(py_time)
        results['prefix']['c'].append(c_time)

        # Benchmark suffix
        py_time = benchmark_function(py_suffix, s1, s2)
        c_time = benchmark_function(c_utils.find_common_suffix, s1, s2)
        results['suffix']['python'].append(py_time)
        results['suffix']['c'].append(c_time)

        # Benchmark diff
        py_time = benchmark_function(py_diff, s1, s2)
        c_time = benchmark_function(c_utils.extract_intermediate_diff, s1, s2)
        results['diff']['python'].append(py_time)
        results['diff']['c'].append(c_time)

        # Benchmark indices
        py_time = benchmark_function(py_indices, text, "hello")
        c_time = benchmark_function(c_utils.find_all_indices, text, "hello")
        results['indices']['python'].append(py_time)
        results['indices']['c'].append(c_time)

        print(f"\nResults for size {size}:")
        for func_name in results:
            speedup = results[func_name]['python'][-1] / results[func_name]['c'][-1]
            print(f"{func_name}:")
            print(f"  Python: {results[func_name]['python'][-1]*1e6:.2f} µs")
            print(f"  C:      {results[func_name]['c'][-1]*1e6:.2f} µs")
            print(f"  Speedup: {speedup:.2f}x")

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    plots = {
        'prefix': ax1,
        'suffix': ax2,
        'diff': ax3,
        'indices': ax4
    }

    for func_name, ax in plots.items():
        ax.plot(sizes, results[func_name]['python'], 'b-', label='Python')
        ax.plot(sizes, results[func_name]['c'], 'r-', label='C')
        ax.set_title(f'{func_name} Performance')
        ax.set_xlabel('Input Size')
        ax.set_ylabel('Time (seconds)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    run_benchmarks()
