import time
import cProfile
import pstats
from memory_profiler import memory_usage
from functools import lru_cache

# --- Normal Recursion ---
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# --- Tail-recursive style (note: Python doesn't optimize tail calls) ---
def tail_factorial(n, acc=1):
    if n <= 1:
        return acc
    return tail_factorial(n - 1, acc * n)

# --- Memoized version using LRU Cache ---
@lru_cache(maxsize=None)
def memo_factorial(n):
    if n <= 1:
        return 1
    return n * memo_factorial(n - 1)

# --- Runner with Memory Profiling ---
def run_with_memory(func, depth):
    print(f"\n=== Memory Profile: {func.__name__}({depth}) ===")
    t0 = time.time()
    mem_usage = memory_usage((func, (depth,)), interval=0.1)
    t1 = time.time()
    print(f"Execution Time: {t1 - t0:.4f} s")
    print(f"Peak Memory Usage: {max(mem_usage):.2f} MiB")

# --- Runner with cProfile ---
def run_with_cprofile(func, depth):
    print(f"\n=== cProfile: {func.__name__}({depth}) ===")
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = func(depth)
        print(f"Result: {result}")
    except RecursionError:
        print("❌ RecursionError: Stack limit reached.")
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10)

# --- Main orchestrator ---
def main():
    test_depths = [500, 1000, 2000, 4000]  # Python default limit ≈ 1000
    funcs = [factorial, tail_factorial, memo_factorial]

    for f in funcs:
        for depth in test_depths:
            if f == memo_factorial:
                memo_factorial.cache_clear()
            try:
                run_with_memory(f, depth)
                run_with_cprofile(f, depth)
            except RecursionError:
                print(f"❌ RecursionError at depth {depth}")
            print("-" * 60)

if __name__ == "__main__":
    main()
