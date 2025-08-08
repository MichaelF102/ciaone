#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

// === Normal Recursive Factorial ===
long long factorial(int n) {
    if (n <= 1) return 1;
    return static_cast<long long>(n) * factorial(n - 1);
}

// === Tail-Recursive Factorial (simulated) ===
long long tail_factorial(int n, long long acc = 1) {
    if (n <= 1) return acc;
    return tail_factorial(n - 1, acc * n);
}

// === Memoized Factorial ===
map<int, long long> memo;

long long memoized_factorial(int n) {
    if (n <= 1) return 1;
    if (memo.count(n)) return memo[n];
    memo[n] = static_cast<long long>(n) * memoized_factorial(n - 1);
    return memo[n];
}

// === Profiling Wrapper ===
template <typename Func>
void profile(const string& label, Func func, int depth) {
    cout << "=== " << label << "(" << depth << ") ===\n";

    auto start = high_resolution_clock::now();
    try {
        long long result = func(depth);
        auto end = high_resolution_clock::now();
        double elapsed = duration<double>(end - start).count();

        cout << "Result: " << result << "\n";
        cout << fixed << setprecision(6)
             << "Execution Time: " << elapsed << " s\n";
    } catch (const exception& e) {
        cout << "❌ Exception: " << e.what() << "\n";
    } catch (...) {
        cout << "❌ Unknown Exception or Stack Overflow\n";
    }

    cout << "-----------------------------\n";
}

// === Main Benchmark ===
int main() {
    vector<int> depths = {100, 500, 1000, 2000, 5000, 10000};

#ifdef _OPENMP
    cout << "OpenMP Enabled. Max Threads = " << omp_get_max_threads() << "\n";
#else
    cout << "OpenMP Not Enabled.\n";
#endif

    for (int depth : depths) {
        profile("Normal Factorial", factorial, depth);

        profile("Tail Recursion", [](int n) {
            return tail_factorial(n);
        }, depth);

        memo.clear(); // reset before each memoized run
        profile("Memoized Factorial", memoized_factorial, depth);
    }

    return 0;
}
