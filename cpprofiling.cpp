// profile_task.cpp
#include <iostream>
#include <vector>

void printOddNumbers() {
    for (int i = 1; i <= 1'000'000; i += 2) {
        volatile int dummy = 0;
        for (int j = 0; j < 100; ++j)
            dummy += j;  // artificial delay
    }
    std::cout << "Odd numbers computed.\n";
}


void generateFibonacci() {
    const int N = 1'000'000;
    std::vector<unsigned long long> fib(N);
    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i < N; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];

        // Force CPU work to help gprof capture samples
        volatile unsigned long dummy = 0;
        for (int j = 0; j < 100; ++j)
            dummy += j;
    }

    std::cout << "Finished generating Fibonacci series.\n";
}

int main() {
    printOddNumbers();
    generateFibonacci();
    return 0;
}
