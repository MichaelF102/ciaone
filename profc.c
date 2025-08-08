#include <stdio.h>

void printOddNumbers() {
    for (int i = 1; i <= 1000000; i += 2) {
        volatile int dummy = 0;
        for (int j = 0; j < 100; ++j)
            dummy += j;  // artificial delay
    }
    printf("Odd numbers computed.\n");
}

void generateFibonacci() {
    const int N = 1000000;
    static unsigned long long fib[1000000];  // static to avoid stack overflow

    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i < N; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];

        // Artificial CPU delay
        volatile unsigned long dummy = 0;
        for (int j = 0; j < 100; ++j)
            dummy += j;
    }

    printf("Finished generating Fibonacci series.\n");
}

int main() {
    printOddNumbers();
    generateFibonacci();
    return 0;
}
